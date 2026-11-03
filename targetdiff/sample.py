import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[0]))

# 你的导入（完全不变）
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.cref import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from rejection.train import timestep_embedding, EnhancedGNNClassifier



def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples,num_particles,extra_proteins,r_model,normalization_params_path,gamma1,gamma2,IRIE_step,Rejection_threshold,last_rejection,ε_v,ε_q,ε_s,batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',evaluate='vina',
                            sample_num_atoms='prior', window=20):
    

    # 直接使用外部传入的10个蛋白质数据构建batch_protein_features
    batch_protein_features = []
    for protein_data in extra_proteins:  # 遍历10个蛋白质数据
        # 提取当前蛋白质的pos、v和batch信息

        protein_pos = protein_data.protein_pos
        protein_v = protein_data.protein_atom_feature.float()
        
        # 复制坐标和特征num_samples次（保持原加工逻辑）
        repeated_pos = protein_pos.repeat(num_samples, 1).cpu()  # [n*num_samples, 3]
        repeated_v = protein_v.repeat(num_samples, 1).cpu()      # [n*num_samples, feat_dim]
        
        # 创建批次索引：n个0, n个1, ..., n个(num_samples-1)
        batch_indices = []
        for i in range(num_samples):
            batch_indices.append(torch.full((protein_pos.size(0),), i, dtype=torch.long))
        repeated_batch = torch.cat(batch_indices, dim=0)  # 形状: [12*n]
        batch_protein_features.append((repeated_pos, repeated_v, repeated_batch))
    

    print(f'随机挑选的蛋白质特征数量: {len(batch_protein_features)}')
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    init_ligand_pos_list , init_ligand_v_list = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            for i in range(num_particles):
                center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                batch_center_pos = center_pos[batch_ligand]
                init_ligand_pos_alone = batch_center_pos + torch.randn_like(batch_center_pos)
                init_ligand_pos_list.append(init_ligand_pos_alone)

                # init ligand v
                if pos_only:
                    init_ligand_v_alone = batch.ligand_atom_feature_full
                    init_ligand_v_list.append(init_ligand_v_alone)
                    
                else:
                    uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                    init_ligand_v_alone = log_sample_categorical(uniform_logits)
                    init_ligand_v_list.append(init_ligand_v_alone)

            init_ligand_pos = torch.stack(init_ligand_pos_list, dim=0)  # [num_particles, num_atoms, 3]
            init_ligand_v = torch.stack(init_ligand_v_list, dim=0)
            init_ligand_pos = init_ligand_pos.view(-1, 3)
            init_ligand_v = init_ligand_v.view(-1) 
            num_atoms = len(batch_ligand)
            num_protein_atoms = len(batch_protein)
            batch_ligand = torch.arange(num_particles, device=device).repeat_interleave(num_atoms)  # [num_particles * num_atoms]
            batch_protein = torch.arange(num_particles, device=device).repeat_interleave(num_protein_atoms)  # [num_particles * num_protein_atoms]
            batch.protein_pos = batch.protein_pos.repeat(num_particles, 1)
            batch.protein_atom_feature = batch.protein_atom_feature.repeat(num_particles, 1)
                 
            r = model.sample_diffusion(
                r_model=r_model,
                normalization_params_path=normalization_params_path,
                data=data,
                ligand_num_atoms=ligand_num_atoms,
                num_particles=num_particles,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                other_protein_features=batch_protein_features,
                gamma1=gamma1,
                gamma2=gamma2,
                IRIE_step=IRIE_step,
                Rejection_threshold=Rejection_threshold,
                last_rejection=last_rejection,    
                evaluate=evaluate,
                ε_v=ε_v,
                ε_q=ε_q,
                ε_s=ε_s,
                window=window
                
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./configs/sampling.yml')
    parser.add_argument('-i', '--data_id', type=int,default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--start_index', type=int,default=0, help='主数据ID起始索引')
    parser.add_argument('--end_index', type=int,default=99, help='主数据ID结束索引（包含）')
    parser.add_argument('--num_particles', type=int,default=1, help='ligand初始粒子数')
    parser.add_argument('--num_extra_proteins', type=int, default=10, help='额外蛋白质数量')
    parser.add_argument('--max_data_id', type=int, default=99989, help='蛋白质ID最大值')
    parser.add_argument('--rejection_path', type=str, default='./rejection/best_model.pt')
    parser.add_argument('--normalization_params_path', type=str, default='./rejection/normalization_params.npz')
    parser.add_argument('--gamma1', type=int, default=5.5, help='gamma 1参数')
    parser.add_argument('--gamma2', type=int, default=0, help='gamma 2参数')
    parser.add_argument('--IRIE_step', type=int, default=0, help='IRIE迭代步数,0表示不使用IRIE')
    parser.add_argument('--Rejection_threshold', type=float, default=0.5, help='拒绝采样阈值(修正为float更合理)')
    parser.add_argument('--last_rejection', type=int, default=730 , help='最后拒绝采样次数,大于1000表示不使用拒绝采样')
    parser.add_argument('--evaluate', type=str,default='vina' ,choices=['vina', 'QED','SA','None'], help='是否在采样时进行评估')
    parser.add_argument('--ε_v', type=float, default=-5.3, help='Vina极小化指标阈值')
    parser.add_argument('--ε_q', type=float, default=0.4, help='QED极大化指标阈值')
    parser.add_argument('--ε_s', type=float, default=0.5, help='SA极大化指标阈值')
    parser.add_argument('--window', type=int, default=20, help='采样配体原子数的方式')
    parser.add_argument('--data_root', default=None, help='Override dataset location embedded in the backbone checkpoint')
    parser.add_argument('--split_path', default=None, help='Override checkpoint dataset split location')
    args = parser.parse_args()


    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    if args.data_root is not None:
        ckpt['config'].data.path = args.data_root
    if args.split_path is not None:
        ckpt['config'].data.split = args.split_path

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # 加载数据集（重点：使用训练集获取额外蛋白质）
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'数据集加载完成（训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}）')

    # 生成10个随机data_id（0到max_data_id之间）
    random.seed(config.sample.seed)  # 固定种子确保可复现
    extra_data_ids = random.sample(range(args.max_data_id), args.num_extra_proteins)
    logger.info(f'随机选择的蛋白质ID: {extra_data_ids}')

    # 加载这10个蛋白质数据（从训练集）
    extra_proteins = []
    for data_id in extra_data_ids:
        if data_id >= len(train_set):
            logger.warning(f'ID {data_id} 超出训练集范围，跳过')
            continue
        protein_data = train_set[data_id]
        extra_proteins.append(protein_data)
    # 确保最终有10个有效蛋白质
    if len(extra_proteins) < args.num_extra_proteins:
        logger.warning(f'实际获取 {len(extra_proteins)} 个蛋白质（不足10个）')


    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    data_ids = list(range(args.start_index, args.end_index + 1))
    logger.info(f'主数据ID范围: {data_ids}')
    r_model = EnhancedGNNClassifier(
        ligand_dim=4,          # 配体特征维度（3位置+1速度）
        protein_dim=30,        # 蛋白质特征维度（3位置+27其他特征）
        hidden_dim=128,        # 隐藏层维度（训练时用的128）
        num_layers=5,          # GNN层数（训练时main函数中设的5）
        max_time_steps=1000    # 最大时间步（与训练参数一致）
        )
    r_model.load_state_dict(torch.load(args.rejection_path, map_location=args.device))
    r_model.to(args.device)
    r_model.eval()  # 切换到评估模式（关闭dropout等）
    for data_id in data_ids:
        logger.info(f'开始处理主数据ID: {data_id}')

        data = test_set[data_id]
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,args.num_particles,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            IRIE_step=args.IRIE_step,
            Rejection_threshold=args.Rejection_threshold,
            last_rejection=args.last_rejection,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            extra_proteins=extra_proteins,
            normalization_params_path=args.normalization_params_path,
            r_model=r_model,
            evaluate=args.evaluate,
            ε_v=args.ε_v,
            ε_q=args.ε_q,
            ε_s=args.ε_s,
            window=args.window
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list
        }
        logger.info(f'主数据ID {data_id} 采样完成!')


        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))






