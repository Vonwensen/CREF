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
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.cref import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from graphbap.bapnet import BAPNet

from rejection.train import timestep_embedding, EnhancedGNNClassifier


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, num_samples, num_particles, extra_proteins, r_model,
                            threshold_change, normalization_params_path,
                            gamma1, gamma2, IRIE_step, Rejection_threshold, last_rejection,
                            batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', net_cond=None, cond_dim=128, ε_v=-5, ε_q=0.6, ε_s=0.6):
    # 构造额外蛋白质特征组 + 统一设备到GPU
    batch_protein_features = []
    for protein_data in extra_proteins:
        protein_pos = protein_data.protein_pos
        protein_v = protein_data.protein_atom_feature.float()

        repeated_pos = protein_pos.repeat(num_particles, 1).cpu()
        repeated_v = protein_v.repeat(num_particles, 1).cpu()

        batch_indices = []
        for i in range(num_particles):
            batch_indices.append(torch.full((protein_pos.size(0),), i, dtype=torch.long))
        repeated_batch = torch.cat(batch_indices, dim=0)
        batch_protein_features.append((repeated_pos, repeated_v, repeated_batch))

    # 设备迁移：CPU -> GPU
    batch_protein_features = [
        (pos.to(device), v.to(device), b.to(device))
        for pos, v, b in batch_protein_features
    ]

    print(f'随机挑选的蛋白质特征数量: {len(batch_protein_features)}')
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0

    for i in tqdm(range(num_batch)):
        # 每个batch重置列表，防止数据累积/内存泄漏
        init_ligand_pos_list, init_ligand_v_list = [], []
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

            # 初始化配体坐标 & 原子类型
            for _ in range(num_particles):
                center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                batch_center_pos = center_pos[batch_ligand]
                init_ligand_pos_alone = batch_center_pos + torch.randn_like(batch_center_pos)
                init_ligand_pos_list.append(init_ligand_pos_alone)

                if pos_only:
                    init_ligand_v_alone = batch.ligand_atom_feature_full
                    init_ligand_v_list.append(init_ligand_v_alone)
                else:
                    uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                    init_ligand_v_alone = log_sample_categorical(uniform_logits)
                    init_ligand_v_list.append(init_ligand_v_alone)

            init_ligand_pos = torch.stack(init_ligand_pos_list, dim=0)
            init_ligand_v = torch.stack(init_ligand_v_list, dim=0)
            init_ligand_pos = init_ligand_pos.view(-1, 3)
            init_ligand_v = init_ligand_v.view(-1)

            num_atoms = len(batch_ligand)
            num_protein_atoms = len(batch_protein)
            batch_ligand = torch.arange(num_particles, device=device).repeat_interleave(num_atoms)
            batch_protein = torch.arange(num_particles, device=device).repeat_interleave(num_protein_atoms)

            # 重复蛋白质特征以匹配粒子数
            batch.protein_pos = batch.protein_pos.repeat(num_particles, 1)
            batch.protein_atom_feature = batch.protein_atom_feature.repeat(num_particles, 1)

            # 完整调用迁移后的 sample_diffusion（补齐所有参数）
            r = model.sample_diffusion(
                r_model=r_model,
                data=data,
                num_particles=num_particles,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                other_protein_features=batch_protein_features,
                gamma1=gamma1,
                gamma2=gamma2,
                IRIE_step=IRIE_step,
                Rejection_threshold=Rejection_threshold,
                last_rejection=last_rejection,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                net_cond=net_cond,
                cond_dim=cond_dim,
                threshold_change=threshold_change,
                normalization_params_path=normalization_params_path,
                ε_v=ε_v,
                ε_q=ε_q,
                ε_s=ε_s
            )

            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']

            # 解包坐标轨迹
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            all_pred_pos_traj += [p for p in all_step_pos]

            # 解包原子类型
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
    root_dir = str(__import__("pathlib").Path(__file__).resolve().parents[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=root_dir + '/configs/sampling.yml')
    parser.add_argument('--train_config', type=str, default=root_dir + '/configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=99)
    parser.add_argument('--num_particles', type=int, default=1, help='ligand初始粒子数')
    parser.add_argument('--num_extra_proteins', type=int, default=10, help='额外蛋白质数量')
    parser.add_argument('--max_data_id', type=int, default=99989, help='蛋白质ID最大值')
    parser.add_argument('--rejection_path', type=str, default='./rejection/best_model.pt')
    parser.add_argument('--normalization_params_path', type=str, default='./rejection/normalization_params.npz')
    parser.add_argument('--Threshold_change', type=float, default=0, help='拒绝采样系数')
    parser.add_argument('--gamma1', type=float, default=5.5, help='IPdiff引导系数')
    parser.add_argument('--gamma2', type=float, default=0.0, help='ZigZag反演系数')
    parser.add_argument('--IRIE_step', type=int, default=10000, help='IRIE分支启动时间步')
    parser.add_argument('--Rejection_threshold', type=float, default=0.5, help='拒绝采样置信度阈值')
    parser.add_argument('--last_rejection', type=int, default=830, help='拒绝采样初始计数')
    parser.add_argument('--ε_v', type=float, default=-5, help='Vina极小化指标阈值')
    parser.add_argument('--ε_q', type=float, default=0.6, help='QED极大化指标阈值')
    parser.add_argument('--ε_s', type=float, default=0.6, help='SA极大化指标阈值')

    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # 加载配置 & 固定随机种子
    config = misc.load_config(args.config)
    train_config = misc.load_config(args.train_config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # 加载模型权重
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {train_config}")

    # 数据变换
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # 加载数据集
    dataset, subsets = get_dataset(config=train_config.data, transform=transform)
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # 随机选取额外蛋白质
    random.seed(config.sample.seed)
    extra_data_ids = random.sample(range(args.max_data_id), args.num_extra_proteins)
    logger.info(f'随机选择的蛋白质ID: {extra_data_ids}')

    extra_proteins = []
    for data_id in extra_data_ids:
        if data_id >= len(train_set):
            logger.warning(f'ID {data_id} 超出训练集范围，跳过')
            continue
        protein_data = train_set[data_id]
        extra_proteins.append(protein_data)
    if len(extra_proteins) < args.num_extra_proteins:
        logger.warning(f'实际获取 {len(extra_proteins)} 个蛋白质（不足{args.num_extra_proteins}个）')

    # 加载主模型
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    # 加载BAP条件网络
    net_cond = BAPNet(
        ckpt_path=train_config.net_cond.ckpt_path,
        hidden_nf=train_config.net_cond.hidden_dim
    ).to(args.device)

    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # 加载拒绝采样分类模型
    r_model = EnhancedGNNClassifier(
        ligand_dim=4,
        protein_dim=30,
        hidden_dim=128,
        num_layers=5,
        max_time_steps=1000
    )
    r_model.load_state_dict(torch.load(args.rejection_path, map_location=args.device))
    r_model.to(args.device)
    r_model.eval()

    # 遍历测试集执行采样
    data_ids = list(range(args.start_index, args.end_index + 1))
    logger.info(f'主数据ID范围: {data_ids}')
    for data_id in data_ids:
        logger.info(f'开始处理主数据ID: {data_id}')
        data = test_set[data_id]

        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model=model,
            num_samples=config.sample.num_samples,
            num_particles=args.num_particles,
            extra_proteins=extra_proteins,
            r_model=r_model,
            threshold_change=args.Threshold_change,
            normalization_params_path=args.normalization_params_path,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
            IRIE_step=args.IRIE_step,
            Rejection_threshold=args.Rejection_threshold,
            last_rejection=args.last_rejection,
            batch_size=args.batch_size,
            device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            net_cond=net_cond,
            cond_dim=train_config.model.cond_dim,
            ε_v=args.ε_v,
            ε_q=args.ε_q,
            ε_s=args.ε_s
        )

        # 保存结果
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list
        }
        logger.info('Sample done!')
        print('sampled data_id: ', data_id)

        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))