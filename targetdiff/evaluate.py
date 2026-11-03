import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[0]))
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


# 通用统计函数，放开logger打印输出
def calculate_smallest_stats(values, n, name, logger):
    if len(values) == 0:
        logger.info(f'{name} (Smallest {n}):    Mean: None Median: None')
        return None, None, []

    sorted_vals = sorted(values, reverse=False)
    smallest_n = sorted_vals[:n] if len(sorted_vals) >= n else sorted_vals

    mean_val = np.mean(smallest_n)
    median_val = np.median(smallest_n)
    logger.info(f'{name} (Smallest {n}):    Mean: %.3f Median: %.3f' % (mean_val, median_val))

    return mean_val, median_val, smallest_n


# 计算单分子Lipinski五规则达标数量 + 各指标
def calc_lipinski(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)

    rules = [
        mw <= 500,
        logp <= 5,
        hbd <= 5,
        hba <= 10,
        rot <= 10
    ]
    comply_cnt = sum(rules)
    return comply_cnt, mw, logp, hbd, hba, rot


# 计算LipE (Vina affinity kcal/mol -> pKi)
def calc_lipe(vina_affinity, clogp):
    # pKi = -affinity / 1.36
    pki = - vina_affinity / 1.36
    lipe = pki - clogp
    return lipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default='./SE_RL_0605_reward_results')
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'],
                        default='vina_score')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=10, help='输出vina_score最优的前k个结果（仅日志打印，不存入文件）')
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    vina_score_with_example_id = []

    # 新增存储Lipinski、LipE数据
    lipinski_comply = []
    all_clogp = []
    all_lipe_score = []
    all_lipe_min = []
    all_lipe_dock = []

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += 1

        sample_idx = 0
        pred_pos, pred_v = all_pred_ligand_pos[sample_idx], all_pred_ligand_v[sample_idx]
        pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

        sample_unique_id = str(example_idx)

        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
        all_atom_types += Counter(pred_atom_type)
        r_stable = analyze.check_stability(pred_pos, pred_atom_type)
        all_mol_stable += r_stable[0]
        all_atom_stable += r_stable[1]
        all_n_atom += r_stable[2]

        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        try:
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            if args.verbose:
                logger.warning('Reconstruct failed example_idx: %s' % sample_unique_id)
            continue
        n_recon_success += 1

        if '.' in smiles:
            continue
        n_complete += 1

        try:
            chem_results = scoring_func.get_chem(mol)
            clogp_mol = Descriptors.MolLogP(mol)
            comply, _, _, _, _, _ = calc_lipinski(mol)
            lipinski_comply.append(comply)
            all_clogp.append(clogp_mol)

            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(
                    mol, r['data'].ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            elif args.docking_mode in ['vina_score', 'vina_dock']:
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, r['data'].ligand_filename, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                # 计算Score Only LipE
                aff_score = score_only_results[0]['affinity']
                lipe_s = calc_lipe(aff_score, clogp_mol)
                all_lipe_score.append(lipe_s)
                # Minimize LipE
                aff_min = minimize_results[0]['affinity']
                lipe_m = calc_lipe(aff_min, clogp_mol)
                all_lipe_min.append(lipe_m)

                if args.docking_mode == 'vina_dock':
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results['dock'] = docking_results
                    aff_dock = docking_results[0]['affinity']
                    lipe_d = calc_lipe(aff_dock, clogp_mol)
                    all_lipe_dock.append(lipe_d)
            else:
                vina_results = None

            n_eval_success += 1
        except Exception as e:
            if args.verbose:
                logger.warning('Evaluation failed for example_idx: %s err: %s' % (sample_unique_id, str(e)))
            continue

        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist
        success_pair_dist += pair_dist
        success_atom_types += Counter(pred_atom_type)

        current_result = {
            'mol': mol,
            'smiles': smiles,
            'ligand_filename': r['data'].ligand_filename,
            'pred_pos': pred_pos,
            'pred_v': pred_v,
            'chem_results': chem_results,
            'vina': vina_results,
            'example_idx': example_idx
        }
        results.append(current_result)

        if args.docking_mode in ['vina_score', 'vina_dock']:
            vina_score_val = vina_results['score_only'][0]['affinity']
            vina_score_with_example_id.append((vina_score_val, example_idx))

    # 输出样本路径
    logger.info(f"Sample Path: {args.sample_path}")
    # 有效性字典相关全部注释
    # fraction_mol_stable = all_mol_stable / num_samples
    # fraction_atm_stable = all_atom_stable / all_n_atom
    # fraction_recon = n_recon_success / num_samples
    # fraction_eval = n_eval_success / num_samples
    # fraction_complete = n_complete / num_samples
    # validity_dict = {
    #     'mol_stable': fraction_mol_stable,
    #     'atm_stable': fraction_atm_stable,
    #     'recon_success': fraction_recon,
    #     'eval_success': fraction_eval,
    #     'complete': fraction_complete
    # }
    # print_dict(validity_dict, logger)

    # 键长JS计算与打印全部注释
    # c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    # c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    # logger.info('JS bond distances of complete mols: ')
    # print_dict(c_bond_length_dict, logger)

    # success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    # success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    # print_dict(success_js_metrics, logger)

    # atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    # logger.info('Atom type JS: %.4f' % atom_type_js)

    # 注释绘图代码，消除未定义变量报错
    # if args.save:
    #     eval_bond_length.plot_distance_hist(success_pair_length_profile,
    #                                         metrics=success_js_metrics,
    #                                         save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (n_recon_success, n_complete, len(results)))

    status = {'attempted_pockets': num_examples, 'reconstructed': n_recon_success,
              'connected': n_complete, 'evaluated': len(results)}
    with open(os.path.join(result_path, 'evaluation_status.json'), 'w') as f:
        __import__('json').dump(status, f, indent=2)
    if not results:
        raise RuntimeError('No molecules were successfully evaluated. See evaluation_status.json and rerun with --verbose True for failure details.')

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    # 仅保留4行目标输出
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))

    sorted_vina_data = {
        'vina_score_only': [],
        'vina_min': [],
        'vina_dock': [],
        'vina_score_only_smallest_50': [],
        'vina_score_only_smallest_20': [],
        'vina_score_only_smallest_10': [],
        'vina_min_smallest_50': [],
        'vina_min_smallest_20': [],
        'vina_min_smallest_10': [],
        'vina_dock_smallest_50': [],
        'vina_dock_smallest_20': [],
        'vina_dock_smallest_10': []
    }

    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        sorted_vina_data['qvina'] = sorted(vina, reverse=False)
        # 屏蔽TopN打印
        # calculate_smallest_stats(vina, 50, 'Vina (QVina)', logger)
        # calculate_smallest_stats(vina, 20, 'Vina (QVina)', logger)
        # calculate_smallest_stats(vina, 10, 'Vina (QVina)', logger)

    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]

        # 仅打印Vina核心两行
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))

        # 注释所有TopN打印调用，屏蔽Smallest50/20/10日志
        # score_50_mean, score_50_median, score_50_vals = calculate_smallest_stats(vina_score_only, 50, 'Vina Score', logger)
        # score_20_mean, score_20_median, score_20_vals = calculate_smallest_stats(vina_score_only, 20, 'Vina Score', logger)
        # score_10_mean, score_10_median, score_10_vals = calculate_smallest_stats(vina_score_only, 10, 'Vina Score', logger)

        # min_50_mean, min_50_median, min_50_vals = calculate_smallest_stats(vina_min, 50, 'Vina Min', logger)
        # min_20_mean, min_20_median, min_20_vals = calculate_smallest_stats(vina_min, 20, 'Vina Min', logger)
        # min_10_mean, min_10_median, min_10_vals = calculate_smallest_stats(vina_min, 10, 'Vina Min', logger)

        sorted_vina_data['vina_score_only'] = sorted(vina_score_only, reverse=False)
        sorted_vina_data['vina_min'] = sorted(vina_min, reverse=False)
        # sorted_vina_data['vina_score_only_smallest_50'] = score_50_vals
        # sorted_vina_data['vina_score_only_smallest_20'] = score_20_vals
        # sorted_vina_data['vina_score_only_smallest_10'] = score_10_vals
        # sorted_vina_data['vina_min_smallest_50'] = min_50_vals
        # sorted_vina_data['vina_min_smallest_20'] = min_20_vals
        # sorted_vina_data['vina_min_smallest_10'] = min_10_vals

        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            # dock_50_mean, dock_50_median, dock_50_vals = calculate_smallest_stats(vina_dock, 50, 'Vina Dock', logger)
            # dock_20_mean, dock_20_median, dock_20_vals = calculate_smallest_stats(vina_dock, 20, 'Vina Dock', logger)
            # dock_10_mean, dock_10_median, dock_10_vals = calculate_smallest_stats(vina_dock, 10, 'Vina Dock', logger)
            sorted_vina_data['vina_dock'] = sorted(vina_dock, reverse=False)
            # sorted_vina_data['vina_dock_smallest_50'] = dock_50_vals
            # sorted_vina_data['vina_dock_smallest_20'] = dock_20_vals
            # sorted_vina_data['vina_dock_smallest_10'] = dock_10_vals

    # 屏蔽LipE、环分布、TopK排名打印
    # if len(all_lipe_score) > 0:
    #     logger.info('LipE (Vina Score): Mean: %.3f Median: %.3f' % (np.mean(all_lipe_score), np.median(all_lipe_score)))
    # if len(all_lipe_min) > 0:
    #     logger.info('LipE (Vina Min):  Mean: %.3f Median: %.3f' % (np.mean(all_lipe_min), np.median(all_lipe_min)))
    # if args.docking_mode == 'vina_dock' and len(all_lipe_dock) > 0:
    #     logger.info('LipE (Vina Dock): Mean: %.3f Median: %.3f' % (np.mean(all_lipe_dock), np.median(all_lipe_dock)))
    # print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)
    # if args.docking_mode in ['vina_score', 'vina_dock'] and vina_score_with_example_id:
    #     sorted_vina_score_with_id = sorted(vina_score_with_example_id, key=lambda x: x[0], reverse=False)
    #     top_k = args.top_k
    #     top_vina_results = sorted_vina_score_with_id[:top_k]
    #     logger.info(f'\n========== Vina Score 最优前{len(top_vina_results)}个结果（数值越小越优） ==========')
    #     for rank, (score, example_id) in enumerate(top_vina_results, 1):
    #         logger.info(f'排名{rank:2d} | Example ID: {example_id:6d} | Vina Score: {score:.4f}')

    if args.save:
        torch.save({
            'bond_length': all_bond_dist,
            'all_results': results,
            'sorted_vina_data': sorted_vina_data,
            'lipinski_comply': lipinski_comply,
            'lipe_score': all_lipe_score,
            'lipe_min': all_lipe_min,
            'lipe_dock': all_lipe_dock
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))