# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import warnings

# 屏蔽Vina exhaustiveness低线程警告
warnings.filterwarnings("ignore", message="At low exhaustiveness, it may be impossible to utilize all CPUs.")

import numpy as np
from rdkit import RDLogger
from rdkit.Chem import Descriptors
import torch
from tqdm.auto import tqdm
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from glob import glob
from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from utils.evaluation import eval_bond_length
from utils.transforms import get_atomic_number_from_index
import utils.transforms as trans
from rdkit import Chem
from utils.reconstruct import reconstruct_from_generated_with_bond

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[0]))

# ===================== 绘图全局字体配置（和你旧绘图代码完全一致） =====================
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8


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


# 【修复】百分比截取统计函数
def calculate_smallest_stats(values, total_valid_num, percent, name, logger):
    if len(values) == 0 or total_valid_num <= 0:
        logger.info(f'{name} (Smallest {percent}%):    Mean: None Median: None')
        return None, None, []
    take_n = int(np.ceil(total_valid_num * percent / 100))
    take_n = min(take_n, len(values))
    sorted_vals = sorted(values, reverse=False)
    smallest_n = sorted_vals[:take_n]
    mean_val = np.mean(smallest_n)
    median_val = np.median(smallest_n)
    # 统一只用f-string，彻底消除混合格式化冲突
    logger.info(
        f"{name} (Smallest {percent}%, take {take_n}/{total_valid_num}):    Mean: {mean_val:.3f} Median: {median_val:.3f}"
    )
    return mean_val, median_val, smallest_n

# 单分子Lipinski五规则，返回达标数量
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
    return comply_cnt, logp


# 计算LipE = pKi - cLogP，pKi = -affinity / 1.36
def calc_lipe(vina_affinity, clogp):
    pki = - vina_affinity / 1.36
    lipe = pki - clogp
    return lipe


# ===================== 新增：绘图函数（完全复用你之前的散点图逻辑） =====================
def plot_metrics_vs_atom_count(all_valid_data, save_path, plot_filename, logger):
    data = []
    for mol_dict in all_valid_data:
        mol = mol_dict['mol']
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 0 or n_atoms > 40:
            continue
        sa = mol_dict['chem_results']['sa']
        qed = mol_dict['chem_results']['qed']
        vina_val = None
        if mol_dict['vina'] is not None and 'score_only' in mol_dict['vina']:
            vina_val = mol_dict['vina']['score_only'][0]['affinity']
        elif mol_dict['vina'] is not None and isinstance(mol_dict['vina'], list):
            vina_val = mol_dict['vina'][0]['affinity']
        if vina_val is None or vina_val > 0:
            continue
        data.append({
            'n_atoms': n_atoms,
            'sa': sa,
            'qed': qed,
            'vina': vina_val
        })
    if len(data) == 0:
        logger.warning('No valid data for scatter plot, skip drawing.')
        return None
    atom_counts = np.array([d['n_atoms'] for d in data])
    sa_scores = np.array([d['sa'] for d in data])
    qed_scores = np.array([d['qed'] for d in data])
    vina_scores = np.array([d['vina'] for d in data])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    ax1 = axes[0]
    ax1.scatter(atom_counts, sa_scores, alpha=0.8, s=40, c='#1A5F7A', edgecolors='none', linewidths=0.5)
    ax1.set_xlabel('# Atoms', fontsize=10)
    ax1.set_ylabel('SA', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('SA', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 40)
    ax2 = axes[1]
    ax2.scatter(atom_counts, qed_scores, alpha=0.8, s=40, c='#802659', edgecolors='none', linewidths=0.5)
    ax2.set_xlabel('# Atoms', fontsize=10)
    ax2.set_ylabel('QED', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('QED', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 40)
    ax3 = axes[2]
    ax3.scatter(atom_counts, vina_scores, alpha=0.8, s=40, c='#D47500', edgecolors='none', linewidths=0.5)
    ax3.set_xlabel('# Atoms', fontsize=10)
    ax3.set_ylabel('Vina Score', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_title('Vina Score', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 40)
    plt.tight_layout()
    img_path = os.path.join(save_path, f'{plot_filename}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f'Scatter plot saved at {img_path}, valid points: {len(atom_counts)}')
    return img_path


def eval_single_datapoint(index, id, args):
    if isinstance(index, dict):
        index = [index]

    ligand_filename = index[0]['ligand_filename']
    num_samples = len(index[:100])
    results = []
    n_eval_success = 0
    all_pair_dist, all_bond_dist = [], []

    batch_lipinski = []
    batch_clogp = []
    batch_aff_score = []
    batch_aff_min = []
    batch_aff_dock = []
    # 新增：保存本批完整分子数据，用于绘图
    batch_mol_draw_data = []

    for sample_idx, sample_dict in enumerate(index[:num_samples]):
        if sample_dict['mol'] is None:
            try:
                pred_atom_type = trans.get_atomic_number_from_index(sample_dict['pred_v'], mode='basic')
                mol = reconstruct_from_generated_with_bond(
                    xyz=sample_dict['pred_pos'],
                    atomic_nums=pred_atom_type,
                    bond_index=sample_dict['pred_bond_index'],
                    bond_type=sample_dict['pred_bond_type']
                )
                smiles = Chem.MolToSmiles(mol)
            except:
                mol, smiles = None, None
        else:
            mol = sample_dict['mol']
            smiles = sample_dict['smiles']

        if mol is None or '.' in smiles:
            continue

        lip_comply, clogp = calc_lipinski(mol)
        batch_lipinski.append(lip_comply)
        batch_clogp.append(clogp)

        chem_results = None
        vina_results = None
        aff_s, aff_m, aff_d = None, None, None
        try:
            chem_results = scoring_func.get_chem(mol)
            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            elif args.docking_mode == 'vina':
                vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run(mode='dock')
            elif args.docking_mode in ['vina_full', 'vina_score']:
                vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol), ligand_filename, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                aff_s = score_only_results[0]['affinity']
                aff_m = minimize_results[0]['affinity']
                batch_aff_score.append(aff_s)
                batch_aff_min.append(aff_m)
                if args.docking_mode == 'vina_full':
                    dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results.update({'dock': dock_results})
                    aff_d = dock_results[0]['affinity']
                    batch_aff_dock.append(aff_d)
            elif args.docking_mode == 'none':
                vina_results = None
            else:
                raise NotImplementedError
            n_eval_success += 1
        except Exception as e:
            continue

        pred_pos, pred_v = sample_dict['pred_pos'], sample_dict['pred_v']
        pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist
        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist

        res_item = {
            **sample_dict,
            'chem_results': chem_results,
            'vina': vina_results
        }
        results.append(res_item)
        # 收集绘图所需完整分子数据
        batch_mol_draw_data.append(res_item)

    return (
        results, all_pair_dist, all_bond_dist,
        batch_lipinski, batch_clogp,
        batch_aff_score, batch_aff_min, batch_aff_dock,
        batch_mol_draw_data
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_file', type=str)
    parser.add_argument('-n', '--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--docking_mode', type=str, default='vina_full',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--aggregate_meta', type=eval, default=False)
    parser.add_argument('--percentiles', nargs='+', type=int, default=[10, 20, 50],
                        help='截取最优Vina分子百分比，默认10 20 50')
    # 新增绘图控制参数
    parser.add_argument('--save_plot', type=eval, default=True, choices=[True, False],
                        help='是否输出SA/QED/Vina vs 原子数散点图')
    parser.add_argument('--plot_filename', type=str, default='decomp_metrics_atom',
                        help='散点图输出文件名（无需加.png）')
    args = parser.parse_args()

    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(f"===== Start evaluating target folder: {args.meta_file} =====")
    logger.info(f"Percentiles for top Vina subset: {args.percentiles}")
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    if args.aggregate_meta:
        meta_file_list = sorted(glob(os.path.join(args.meta_file, '*/result.pt')))
        test_index = []
        for f in tqdm(meta_file_list, desc='Load meta files'):
            test_index.append(torch.load(f))
    else:
        test_index = torch.load(args.meta_file)
        if isinstance(test_index[0], dict):
            test_index = [test_index]

    testset_results = []
    testset_pair_dist, testset_bond_dist = [], []
    all_lipinski = []
    all_clogp = []
    all_aff_score = []
    all_aff_min = []
    all_aff_dock = []
    # 全局绘图数据容器：存放所有完整分子dict
    global_draw_mol_data = []

    with Pool(args.num_workers) as p:
        task_args = zip(test_index[:args.eval_num_examples], list(range(args.eval_num_examples)))
        for ret in tqdm(p.starmap(partial(eval_single_datapoint, args=args), task_args),
                        total=args.eval_num_examples, desc='Overall Eval'):
            r, pd, bd, lip, clogp, affs, affm, affd, draw_batch = ret
            testset_results.append(r)
            testset_pair_dist += pd
            testset_bond_dist += bd
            all_lipinski.extend(lip)
            all_clogp.extend(clogp)
            all_aff_score.extend(affs)
            all_aff_min.extend(affm)
            all_aff_dock.extend(affd)
            global_draw_mol_data.extend(draw_batch)

    if args.result_path:
        torch.save(testset_results, os.path.join(args.result_path, f'eval_all.pt'))

    status = {'attempted_pockets': len(test_index[:args.eval_num_examples]),
              'evaluated': sum(len(batch) for batch in testset_results)}
    if args.result_path:
        with open(os.path.join(args.result_path, 'evaluation_status.json'), 'w') as f:
            __import__('json').dump(status, f, indent=2)
    if status['evaluated'] == 0:
        raise RuntimeError('No molecules were successfully evaluated. Rerun with --verbose True and inspect evaluation_status.json.')

    qed = [x['chem_results']['qed'] for r in testset_results for x in r]
    sa = [x['chem_results']['sa'] for r in testset_results for x in r]
    total_valid_vina = len(all_aff_score)

    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if len(all_lipinski) > 0:
        logger.info('Lipinski Rule Comply Count: Mean: %.3f Median: %.3f'
                    % (np.mean(all_lipinski), np.median(all_lipinski)))

    if args.docking_mode in ['vina_full', 'vina_score']:
        logger.info('Vina Score:  Mean: %.3f Median: %.3f'
                    % (np.mean(all_aff_score), np.median(all_aff_score)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f'
                    % (np.mean(all_aff_min), np.median(all_aff_min)))
        if args.docking_mode == 'vina_full' and len(all_aff_dock) > 0:
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f'
                        % (np.mean(all_aff_dock), np.median(all_aff_dock)))

        for pct in args.percentiles:
            calculate_smallest_stats(all_aff_score, total_valid_vina, pct, "Vina Score", logger)
        for pct in args.percentiles:
            calculate_smallest_stats(all_aff_min, total_valid_vina, pct, "Vina Min", logger)
        if args.docking_mode == 'vina_full' and len(all_aff_dock) > 0:
            for pct in args.percentiles:
                calculate_smallest_stats(all_aff_dock, total_valid_vina, pct, "Vina Dock", logger)

        lipe_score = [calc_lipe(aff, clogp) for aff, clogp in zip(all_aff_score, all_clogp)]
        lipe_min = [calc_lipe(aff, clogp) for aff, clogp in zip(all_aff_min, all_clogp)]
        logger.info('LipE (Vina Score): Mean: %.3f Median: %.3f'
                    % (np.mean(lipe_score), np.median(lipe_score)))
        logger.info('LipE (Vina Min):  Mean: %.3f Median: %.3f'
                    % (np.mean(lipe_min), np.median(lipe_min)))
        if args.docking_mode == 'vina_full' and len(all_aff_dock) > 0:
            lipe_dock = [calc_lipe(aff, clogp) for aff, clogp in zip(all_aff_dock, all_clogp)]
            logger.info('LipE (Vina Dock): Mean: %.3f Median: %.3f'
                        % (np.mean(lipe_dock), np.median(lipe_dock)))

    # ===================== 执行绘图 =====================
    if args.save_plot and args.result_path is not None and len(global_draw_mol_data) > 0:
        plot_metrics_vs_atom_count(
            all_valid_data=global_draw_mol_data,
            save_path=args.result_path,
            plot_filename=args.plot_filename,
            logger=logger
        )