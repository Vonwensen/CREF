
import numpy as np
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from tqdm.auto import tqdm
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1])) 
import utils.guidance_funcs as guidance
from models.encoders import get_refine_net
from models.common import compose_context, compose_context_with_prior, ShiftedSoftplus, GaussianSmearing, \
    to_torch_const, extract
from models.transitions import cosine_beta_schedule, get_beta_schedule, DiscreteTransition, index_to_log_onehot, \
    log_sample_categorical


from utils import misc, reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask
from rdkit import Chem
from utils.evaluation import scoring_func

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        if ligand_pos is not None:
            ligand_pos = ligand_pos - offset[batch_ligand]
        # print('offset: ', offset)
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class DecompScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, num_classes,
                 prior_atom_types=None, prior_bond_types=None):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']

        self.add_prior_node = getattr(config, 'add_prior_node', False)
        self.bond_diffusion = getattr(config, 'bond_diffusion', False)
        self.bond_net_type = getattr(config, 'bond_net_type', 'mlp')
        self.center_prox_loss = getattr(config, 'center_prox_loss', False)
        self.armsca_prox_loss = getattr(config, 'armsca_prox_loss', False)
        self.clash_loss = getattr(config, 'clash_loss', False)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        print(f'Loss pos mode {self.loss_pos_type} applied!')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))
        self.pos_score_coef = to_torch_const(betas / np.sqrt(alphas))

        # atom / bond type transition
        self.num_classes = num_classes
        self.num_bond_classes = getattr(config, 'num_bond_classes', 1)

        self.atom_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_classes, prior_probs=prior_atom_types
        )
        self.bond_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_bond_classes, prior_probs=prior_bond_types
        )

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        print('Node indicator: ', self.config.node_indicator)
        if self.config.node_indicator:
            if self.add_prior_node:
                emb_dim = self.hidden_dim - 3
            else:
                emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        if self.add_prior_node:
            self.prior_std_expansion = GaussianSmearing(0., 5., num_gaussians=20, fix_offset=False)
            self.prior_atom_emb = nn.Linear(20, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        if self.refine_net_type == 'uni_o2_bond':
            self.ligand_bond_emb = nn.Linear(self.num_bond_classes, self.hidden_dim)

        # atom type prediction
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        if self.bond_diffusion:
            self.distance_expansion = GaussianSmearing(0., 5., num_gaussians=config.num_r_gaussian, fix_offset=False)
            if self.bond_net_type == 'pre_att':
                bond_input_dim = config.num_r_gaussian + self.hidden_dim
            elif self.bond_net_type == 'lin':
                bond_input_dim = self.hidden_dim
            else:
                raise ValueError(self.bond_net_type)
            self.bond_inference = nn.Sequential(
                nn.Linear(bond_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.num_bond_classes)
            )

    def forward(self, protein_pos, protein_v, batch_protein, protein_group_idx,
                init_ligand_pos, init_ligand_v, init_ligand_v_aux, batch_ligand, ligand_group_idx,
                prior_centers, prior_stds, batch_prior, prior_group_idx,
                ligand_fc_bond_index, init_ligand_fc_bond_type,
                ligand_atom_mask=None, time_step=None, return_all=False):

        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # add decomp indicator feature
        init_ligand_v = torch.cat([init_ligand_v, init_ligand_v_aux], -1)
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.add_prior_node:
            prior_std_feat = self.prior_std_expansion(prior_stds)
            prior_h = self.prior_atom_emb(prior_std_feat)

        if self.config.node_indicator:
            if self.add_prior_node:
                pro_ind = torch.tensor([1, 0, 0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)
                lig_ind = torch.tensor([0, 1, 0]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)
                prior_ind = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(len(prior_h), 1).to(h_protein)
                prior_h = torch.cat([prior_h, prior_ind], -1)
            else:
                pro_ind = torch.tensor([0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)
                lig_ind = torch.tensor([1]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)

            h_protein = torch.cat([h_protein, pro_ind], -1)
            init_ligand_h = torch.cat([init_ligand_h, lig_ind], -1)

        if self.add_prior_node:
            h_all, pos_all, group_idx_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = \
                compose_context_with_prior(
                    h_protein=h_protein,
                    h_ligand=init_ligand_h,
                    h_prior=prior_h,

                    pos_protein=protein_pos,
                    pos_ligand=init_ligand_pos,
                    pos_prior=prior_centers,

                    batch_protein=batch_protein,
                    batch_ligand=batch_ligand,
                    batch_prior=batch_prior,

                    protein_group_idx=protein_group_idx,
                    ligand_group_idx=ligand_group_idx,
                    prior_group_idx=prior_group_idx,
                    ligand_atom_mask=ligand_atom_mask
                )
        else:
            h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=init_ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_atom_mask=ligand_atom_mask
            )
            group_idx_all = None

        if ligand_fc_bond_index is not None:
            bond_index_in_all = l_index_in_ctx[ligand_fc_bond_index]
        else:
            bond_index_in_all = None

        if self.refine_net_type == 'uni_o2_bond':
            bond_type = F.one_hot(init_ligand_fc_bond_type, num_classes=self.num_bond_classes).float()
            h_bond = self.ligand_bond_emb(bond_type)
            outputs = self.refine_net(
                h=h_all, x=pos_all, group_idx=group_idx_all,
                bond_index=bond_index_in_all, h_bond=h_bond,
                mask_ligand=mask_ligand,
                mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                batch=batch_all,
                return_all=return_all
            )
        else:
            outputs = self.refine_net(
                h=h_all, x=pos_all, group_idx=group_idx_all,
                bond_index=bond_index_in_all, bond_type=init_ligand_fc_bond_type,
                mask_ligand=mask_ligand,
                mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                batch=batch_all,
                return_all=return_all
            )
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand_atom], final_h[mask_ligand_atom]
        final_ligand_v = self.v_inference(final_ligand_h)
        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v
        }

        if self.bond_diffusion:
            # bond inference input
            if self.bond_net_type == 'pre_att':
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                if self.bond_net_type == 'pre_att':
                    hi, hj = final_h[dst], final_h[src]
                    bond_inf_input = torch.cat([r_feat, (hi + hj) / 2], -1)
                else:
                    raise NotImplementedError
            elif self.bond_net_type == 'lin':
                bond_inf_input = outputs['h_bond']
            else:
                raise ValueError(self.bond_net_type)
            pred_bond = self.bond_inference(bond_inf_input)
            preds.update({
                'pred_bond': pred_bond
            })

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand_atom] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand_atom]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(pos0)
        pos_noise.normal_()
        pos_perturbed = a_pos.sqrt() * pos0 + (1.0 - a_pos).sqrt() * pos_noise
        pos_prior = torch.randn_like(pos_perturbed)
        kl_prior = torch.mean((pos_perturbed - pos_prior) ** 2)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, protein_group_idx,
            ligand_pos, ligand_v, ligand_v_aux, batch_ligand, ligand_group_idx,
            prior_centers, prior_stds, prior_num_atoms, batch_prior, prior_group_idx,

            ligand_decomp_batch, ligand_decomp_index,
            ligand_fc_bond_index=None, ligand_fc_bond_type=None, batch_ligand_bond=None, ligand_atom_mask=None,
            time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos, v, (and bond)
        assert len(ligand_decomp_batch) == prior_num_atoms.sum().item()
        batch_noise_centers = prior_centers[ligand_decomp_batch]
        batch_noise_stds = prior_stds[ligand_decomp_batch]
        assert len(batch_noise_centers) == len(batch_ligand)
        assert len(batch_noise_stds) == len(batch_ligand)
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * (ligand_pos - batch_noise_centers) + \
                               (1.0 - a_pos).sqrt() * pos_noise * batch_noise_stds + batch_noise_centers
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.atom_type_trans.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        if self.bond_diffusion:
            log_ligand_b0 = index_to_log_onehot(ligand_fc_bond_type, self.num_bond_classes)
            ligand_b_perturbed, log_ligand_bt = self.bond_type_trans.q_v_sample(
                log_ligand_b0, time_step, batch_ligand_bond)
        else:
            ligand_b_perturbed = None

        protein_pos, ligand_pos_perturbed, offset = center_pos(
            protein_pos, ligand_pos_perturbed, batch_protein, batch_ligand, mode=self.center_pos_mode)
        ligand_pos = ligand_pos - offset[batch_ligand]
        prior_centers = prior_centers - offset[batch_prior]
        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self.forward(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            protein_group_idx=protein_group_idx,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            init_ligand_v_aux=ligand_v_aux,
            batch_ligand=batch_ligand,
            ligand_group_idx=ligand_group_idx,
            ligand_fc_bond_index=ligand_fc_bond_index,
            init_ligand_fc_bond_type=ligand_b_perturbed,

            prior_centers=prior_centers,
            prior_stds=prior_stds,
            batch_prior=batch_prior,
            prior_group_idx=prior_group_idx,

            time_step=time_step,
            ligand_atom_mask=ligand_atom_mask
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed
        # atom position
        if self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos_perturbed, eps=pred_pos_noise, t=time_step, batch=batch_ligand)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        elif self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError
        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.atom_type_trans.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.atom_type_trans.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        # compute kl
        # kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
        #                              xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        if self.bond_diffusion:
            log_ligand_b_recon = F.log_softmax(preds['pred_bond'], dim=-1)
            log_b_model_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b_recon, log_ligand_bt, time_step, batch_ligand_bond)
            log_b_true_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b0, log_ligand_bt, time_step, batch_ligand_bond)
            kl_b = self.compute_v_Lt(log_v_model_prob=log_b_model_prob, log_v0=log_ligand_b0,
                                     log_v_true_prob=log_b_true_prob, t=time_step, batch=batch_ligand_bond)
            loss_bond = torch.mean(kl_b)
        else:
            loss_bond = torch.tensor(0.)

        if self.loss_pos_type == 'mse':
            # unweighted
            if self.model_mean_type == 'C0':
                target, pred = ligand_pos, pred_ligand_pos
            elif self.model_mean_type == 'noise':
                target, pred = pos_noise, pred_pos_noise
            else:
                raise ValueError
            loss_pos = scatter_mean((((pred - target) ** 2) / (batch_noise_stds ** 2)).sum(-1), batch_ligand, dim=0)
            loss_pos = torch.mean(loss_pos)
        else:
            raise ValueError
        loss_v = torch.mean(kl_v)
        results = {
            'losses': {
                'pos': loss_pos,
                'v': loss_v,
            },
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

        if self.bond_diffusion:
            results['losses']['bond'] = loss_bond
            results['ligand_b_recon'] = F.softmax(preds['pred_bond'], dim=-1)
        return results

# ===================== DecompDiff 核心采样函数（含奖励融合） =====================

    @torch.no_grad()
    def sample_diffusion(self, r_model, data, ligand_name, 
                         protein_pos, protein_v, batch_protein, protein_group_idx,
                         normalization_params_path,
                         init_ligand_pos, init_ligand_v, ligand_v_aux, 
                         batch_ligand, ligand_group_idx,
                         gamma1, gamma2, IRIE_step,
                         Rejection_threshold, last_rejection,threshold_change,
                         prior_centers, prior_stds, prior_num_atoms, 
                         batch_prior, prior_group_idx,
                         ligand_decomp_batch, ligand_decomp_index,
                         batch_protein_features, reward_need,
                         ligand_atom_mask=None,
                         ligand_fc_bond_index=None, init_ligand_fc_bond_type=None, 
                         batch_ligand_bond=None,
                         num_steps=None, center_pos_mode=None,
                         energy_drift_opt=None,
                         full_protein_pos=None, full_batch_protein=None,ε_v=-5, ε_q=0.6, ε_s=0.6,
                         pos_only=False):
        """
        扩散采样主函数（适配DecompDiff，含奖励融合）
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        
        # 中心化
        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        # 初始化轨迹
        pos_traj, v_traj, bond_traj = [], [], []
        v0_pred_traj, vt_pred_traj = [], []
        bt_pred_traj = []
        hbap_protein_traj, hbap_ligand_traj = [], []
        shift_cond_t_traj, shift_cond_t_minus1_traj = [], []
        
        ligand_pos, ligand_v, ligand_bond = init_ligand_pos, init_ligand_v, init_ligand_fc_bond_type
        
        # 初始化历史最优Vina（用于拒绝采样）
        best_reward = 1.0
        
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        device = protein_pos.device
        last_r_idx = last_rejection

        for idx, i in tqdm(enumerate(time_seq), desc='采样', total=num_steps):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)
            
            # ============ 常规扩散采样 ============
            ligand_pos, ligand_v, ligand_bond, pos_traj, v_traj, bond_traj, \
            v0_pred_traj, vt_pred_traj, bt_pred_traj = self.sample_diffusion_onestep(
                protein_pos=protein_pos, protein_v=protein_v,
                batch_protein=batch_protein, protein_group_idx=protein_group_idx,
                init_ligand_pos=ligand_pos, init_ligand_v=ligand_v,
                ligand_v_aux=ligand_v_aux, batch_ligand=batch_ligand,
                ligand_group_idx=ligand_group_idx,
                prior_centers=prior_centers, prior_stds=prior_stds,
                prior_num_atoms=prior_num_atoms, batch_prior=batch_prior,
                prior_group_idx=prior_group_idx,
                t=t, i=i,
                ligand_decomp_batch=ligand_decomp_batch,
                ligand_decomp_index=ligand_decomp_index,
                pos_traj=pos_traj, v_traj=v_traj, bond_traj=bond_traj,
                v0_pred_traj=v0_pred_traj, vt_pred_traj=vt_pred_traj,
                bt_pred_traj=bt_pred_traj, offset=offset,
                ligand_atom_mask=ligand_atom_mask,
                ligand_fc_bond_index=ligand_fc_bond_index,
                init_ligand_fc_bond_type=ligand_bond,
                batch_ligand_bond=batch_ligand_bond,
                num_steps=num_steps, center_pos_mode=center_pos_mode,
                energy_drift_opt=energy_drift_opt,
                full_protein_pos=full_protein_pos,
                full_batch_protein=full_batch_protein,
                reward=None
            )
            
            # ============ 拒绝采样（每50步检查） ============
            if idx - last_r_idx >= 20:
                # 随机选分子评估
                selected_protein_pos, selected_protein_v, selected_ligand_pos, selected_ligand_v = \
                    self.random_select_molecule(protein_pos, protein_v, batch_protein,
                                                 batch_ligand, ligand_pos, ligand_v, offset)
                
                params = np.load(normalization_params_path)
                lig_state, prot_state, time = preprocess_single_sample(
                    device=device,
                    ligand_pos=selected_ligand_pos,
                    ligand_v=selected_ligand_v,
                    protein_pos=selected_protein_pos,
                    protein_v=selected_protein_v,
                    idx=idx,
                    max_ligand_atoms=100,
                    max_protein_atoms=500,
                    ligand_mean=params['ligand_mean'],
                    ligand_std=params['ligand_std'],
                    protein_mean=params['protein_mean'],
                    protein_std=params['protein_std']
                )
                
                outputs = r_model(ligand_state=lig_state, protein_state=prot_state, time=time)
                probabilities = torch.softmax(outputs, dim=1)
                pred_p = probabilities[0, 1].item()
                
                # 触发拒绝采样
                if pred_p <= Rejection_threshold and idx - last_r_idx >= 20 and idx < 950:
                    last_r_idx = idx
                    print(f'idx: {idx}, 拒绝采样触发, pred_p={pred_p:.4f}')
                    
                    # 回退到10步前
                    ligand_pos_r = pos_traj[idx-10].to(device)
                    ligand_pos_r = ligand_pos_r - offset[batch_ligand]
                    ligand_v_r = v_traj[idx-10].to(device)
                    ligand_bond_r = bond_traj[idx-10].to(device)
                    idx_r = idx - 10
                    t_minus1 = torch.full(size=(num_graphs,), fill_value=idx_r-1, dtype=torch.long, device=device)
                    
                    # 准备条件引导参数（如果有）
                    net_cond = None  # 根据实际需要设置
                    hbap_protein_r = None
                    hbap_ligand_r = None
                    shift_cond_t_r = None
                    shift_cond_t_minus1_r = None
                    
                    # ============ 执行拒绝采样 ============
                    result = self.rejection(
                        data=data,
                        device=device,
                        init_ligand_pos=ligand_pos_r,
                        init_ligand_v=ligand_v_r,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        offset=offset,
                        batch_protein=batch_protein,
                        batch_ligand=batch_ligand,
                        idx=idx_r,
                        protein_group_idx=protein_group_idx,
                        ligand_group_idx=ligand_group_idx,
                        ligand_decomp_batch=ligand_decomp_batch,
                        ligand_decomp_index=ligand_decomp_index,
                        ligand_fc_bond_index=ligand_fc_bond_index,
                        init_ligand_fc_bond_type=ligand_bond_r,
                        batch_ligand_bond=batch_ligand_bond,
                        prior_centers=prior_centers,
                        prior_stds=prior_stds,
                        prior_num_atoms=prior_num_atoms,
                        batch_prior=batch_prior,
                        prior_group_idx=prior_group_idx,
                        ligand_atom_mask=ligand_atom_mask,
                        ligand_v_aux=ligand_v_aux,
                        ligand_name=ligand_name,
                        num_steps=num_steps,
                        center_pos_mode=center_pos_mode,
                        energy_drift_opt=energy_drift_opt,
                        full_batch_protein=full_batch_protein,
                        full_protein_pos=full_protein_pos,
                        reward_need=reward_need,
                        pos_only=pos_only,
                        ε_v=ε_v, ε_q=ε_q, ε_s=ε_s
                    )
                    
                    ligand_pos_traj_r = result['ligand_pos_traj_r']
                    ligand_v_traj_r = result['ligand_v_traj_r']
                    v0_pred_traj_r = result['v0_pred_traj_r']
                    vt_pred_traj_r = result['vt_pred_traj_r']
                    bt_pred_traj_r = result['bt_pred_traj_r']
                    bond_traj_r = result['bond_traj_r']

                    best_vina = result['best_vina']
                    reward = result['reward']

                    
                    if reward > best_reward:
                        print(f"idx: {idx}, 当前pareto_reward({reward:.2f}) 劣于历史最优({best_reward:.2f})，舍弃")
                        continue
                    
                    # ============ 更新全局最优 ============
                    best_reward = reward
                    print(f"idx: {idx}, 发现更优结果！新全局最优pareto_reward: {best_reward:.2f}")
                    
                    # ============ 获取最优轨迹的最终状态 ============
                    ligand_pos_r = ligand_pos_traj_r[9].to(device)
                    ligand_pos_r = ligand_pos_r 
                    ligand_v_r = ligand_v_traj_r[9].to(device)
                    ligand_bond_r = result.get('bond_traj_r', [init_ligand_fc_bond_type.cpu()])[9].to(device)
                    ligand_fc_bond_index_r = result.get('ligand_fc_bond_index_r', None)

                        
                        # 使用奖励进行一步引导采样
                    ligand_pos_r, ligand_v_r, ligand_bond_r, _, _, _, _, _, _ = \
                            self.sample_diffusion_onestep(
                                protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                protein_group_idx=protein_group_idx,
                                init_ligand_pos=ligand_pos_r,
                                init_ligand_v=ligand_v_r,
                                ligand_v_aux=ligand_v_aux,
                                batch_ligand=batch_ligand,
                                ligand_group_idx=ligand_group_idx,
                                prior_centers=prior_centers,
                                prior_stds=prior_stds,
                                prior_num_atoms=prior_num_atoms,
                                batch_prior=batch_prior,
                                prior_group_idx=prior_group_idx,
                                t=t,
                                i=i+1,
                                ligand_decomp_batch=ligand_decomp_batch,
                                ligand_decomp_index=ligand_decomp_index,
                                pos_traj=[],
                                v_traj=[],
                                bond_traj=[],
                                v0_pred_traj=[],
                                vt_pred_traj=[],
                                bt_pred_traj=[],
                                offset=offset,
                                ligand_atom_mask=ligand_atom_mask,
                                ligand_fc_bond_index=ligand_fc_bond_index,
                                init_ligand_fc_bond_type=ligand_bond_r,
                                batch_ligand_bond=batch_ligand_bond,
                                num_steps=num_steps,
                                center_pos_mode=center_pos_mode,
                                energy_drift_opt=energy_drift_opt,
                                full_protein_pos=full_protein_pos,
                                full_batch_protein=full_batch_protein,
                                reward=reward
                            )
                    
                    # ============ 验证改进 ============
                    selected_protein_pos, selected_protein_v, selected_ligand_pos, selected_ligand_v = \
                        self.random_select_molecule(protein_pos, protein_v, batch_protein,
                                                     batch_ligand, ligand_pos_r, ligand_v_r, offset)
                    
                    lig_state, prot_state, time = preprocess_single_sample(
                        device=device,
                        ligand_pos=selected_ligand_pos,
                        ligand_v=selected_ligand_v,
                        protein_pos=selected_protein_pos,
                        protein_v=selected_protein_v,
                        idx=idx,
                        max_ligand_atoms=100,
                        max_protein_atoms=500,
                        ligand_mean=params['ligand_mean'],
                        ligand_std=params['ligand_std'],
                        protein_mean=params['protein_mean'],
                        protein_std=params['protein_std']
                    )
                    
                    outputs_r = r_model(ligand_state=lig_state, protein_state=prot_state, time=time)
                    probabilities_r = torch.softmax(outputs_r, dim=1)
                    pred_p_r = probabilities_r[0, 1].item()
                    
                    # ============ 接受更好状态 ============
                    if pred_p_r > pred_p:
                        print(f'idx: {idx}, 拒绝采样接受! pred_p: {pred_p:.4f} -> {pred_p_r:.4f}')
                        # 坐标仅转float；原子、键索引必须保持long整数
                        ligand_pos = ligand_pos_r.float()
                        ligand_v = ligand_v_r.long()
                        ligand_bond = ligand_bond_r.long()  # 修复：键类型不转float，保持索引整数

                        n = len(ligand_pos_traj_r)
                        pos_traj[-n:] = [t.float() for t in ligand_pos_traj_r]
                        v_traj[-n:] = [t.long() for t in ligand_v_traj_r]
                        bond_traj[-n:] = [t.long() for t in bond_traj_r]
                        v0_pred_traj[-n:] = [t.float() for t in v0_pred_traj_r]
                        vt_pred_traj[-n:] = [t.float() for t in vt_pred_traj_r]

                        # 仅n>0才更新键预测轨迹
                        if n > 0 and len(bt_pred_traj_r) > 0:
                            bt_pred_traj[-n:] = bt_pred_traj_r
                        if n > 0 and len(bond_traj_r) > 0:
                            bond_traj[-n:] = bond_traj_r
        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': ligand_bond,
            'bond_traj': bond_traj,
            'bt_traj': bt_pred_traj,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
        }
    @torch.no_grad()
    def sample_diffusion_onestep(self, protein_pos, protein_v, batch_protein, protein_group_idx,
                                init_ligand_pos, init_ligand_v, ligand_v_aux, batch_ligand, 
                                ligand_group_idx,
                                prior_centers, prior_stds, prior_num_atoms, batch_prior, 
                                prior_group_idx, t, i,
                                ligand_decomp_batch, ligand_decomp_index, 
                                pos_traj, v_traj, bond_traj, v0_pred_traj, vt_pred_traj, 
                                bt_pred_traj, offset,
                                ligand_atom_mask=None,
                                ligand_fc_bond_index=None, init_ligand_fc_bond_type=None, 
                                batch_ligand_bond=None,
                                num_steps=None, center_pos_mode=None,
                                energy_drift_opt=None, reward=None,
                                full_protein_pos=None, full_batch_protein=None):
        """
        常规单步扩散采样
        """
        ligand_pos, ligand_v, ligand_bond = init_ligand_pos, init_ligand_v, init_ligand_fc_bond_type
        
        # 前向传播
        preds = self(
            protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
            protein_group_idx=protein_group_idx,
            init_ligand_pos=ligand_pos, init_ligand_v=ligand_v,
            init_ligand_v_aux=ligand_v_aux, batch_ligand=batch_ligand,
            ligand_group_idx=ligand_group_idx,
            ligand_fc_bond_index=ligand_fc_bond_index,
            init_ligand_fc_bond_type=ligand_bond,
            prior_centers=prior_centers, prior_stds=prior_stds,
            batch_prior=batch_prior, prior_group_idx=prior_group_idx,
            ligand_atom_mask=ligand_atom_mask,
            time_step=t
        )

        # 计算 x0 预测
        if self.model_mean_type == 'noise':
            pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
            pos0_from_e = self._predict_x0_from_eps(
                xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
            v0_from_e = preds['pred_ligand_v']
        elif self.model_mean_type == 'C0':
            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']
        else:
            raise ValueError

        # 后验均值
        pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
        pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
        nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

        # 采样位置
        if reward is None:
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
        else:
            noise_scale = 0.5 * reward
            variance_scale = (0.5 * pos_log_variance).exp() + noise_scale.unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * variance_scale * torch.randn_like(ligand_pos)

        if ligand_atom_mask is not None:
            ligand_pos_next[ligand_atom_mask == 0] = ligand_pos[ligand_atom_mask == 0]

        # 更新原子类型
        log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
        log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
        log_model_prob = self.atom_type_trans.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
        ligand_v_next = log_sample_categorical(log_model_prob)
        if ligand_atom_mask is not None:
            ligand_v_next[ligand_atom_mask == 0] = ligand_v[ligand_atom_mask == 0]

        # 更新键类型
        if self.bond_diffusion:
            log_ligand_b_recon = F.log_softmax(preds['pred_bond'], dim=-1)
            log_ligand_b = index_to_log_onehot(ligand_bond, self.num_bond_classes)
            log_b_model_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b_recon, log_ligand_b, t, batch_ligand_bond)
            ligand_b_next = log_sample_categorical(log_b_model_prob)
            ligand_bond = ligand_b_next
            bt_pred_traj.append(log_b_model_prob.clone().cpu())
            bond_traj.append(ligand_bond.clone().cpu())

        # 保存轨迹
        v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
        vt_pred_traj.append(log_model_prob.clone().cpu())
        ligand_pos = ligand_pos_next
        ligand_v = ligand_v_next
        
        ori_ligand_pos = ligand_pos + offset[batch_ligand]
        pos_traj.append(ori_ligand_pos.clone().cpu())
        v_traj.append(ligand_v.clone().cpu())
        
        return ligand_pos, ligand_v, ligand_bond, pos_traj, v_traj, bond_traj, \
            v0_pred_traj, vt_pred_traj, bt_pred_traj
    @torch.no_grad()
    def rejection(self, data, device, init_ligand_pos, init_ligand_v,
                protein_pos, protein_v, offset,
                batch_protein, batch_ligand, idx,  # 删除 net_cond, hbap_*, shift_cond_*
                protein_group_idx, ligand_group_idx,
                ligand_decomp_batch, ligand_decomp_index,
                ligand_fc_bond_index, init_ligand_fc_bond_type, batch_ligand_bond,
                prior_centers, prior_stds, prior_num_atoms, batch_prior, prior_group_idx,
                ligand_atom_mask, ligand_v_aux, ligand_name,
                num_steps, center_pos_mode, energy_drift_opt,
                full_batch_protein, full_protein_pos, reward_need,
                pos_only=False, num_parallel=8, ε_v=-5, ε_q=0.6, ε_s=0.6):
        """
        拒绝采样（含Vina评估和奖励计算）
        """
        bond_count_per_sample = init_ligand_fc_bond_type.size(0) if init_ligand_fc_bond_type is not None else 0
        num_graphs = batch_protein.max().item() + 1
        start_t = 999 - idx
        total_steps = 10
        atom_count = init_ligand_pos.size(0) 
        # ============ 并行输入准备 ============
        protein_pos_rep = protein_pos.repeat(num_parallel, 1).to(device)
        protein_v_rep = protein_v.repeat(num_parallel, 1).to(device)

        # ✅ 修复：蛋白质 batch 索引（每个样本独立）
        batch_protein_rep = torch.cat([batch_protein + i * num_graphs for i in range(num_parallel)], dim=0).to(device)

        # ✅ 修复：使用 unique 计算组数
        num_protein_groups = torch.unique(protein_group_idx).size(0)
        protein_group_idx_rep = torch.cat([protein_group_idx + i * num_protein_groups for i in range(num_parallel)], dim=0).to(device)

        num_ligand_groups = torch.unique(ligand_group_idx).size(0)
        ligand_group_idx_rep = torch.cat([ligand_group_idx + i * num_ligand_groups for i in range(num_parallel)], dim=0).to(device)

        ligand_pos_rep = init_ligand_pos.repeat(num_parallel, 1).to(device)
        ligand_v_rep = init_ligand_v.repeat(num_parallel).to(device)

        # ✅ 修复：配体 batch 索引（每个样本独立）
        batch_ligand_rep = torch.cat([batch_ligand + i * num_graphs for i in range(num_parallel)], dim=0).to(device)

        prior_centers_rep = prior_centers.repeat(num_parallel, 1).to(device)
        prior_stds_rep = prior_stds.repeat(num_parallel, 1).to(device)

        # ✅ 修复：prior batch 索引（每个样本独立）
        batch_prior_rep = torch.cat([batch_prior + i * num_graphs for i in range(num_parallel)], dim=0).to(device)

        num_prior_groups = torch.unique(prior_group_idx).size(0)
        prior_group_idx_rep = torch.cat([prior_group_idx + i * num_prior_groups for i in range(num_parallel)], dim=0).to(device)

        # ✅ 修复：ligand_decomp_batch 偏移（每个样本独立）
        ligand_decomp_batch_rep = torch.cat([ligand_decomp_batch + i * num_graphs for i in range(num_parallel)], dim=0).to(device)
        ligand_decomp_index_rep = ligand_decomp_index.repeat(num_parallel).to(device)

        offset_rep = offset.repeat(num_parallel, 1).to(device)

        # ✅ 修复：键索引偏移（基于配体原子数）
        if ligand_fc_bond_index is not None:
            ligand_fc_bond_index_rep_list = []
            for k in range(num_parallel):
                offset_idx = k * atom_count
                bond_idx_offset = ligand_fc_bond_index + offset_idx
                ligand_fc_bond_index_rep_list.append(bond_idx_offset)
            ligand_fc_bond_index_rep = torch.cat(ligand_fc_bond_index_rep_list, dim=1).to(device)
        else:
            ligand_fc_bond_index_rep = None

        init_ligand_fc_bond_type_rep = init_ligand_fc_bond_type.repeat(num_parallel).to(device)

        # ✅ 修复：键的 batch 索引（每个样本独立）
        batch_ligand_bond_rep = torch.cat([batch_ligand_bond + i * num_graphs for i in range(num_parallel)], dim=0).to(device)

        if ligand_atom_mask is not None:
            ligand_atom_mask_rep = ligand_atom_mask.repeat(num_parallel).to(device)
        else:
            ligand_atom_mask_rep = None

        ligand_v_aux_rep = ligand_v_aux.repeat(num_parallel, 1).to(device)


        # ============ 并行生成轨迹 ============
        pos_traj_list = [[] for _ in range(num_parallel)]
        v_traj_list = [[] for _ in range(num_parallel)]
        v0_traj_list = [[] for _ in range(num_parallel)]
        bond_traj_list = [[] for _ in range(num_parallel)]  # 新增：存储键轨迹
        bt_traj_list = [[] for _ in range(num_parallel)]    # 新增：存储键预测轨迹
        
        current_pos = ligand_pos_rep
        current_v = ligand_v_rep
        current_bond = init_ligand_fc_bond_type_rep

        for step in range(total_steps):
            t = start_t - step
            if t < 0:
                break
            t_tensor = torch.full((num_parallel * num_graphs,), t, dtype=torch.long, device=device)
            
            # 使用带条件引导的采样
            current_pos, current_v, current_bond, _, _, _, v0_traj_step, vt_traj_step, bt_traj_step = \
                self.sample_diffusion_onestep(
                    protein_pos=protein_pos_rep,
                    protein_v=protein_v_rep,
                    batch_protein=batch_protein_rep,
                    protein_group_idx=protein_group_idx_rep,
                    init_ligand_pos=current_pos,
                    init_ligand_v=current_v,
                    ligand_v_aux=ligand_v_aux_rep,
                    batch_ligand=batch_ligand_rep,
                    ligand_group_idx=ligand_group_idx_rep,
                    prior_centers=prior_centers_rep,
                    prior_stds=prior_stds_rep,
                    prior_num_atoms=prior_num_atoms,
                    batch_prior=batch_prior_rep,
                    prior_group_idx=prior_group_idx_rep,
                    t=t_tensor,
                    i=t,
                    ligand_decomp_batch=ligand_decomp_batch_rep,
                    ligand_decomp_index=ligand_decomp_index_rep,
                    pos_traj=[],
                    v_traj=[],
                    bond_traj=[],
                    v0_pred_traj=[],
                    vt_pred_traj=[],
                    bt_pred_traj=[],
                    offset=offset_rep,
                    ligand_atom_mask=ligand_atom_mask_rep,
                    ligand_fc_bond_index=ligand_fc_bond_index_rep,
                    init_ligand_fc_bond_type=current_bond,
                    batch_ligand_bond=batch_ligand_bond_rep,
                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,
                    energy_drift_opt=energy_drift_opt,
                    full_protein_pos=full_protein_pos,
                    full_batch_protein=full_batch_protein,
                    reward=None
                )
            
            # 保存每个粒子的轨迹
            atom_count = init_ligand_pos.size(0)
            for k in range(num_parallel):
                st, ed = k * atom_count, (k + 1) * atom_count
                pos_traj_list[k].append(current_pos[st:ed].cpu().clone())
                v_traj_list[k].append(current_v[st:ed].cpu().clone())
                
                # ✅ 修复：使用正确的键数量
                if current_bond is not None:
                    bond_st, bond_ed = k * bond_count_per_sample, (k + 1) * bond_count_per_sample
                    bond_traj_list[k].append(current_bond[bond_st:bond_ed].cpu().clone())
                else:
                    # 如果没有键，填充空张量
                    bond_traj_list[k].append(torch.zeros(0))
                
                if len(v0_traj_step) > 0:
                    v0_traj_list[k].append(v0_traj_step[-1].clone())
                if len(bt_traj_step) > 0:
                    bt_traj_list[k].append(bt_traj_step[-1].clone())

            # ============ 评估并选择最优 ============
        # ============ 评估并选择最优（单Vina → 帕累托多目标） ============
        valid_records = []  # 格式：(原始轨迹索引k, vina_score, qed_score, sa_score)

        for k in range(num_parallel):
            try:
                final_pos = pos_traj_list[k][-1].to(device)
                final_v = v_traj_list[k][-1].to(device)
                pos_real = final_pos + offset[batch_ligand]
                pos_np = pos_real.cpu().numpy().astype(np.float64)
                v_np = final_v.cpu().numpy().astype(np.float64)

                atom_types = transforms.get_atomic_number_from_index(v_np, mode='add_aromatic')
                aromatic = transforms.is_aromatic_from_index(v_np, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pos_np, atom_types, aromatic)

                if mol is None or mol.GetNumAtoms() < 5:
                    continue

                # 多目标评分计算：Vina + QED + SA
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, ligand_name, protein_root='./data/test_set')
                vina_score = vina_task.run(mode='minimize', exhaustiveness=16)[0]['affinity']
                chem_results = scoring_func.get_chem(mol)
                qed_score = chem_results['qed']
                sa_score = chem_results['sa']
                valid_records.append((k, vina_score, qed_score, sa_score))

            except Exception as e:
                continue

        # ============ 兜底 + 帕累托标量化选优 ============
        if len(valid_records) == 0:
            best_idx = 0
            best_vina = 100.0
            reward = torch.tensor(0.0, device=device)
        else:
            pareto_scores = []
            global_k_list = []
            for k_global, v, q, s in valid_records:
                # Vina 极小化目标：r_v = σ(2*(ε_v - v))，值越小得分越高
                r_v = torch.sigmoid(2 * torch.tensor(ε_v - v, dtype=torch.float32, device=device))
                # QED 极大化目标：r_q = σ(8*(q - ε_q))，值越大得分越高
                r_q = torch.sigmoid(8 * torch.tensor(q - ε_q, dtype=torch.float32, device=device))
                # SA 极大化目标：r_s = σ(8*(s - ε_s))，值越大得分越高
                r_s = torch.sigmoid(8 * torch.tensor(s - ε_s, dtype=torch.float32, device=device))
                # ε-约束乘积形式帕累托标量化总得分
                R_pareto = r_v * r_q * r_s
                pareto_scores.append(R_pareto.item())
                global_k_list.append(k_global)

            local_best_idx = int(np.argmax(pareto_scores))
            best_idx = global_k_list[local_best_idx]
            _, best_vina, _, _ = valid_records[local_best_idx]
            # 奖励计算与原逻辑兼容：总得分越高，奖励值越小（对应扰动越小）
            best_R = torch.tensor(pareto_scores[local_best_idx], dtype=torch.float32, device=device)
            reward = 1.0 - best_R

        # ============ 获取最优轨迹（原逻辑完全不动） ============
        best_traj = pos_traj_list[best_idx]
        best_v_traj = v_traj_list[best_idx]
        best_v0_traj = v0_traj_list[best_idx] if v0_traj_list[best_idx] else []
        best_bond_traj = bond_traj_list[best_idx] if bond_traj_list[best_idx] else []
        best_bt_traj = bt_traj_list[best_idx] if bt_traj_list[best_idx] else []

        # ============ 获取最优轨迹对应的键索引（原逻辑完全不动） ============
        best_bond_index = None
        if ligand_fc_bond_index is not None and ligand_fc_bond_index_rep is not None:
            if isinstance(init_ligand_fc_bond_type, torch.Tensor):
                bond_count_per_sample = init_ligand_fc_bond_type.size(0)
            else:
                bond_count_per_sample = 0
            
            if bond_count_per_sample > 0:
                st_bond = best_idx * bond_count_per_sample
                ed_bond = (best_idx + 1) * bond_count_per_sample
                if ed_bond <= ligand_fc_bond_index_rep.size(1):
                    best_bond_index = ligand_fc_bond_index_rep[:, st_bond:ed_bond]

        print(f"✅ Rejection 帕累托多目标完成 | 最优 Vina: {best_vina:.2f} | Reward: {reward.item():.4f}")

        return {
            'ligand_pos_traj_r': best_traj,
            'ligand_v_traj_r': best_v_traj,
            'v0_pred_traj_r': best_v0_traj,
            'vt_pred_traj_r': [],  # 保留原逻辑，不修改
            'bt_pred_traj_r': best_bt_traj,
            'bond_traj_r': best_bond_traj,
            'ligand_fc_bond_index_r': best_bond_index,
            'best_vina': best_vina,
            'reward': reward,
        }

    def random_select_molecule(self,protein_pos, protein_v, batch_protein,
                          batch_ligand, ligand_pos, ligand_v,offset):
        """
        从批次数据中随机选择一个完整分子的所有相关原子数据
        
        参数:
            各参数均为tensor，其中batch_protein和batch_ligand为分子编号标识
            ligand_x等tensor的前n个数据对应第一个配体，n等于batch_ligand中0的个数
            
        返回:
            选中的分子的所有相关原子数据
        """
        # 获取所有唯一的分子编号
        unique_molecules = torch.unique(torch.cat([batch_protein, batch_ligand]))
        protein_pos = offset[batch_protein]+protein_pos
        ligand_pos = offset[batch_ligand]+ligand_pos
        # 随机选择一个分子编号
        random_idx = random.randint(0, len(unique_molecules) - 1)
        selected_mol_id = unique_molecules[random_idx].item()
        
        # 根据分子编号筛选对应的蛋白质原子
        protein_mask = (batch_protein == selected_mol_id)
        selected_protein_pos = protein_pos[protein_mask]
        selected_protein_v = protein_v[protein_mask]
        
        # 根据分子编号筛选对应的配体原子
        ligand_mask = (batch_ligand == selected_mol_id)
        selected_ligand_pos = ligand_pos[ligand_mask]
        selected_ligand_v = ligand_v[ligand_mask]
        
        return (selected_protein_pos, selected_protein_v, 
                selected_ligand_pos, selected_ligand_v)    

def preprocess_single_sample(device,ligand_pos, ligand_v, protein_pos, protein_v, idx,
                             max_ligand_atoms=100, max_protein_atoms=500,
                             ligand_mean=None, ligand_std=None,
                             protein_mean=None, protein_std=None):
        """
        预处理单个样本，转换为模型可接受的格式
        
        参数：
            ligand_pos: 配体位置，形状 (n_ligand_atoms, 3)
            ligand_v: 配体速度，形状 (n_ligand_atoms, 1)
            protein_pos: 蛋白质位置，形状 (n_protein_atoms, 3)
            protein_v: 蛋白质特征，形状 (n_protein_atoms, 27)
            idx: 时间步（int）
            max_ligand_atoms: 配体最大原子数（与训练一致）
            max_protein_atoms: 蛋白质最大原子数（与训练一致）
            ligand_mean/std: 训练时的配体归一化参数
            protein_mean/std: 训练时的蛋白质归一化参数
        返回：
            处理后的ligand_state、protein_state、time
        """
        ligand_pos = ligand_pos.to(device)
        ligand_v = ligand_v.to(device)
        protein_v = protein_v[:, :27]
        ligand_pos = ligand_pos.cpu().numpy()  # 先移到CPU，再转NumPy
        ligand_v = ligand_v.cpu().numpy()
        ligand_v = ligand_v.reshape(-1, 1)  # 形状变为 (, 1)
        protein_pos = protein_pos.cpu().numpy()
        protein_v= protein_v.cpu().numpy()
        # 1. 拼接配体特征（3位置 + 1速度）
        ligand_state = np.concatenate([ligand_pos, ligand_v], axis=1)  # 形状 (n_ligand_atoms, 4)
        
        # 补齐配体原子数到max_ligand_atoms
        if ligand_state.shape[0] < max_ligand_atoms:
            pad_len = max_ligand_atoms - ligand_state.shape[0]
            ligand_state = np.pad(ligand_state, ((0, pad_len), (0, 0)), mode='constant')
        else:
            ligand_state = ligand_state[:max_ligand_atoms]  # 截断超长部分
        
        # 2. 拼接蛋白质特征（3位置 + 27特征）
        protein_state = np.concatenate([protein_pos, protein_v], axis=1)  # 形状 (n_protein_atoms, 30)
        
        # 补齐蛋白质原子数到max_protein_atoms
        if protein_state.shape[0] < max_protein_atoms:
            pad_len = max_protein_atoms - protein_state.shape[0]
            protein_state = np.pad(protein_state, ((0, pad_len), (0, 0)), mode='constant')
        else:
            protein_state = protein_state[:max_protein_atoms]  # 截断超长部分

        # 3. 归一化（必须与训练时一致）
        if ligand_mean is not None and ligand_std is not None:
            ligand_state = (ligand_state - ligand_mean) / ligand_std
        if protein_mean is not None and protein_std is not None:
            protein_state = (protein_state - protein_mean) / protein_std
        
        # 4. 处理时间步（确保在有效范围内）
        time = idx  # 直接使用idx作为时间步
         # 从()或(1,)变为(1,)（与batch_size匹配）
        # 转换为tensor
        ligand_state = torch.tensor(ligand_state, dtype=torch.float32).to(device)
        protein_state = torch.tensor(protein_state, dtype=torch.float32).to(device)
        time = torch.tensor(time, dtype=torch.float32).to(device)
        ligand_state = ligand_state.unsqueeze(0)
        protein_state = protein_state.unsqueeze(0)
        time = time.unsqueeze(0) 
        return ligand_state, protein_state, time