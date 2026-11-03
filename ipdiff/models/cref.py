import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


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


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


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
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
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

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        k_t = np.sqrt(alphas_cumprod) * (1 - np.sqrt(alphas_cumprod))
        self.k_t = to_torch_const(k_t)

        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

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
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        self.cond_dim = config.cond_dim
        self.emb_mlp = nn.Linear(emb_dim + self.cond_dim, emb_dim)

        self.shift_t_mlp_pos = nn.Sequential(nn.Linear(self.cond_dim + 1, 3))

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand, time_step=None, return_all=False, fix_x=False, hbap_protein_prev=None, hbap_ligand_prev=None, hbap_protein=None, hbap_ligand=None):

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
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
        if hbap_protein is None:
            hbap_protein = torch.zeros([h_protein.shape[0], self.cond_dim]).to(h_protein.device)
        if hbap_ligand is None:
            hbap_ligand = torch.zeros([init_ligand_h.shape[0], self.cond_dim]).to(init_ligand_h.device)

        h_protein = self.emb_mlp(torch.cat([h_protein, hbap_protein], dim=1))
        init_ligand_h = self.emb_mlp(torch.cat([init_ligand_h, hbap_ligand], dim=1))

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand, _ =compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h
        }
        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, t_minus1, batch, shift=None, shift_minus1=None):
        if shift is None or t_minus1 is None:
            pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                             extract(self.posterior_mean_ct_coef, t, batch) * xt
        else:
            if shift is None:
                pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                                 extract(self.posterior_mean_ct_coef, t, batch) * xt + extract(self.k_t, t_minus1, batch) * shift_minus1
            else:
                pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                                 extract(self.posterior_mean_ct_coef, t, batch) * (xt - extract(self.k_t, t, batch) * (shift)) + \
                                 extract(self.k_t, t_minus1, batch) * shift_minus1
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
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


    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, net_cond, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step=None
    ):
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        hbap_ligand = None
        hbap_protein = None
        if self.model_mean_type == 'noise':
            pass
        elif self.model_mean_type == 'C0':
            gt_protein_v = protein_v
            gt_protein_pos = protein_pos
            gt_ligand_v = ligand_v
            gt_ligand_pos = ligand_pos

            gt_lig_a_h = gt_ligand_v
            gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
            gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)

            hbap_ligand, hbap_protein = net_cond.extract_features(gt_ligand_pos, gt_protein_pos, gt_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
        else:
            raise ValueError


        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        k_t = self.k_t.index_select(0, time_step)

        k_t_pos = k_t[batch_ligand].unsqueeze(-1)

        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()

        shift_cond_t = torch.cat([hbap_ligand, time_step[batch_ligand].unsqueeze(-1)], -1)
        shift_cond_t = self.shift_t_mlp_pos(shift_cond_t)
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise + k_t_pos * shift_cond_t
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step,

            hbap_protein=hbap_protein,
            hbap_ligand=hbap_ligand
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)
        loss = loss_pos + loss_v * self.loss_v_weight

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

    @torch.no_grad()
    def sample_diffusion(self,r_model,data,num_particles, protein_pos,normalization_params_path,threshold_change, protein_v, batch_protein,
                        init_ligand_pos, init_ligand_v, batch_ligand,other_protein_features,gamma1=5.5, gamma2=0.0,
                        IRIE_step=0, Rejection_threshold=0.5, last_rejection=0,
                        num_steps=None, center_pos_mode=None, pos_only=False, net_cond=None, cond_dim=128,ε_v=-5, ε_q=0.6, ε_s=0.6):
        log_weights = torch.zeros(num_particles, device=init_ligand_pos.device) - torch.log(torch.tensor(num_particles, device=init_ligand_pos.device))
        device = next(self.parameters()).device
        global_best_vina = float('inf')
        # 全局设备统一：所有基础张量迁移到模型设备
        best_reward = 1.0
        protein_pos = protein_pos.to(device)
        protein_v = protein_v.to(device)
        batch_protein = batch_protein.to(device)
        init_ligand_v = init_ligand_v.to(device)
        batch_ligand = batch_ligand.to(device)

        # 外部蛋白质特征 + 设备迁移
        other_protein_features = [
            (prot_pos.to(device), prot_v.to(device), prot_batch.to(device))
            for (prot_pos, prot_v, prot_batch) in other_protein_features
        ]
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        # 外部蛋白质坐标中心化
        centered_other_proteins = []
        for (prot_pos, prot_v, prot_batch) in other_protein_features:
            prot_pos_centered = prot_pos - offset[prot_batch]
            centered_other_proteins.append((prot_pos_centered, prot_v, prot_batch))
        other_protein_features = centered_other_proteins

        # 轨迹容器
        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        hbap_ligand_traj, hbap_protein_traj = [], []
        shift_cond_t_traj,shift_cond_t_minus1_traj = [], []

        # 初始配体状态
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        reward = None
        last_r_idx = last_rejection

        # 初始化 hbap 张量（强制模型设备）
        hbap_protein = torch.zeros([batch_protein.shape[0], cond_dim], dtype=torch.float, device=device)
        hbap_ligand = torch.zeros([batch_ligand.shape[0], cond_dim], dtype=torch.float, device=device)

        # 外部蛋白质 hbap 初始化
        hbap_other_proteins = []
        hbap_other_ligands = []
        for (prot_pos, prot_v, prot_batch) in other_protein_features:
            hbap_other_prot = torch.zeros([prot_batch.shape[0], cond_dim], dtype=torch.float, device=device)
            hbap_other_lig = torch.zeros([batch_ligand.shape[0], cond_dim], dtype=torch.float, device=device)
            hbap_other_proteins.append(hbap_other_prot)
            hbap_other_ligands.append(hbap_other_lig)

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        tT = torch.full(size=(num_graphs,), fill_value=len(time_seq), dtype=torch.long, device=protein_pos.device)
        shift_cond_t = None
        shift_cond_t_minus1 = self.shift_t_mlp_pos(torch.cat([hbap_ligand, tT[batch_ligand].unsqueeze(-1)], -1))

        # 主采样循环
        for idx, i in tqdm(enumerate(time_seq), desc='sampling', total=num_steps):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            t_minus1 = torch.full(size=(num_graphs,), fill_value=i-1, dtype=torch.long, device=protein_pos.device) if i >= 1 else None

            # IRIE分支 / 普通单步分支
            if i >= IRIE_step:
                ligand_pos, ligand_v, v0_pred_traj, vt_pred_traj, pos_traj, v_traj, hbap_protein, hbap_ligand, shift_cond_t, shift_cond_t_minus1, hbap_ligand_traj, hbap_protein_traj, hbap_other_proteins, hbap_other_ligands, shift_cond_t_traj, shift_cond_t_minus1_traj = self.sample_diffusion_Estep(
                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, net_cond,
                    batch_ligand, i, v0_pred_traj, vt_pred_traj, offset, pos_traj, v_traj,
                    hbap_protein, hbap_ligand, t_minus1, shift_cond_t, shift_cond_t_minus1,
                    hbap_ligand_traj, hbap_protein_traj, hbap_other_proteins, hbap_other_ligands,
                    other_protein_features, shift_cond_t_traj, shift_cond_t_minus1_traj,
                    pos_only=pos_only, gamma1=gamma1, gamma2=gamma2
                )
            else:
                ligand_pos, ligand_v, v0_pred_traj, vt_pred_traj, pos_traj, v_traj,hbap_protein, hbap_ligand, shift_cond_t, shift_cond_t_minus1,hbap_ligand_traj, hbap_protein_traj, shift_cond_t_traj, shift_cond_t_minus1_traj = self.sample_diffusion_onestep(
                    protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, net_cond,
                    batch_ligand, i, v0_pred_traj, vt_pred_traj, offset, pos_traj, v_traj,
                    hbap_protein, hbap_ligand, t_minus1, shift_cond_t, shift_cond_t_minus1,
                    hbap_ligand_traj, hbap_protein_traj, shift_cond_t_traj, shift_cond_t_minus1_traj,
                    pos_only=pos_only
                )

            # 拒绝采样触发条件
            if idx - last_r_idx >= 20 and idx < 950:
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
                # 修复：关键字传参 -> 位置传参
                outputs = r_model(lig_state, prot_state, time)
                probabilities = torch.softmax(outputs, dim=1)
                pred_p = probabilities[0, 1].item()

                # 判定触发拒绝采样
                if pred_p <= Rejection_threshold and idx - last_r_idx >= 20:
                    last_r_idx = idx
                    print(f'idx: {idx}, rejection trigger')
                    # 回溯10步读取历史轨迹 + 强制设备转换 + 判空防护
                    ligand_pos_r = pos_traj[idx - 10].to(device)
                    ligand_v_r = v_traj[idx - 10].to(device)

                    # hbap 设备+空值防护
                    hbap_ligand_r = hbap_ligand_traj[idx - 10]
                    hbap_ligand_r = hbap_ligand_r.to(device) if hbap_ligand_r is not None else torch.zeros_like(ligand_pos_r[:, :0])
                    hbap_protein_r = hbap_protein_traj[idx - 10]
                    hbap_protein_r = hbap_protein_r.to(device) if hbap_protein_r is not None else torch.zeros((1, cond_dim), device=device)

                    # shift 设备+空值防护
                    shift_cond_t_r = shift_cond_t_traj[idx - 10]
                    shift_cond_t_r = shift_cond_t_r.to(device) if shift_cond_t_r is not None else None
                    shift_cond_t_minus1_r = shift_cond_t_minus1_traj[idx - 10]
                    shift_cond_t_minus1_r = shift_cond_t_minus1_r.to(device) if shift_cond_t_minus1_r is not None else None

                    idx_r = idx - 10
                    ligand_pos_r = ligand_pos_r - offset[batch_ligand]

                    # 执行拒绝采样
                    result = self.rejection(
                        data=data,
                        device=device,
                        init_ligand_pos=ligand_pos_r,
                        init_ligand_v=ligand_v_r,
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        offset=offset,
                        net_cond=net_cond,
                        hbap_protein=hbap_protein_r,
                        hbap_ligand=hbap_ligand_r,
                        t_minus1=t_minus1,
                        shift_cond_t=shift_cond_t_r,
                        shift_cond_t_minus1=shift_cond_t_minus1_r,
                        batch_protein=batch_protein,
                        batch_ligand=batch_ligand,
                        threshold_change=threshold_change,
                        idx=idx_r,
                        ε_v=ε_v, ε_q=ε_q, ε_s=ε_s
                    )

                    ligand_pos_traj_r = result['ligand_pos_traj_r']
                    ligand_v_traj_r = result['ligand_v_traj_r']
                    v0_pred_traj_r = result['v0_pred_traj_r']
                    vt_pred_traj_r = result['vt_pred_traj_r']
                    hbap_ligand_traj_r = result['hbap_ligand_traj']
                    hbap_protein_traj_r = result['hbap_protein_traj']
                    shift_cond_t_traj_r = result['shift_cond_t_traj_r']
                    shift_cond_t_minus1_traj_r = result['shift_cond_t_minus1_traj_r']
                    best_vina = result['best_vina']
                    reward = result['reward']

                    # 空轨迹保护
                    if len(ligand_pos_traj_r) == 0:
                        print(f"idx: {idx}, all trajectory invalid, skip")
                        continue


                    ligand_pos_r = ligand_pos_traj_r[9]
                    ligand_pos_r = ligand_pos_r.to(device)
                    ligand_v_r = ligand_v_traj_r[9]
                    ligand_v_r = ligand_v_r.to(device)
                    ligand_pos_r = ligand_pos_r 



                    if reward.item() < best_reward:
                        best_reward = reward.item()
                        print(f"⚠️ reward guide | best_vina={best_vina:.2f} | reward={reward.item():.2f}")
                    # 解包数量匹配：14个返回值
                        ligand_pos_r, ligand_v_r, _, _, _, _, _, _, _, _, _, _, _, _ = self.sample_diffusion_onestep(
                                protein_pos, protein_v, batch_protein, ligand_pos_r, ligand_v_r, net_cond,
                                batch_ligand, i+1, [], [], offset, [], [],
                                hbap_protein_r, hbap_ligand_r, t_minus1, shift_cond_t_r, shift_cond_t_minus1_r,
                                [], [], [], [], reward=reward, pos_only=pos_only
                            )

                        # 二次校验模型
                        selected_protein_pos2, selected_protein_v2, selected_ligand_pos2, selected_ligand_v2 = \
                            self.random_select_molecule(protein_pos, protein_v, batch_protein,
                                                    batch_ligand, ligand_pos_r, ligand_v_r, offset)
                        lig_state2, prot_state2, time2 = preprocess_single_sample(
                            device=device,
                            ligand_pos=selected_ligand_pos2,
                            ligand_v=selected_ligand_v2,
                            protein_pos=selected_protein_pos2,
                            protein_v=selected_protein_v2,
                            idx=idx,
                            max_ligand_atoms=100,
                            max_protein_atoms=500,
                            ligand_mean=params['ligand_mean'],
                            ligand_std=params['ligand_std'],
                            protein_mean=params['protein_mean'],
                            protein_std=params['protein_std']
                        )
                        # 修复：位置传参
                        outputs_r = r_model(lig_state2, prot_state2, time2)
                        probabilities_r = torch.softmax(outputs_r, dim=1)
                        pred_p_r = probabilities_r[0, 1].item()

                        if pred_p > pred_p_r:
                            continue
                        else:
                            print(f'idx: {idx}, rejection accepted! replace trajectory')
                            n = len(ligand_pos_traj_r)
                            
                            
                            # 替换轨迹（确保类型一致）
                            for j in range(n):
                                if j < len(pos_traj):
                                    pos_traj[-(n-j)] = ligand_pos_traj_r[j].cpu()
                                if j < len(v_traj):
                                    v_traj[-(n-j)] = ligand_v_traj_r[j].cpu()
                                if j < len(v0_pred_traj):
                                    v0_pred_traj[-(n-j)] = v0_pred_traj_r[j]
                                if j < len(vt_pred_traj):
                                    vt_pred_traj[-(n-j)] = vt_pred_traj_r[j]
                                if j < len(hbap_ligand_traj):
                                    hbap_ligand_traj[-(n-j)] = hbap_ligand_traj_r[j]
                                if j < len(hbap_protein_traj):
                                    hbap_protein_traj[-(n-j)] = hbap_protein_traj_r[j]
                                if j < len(shift_cond_t_traj):
                                    shift_cond_t_traj[-(n-j)] = shift_cond_t_traj_r[j]
                                if j < len(shift_cond_t_minus1_traj):
                                    shift_cond_t_minus1_traj[-(n-j)] = shift_cond_t_minus1_traj_r[j]
                            
                            # 更新当前状态（确保在正确的设备上）
                            ligand_pos = ligand_pos_r.to(device)
                            ligand_v = ligand_v_r.to(device)
                            if hbap_protein_traj_r:
                                hbap_protein = hbap_protein_traj_r[-1].to(device)
                            if hbap_ligand_traj_r:
                                hbap_ligand = hbap_ligand_traj_r[-1].to(device)
                            if shift_cond_t_traj_r:
                                shift_cond_t = shift_cond_t_traj_r[-1].to(device) if shift_cond_t_traj_r[-1] is not None else None
                            if shift_cond_t_minus1_traj_r:
                                shift_cond_t_minus1 = shift_cond_t_minus1_traj_r[-1].to(device) if shift_cond_t_minus1_traj_r[-1] is not None else None
                    else:
                        print(f"idx: {idx}, rejection rejected | best_vina={best_vina:.2f} | reward={reward.item():.2f}")
                
                # 恢复原始坐标
        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
                'pos': ligand_pos,
                'v': ligand_v,
                'pos_traj': pos_traj,
                'v_traj': v_traj,
                'vt_traj': vt_pred_traj,
                'v0_traj': v0_pred_traj
            }

    def sample_diffusion_onestep(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, net_cond,
                                batch_ligand, i, v0_pred_traj, vt_pred_traj, offset, pos_traj, v_traj,
                                hbap_protein, hbap_ligand, t_minus1, shift_cond_t, shift_cond_t_minus1,
                                hbap_ligand_traj, hbap_protein_traj, shift_cond_t_traj, shift_cond_t_minus1_traj,
                                pos_only=False, enable_grad=False, reward=None):
        
        # ========= 完全使用可行版本的代码 =========
        # reward, when supplied, changes the coordinate noise scale below.
        
        num_graphs = batch_protein.max().item() + 1
        t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
        if i >= 1:
                t_minus1 = torch.full(size=(num_graphs,), fill_value=i - 1, dtype=torch.long, device=protein_pos.device)
        else:
                t_minus1 = None
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos,  # 直接用，不 clone
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            time_step=t,
            hbap_protein=hbap_protein,
            hbap_ligand=hbap_ligand
        )
        
        if self.model_mean_type == 'noise':
            pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
            pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
            v0_from_e = preds['pred_ligand_v']
        elif self.model_mean_type == 'C0':
            pos0_from_e = preds['pred_ligand_pos']
            v0_from_e = preds['pred_ligand_v']
        else:
            raise ValueError

        pos_model_mean = self.q_pos_posterior(
            x0=pos0_from_e, xt=ligand_pos, t=t, t_minus1=t_minus1,
            batch=batch_ligand, shift=shift_cond_t, shift_minus1=shift_cond_t_minus1
        )
        pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
        nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
        if reward==None:  
                ligand_pos = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
        else:
                ligand_pos = pos_model_mean + nonzero_mask * ((0.5 * pos_log_variance).exp() + 0.5*reward) * torch.randn_like(ligand_pos)

        
        gt_protein_pos = protein_pos.detach()
        gt_protein_v = protein_v.detach()
        gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
        gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)
        pred_ligand_pos = pos0_from_e.detach()
        pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)

        hbap_ligand, hbap_protein = net_cond.extract_features(
            pred_ligand_pos, gt_protein_pos, pred_lig_a_h,
            gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein
        )
        hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()
        
        shift_cond_t = shift_cond_t_minus1
        if t_minus1 is not None:
            shift_cond_t_minus1 = self.shift_t_mlp_pos(torch.cat([hbap_ligand, t_minus1[batch_ligand].unsqueeze(-1)], -1))
        
        if not pos_only:
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v = log_sample_categorical(log_model_prob)
            
            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())
        
        ori_ligand_pos = ligand_pos + offset[batch_ligand]
        pos_traj.append(ori_ligand_pos.clone().cpu())
        v_traj.append(ligand_v.clone().cpu())
        hbap_ligand_traj.append(hbap_ligand.clone())
        hbap_protein_traj.append(hbap_protein.clone())
        shift_cond_t_traj.append(shift_cond_t)
        shift_cond_t_minus1_traj.append(shift_cond_t_minus1)
        
        # 返回14个值，保持接口一致
        return (ligand_pos, ligand_v, v0_pred_traj, vt_pred_traj, pos_traj, v_traj,
                hbap_protein, hbap_ligand, shift_cond_t, shift_cond_t_minus1,
                hbap_ligand_traj, hbap_protein_traj, shift_cond_t_traj, shift_cond_t_minus1_traj)
    def rejection(self, data, device, protein_pos, protein_v, batch_protein, init_ligand_pos, offset,
                  net_cond, t_minus1, shift_cond_t, shift_cond_t_minus1, hbap_ligand, hbap_protein,
                  init_ligand_v, batch_ligand, threshold_change, idx, pos_only=False, num_parallel=8, ε_v=-5, ε_q=0.6, ε_s=0.6):
        num_graphs = batch_protein.max().item() + 1
        start_t = self.num_timesteps - idx
        total_steps = 10

        # 并行多粒子初始化 + 强制设备
        protein_pos_rep = protein_pos.repeat(num_parallel, 1).to(device)
        protein_v_rep = protein_v.repeat(num_parallel, 1).to(device)
        batch_protein_rep = torch.cat([batch_protein + i * num_graphs for i in range(num_parallel)], dim=0).to(device)
        ligand_pos_rep = init_ligand_pos.repeat(num_parallel, 1).to(device)
        ligand_v_rep = init_ligand_v.repeat(num_parallel).to(device)
        batch_ligand_rep = torch.cat([batch_ligand + i * num_graphs for i in range(num_parallel)], dim=0).to(device)
        hbap_protein_rep = hbap_protein.repeat(num_parallel, 1).to(device)
        hbap_ligand_rep = hbap_ligand.repeat(num_parallel, 1).to(device)

        curr_shift_t = shift_cond_t.repeat(num_parallel, 1).to(device) if shift_cond_t is not None else None
        curr_shift_tm1 = shift_cond_t_minus1.repeat(num_parallel, 1).to(device) if shift_cond_t_minus1 is not None else None

        all_trajectories = [[] for _ in range(num_parallel)]
        current_pos = ligand_pos_rep.clone()
        current_v = ligand_v_rep.clone()
        current_hbap_prot = hbap_protein_rep.clone()
        current_hbap_lig = hbap_ligand_rep.clone()

        # 并行执行10步去噪
        for step in range(total_steps):
            t = start_t - step
            t_tensor = torch.full((num_parallel * num_graphs,), t, dtype=torch.long, device=device)
            t_m1 = torch.full((num_parallel * num_graphs,), t-1, dtype=torch.long, device=device) if t>=1 else None
            with torch.no_grad():
                preds = self(
                    protein_pos=protein_pos_rep,
                    protein_v=protein_v_rep,
                    batch_protein=batch_protein_rep,
                    init_ligand_pos=current_pos,
                    init_ligand_v=current_v,
                    batch_ligand=batch_ligand_rep,
                    time_step=t_tensor,
                    hbap_protein=current_hbap_prot,
                    hbap_ligand=current_hbap_lig
                )
                if self.model_mean_type == 'C0':
                    pos0_pred = preds['pred_ligand_pos']
                    v0_pred = preds['pred_ligand_v']
                else:
                    pos0_pred = preds['pred_ligand_pos'] - current_pos
                    pos0_pred = self._predict_x0_from_eps(current_pos, pos0_pred, t_tensor, batch_ligand_rep)
                    v0_pred = preds['pred_ligand_v']

                pos_mean = self.q_pos_posterior(x0=pos0_pred, xt=current_pos, t=t_tensor, t_minus1=t_m1,
                                                batch=batch_ligand_rep, shift=curr_shift_t, shift_minus1=curr_shift_tm1)
                log_var = extract(self.posterior_logvar, t_tensor, batch_ligand_rep)
                mask = (1 - (t_tensor == 0).float())[batch_ligand_rep].unsqueeze(-1)
                noise = torch.randn_like(pos_mean)
                current_pos = pos_mean + mask * torch.exp(0.5 * log_var) * noise

                # 更新hbap与偏移条件
                gt_prot_pos = protein_pos_rep.detach()
                gt_prot_v = protein_v_rep.detach()
                gt_p_a_h = torch.argmax(gt_prot_v[:, :6], dim=1)
                gt_p_r_h = torch.argmax(gt_prot_v[:, 6:26], dim=1)
                pred_l_pos = pos0_pred.detach()
                pred_l_a_h = torch.argmax(v0_pred.detach(), dim=1)
                current_hbap_lig, current_hbap_prot = net_cond.extract_features(
                    pred_l_pos, gt_prot_pos, pred_l_a_h, gt_p_a_h, gt_p_r_h, batch_ligand_rep, batch_protein_rep
                )
                current_hbap_lig, current_hbap_prot = current_hbap_lig.detach(), current_hbap_prot.detach()

                curr_shift_t = curr_shift_tm1
                if t_m1 is not None:
                    curr_shift_tm1 = self.shift_t_mlp_pos(torch.cat([current_hbap_lig, t_m1[batch_ligand_rep].unsqueeze(-1)], -1))

                # 更新配体原子类型
                if not pos_only:
                    log_v = F.log_softmax(v0_pred, dim=-1)
                    log_v_t = index_to_log_onehot(current_v, self.num_classes)
                    log_prob = self.q_v_posterior(log_v, log_v_t, t_tensor, batch_ligand_rep)
                    current_v = log_sample_categorical(log_prob)

                ligand_atom_cnt = init_ligand_pos.size(0)
                protein_atom_cnt = protein_pos.size(0)
                for k in range(num_parallel):
                    st_lig, ed_lig = k * ligand_atom_cnt, (k + 1) * ligand_atom_cnt
                    st_prot, ed_prot = k * protein_atom_cnt, (k + 1) * protein_atom_cnt
                    all_trajectories[k].append({
                        'pos': current_pos[st_lig:ed_lig].cpu().clone(),
                        'v': current_v[st_lig:ed_lig].cpu().clone(),
                        'v0': v0_pred[st_lig:ed_lig].cpu().clone(),
                        'hbap_lig': current_hbap_lig[st_lig:ed_lig].cpu().clone(),
                        'hbap_prot': current_hbap_prot[st_prot:ed_prot].cpu().clone(),
                        'shift_t': curr_shift_t[st_lig:ed_lig].cpu().clone() if curr_shift_t is not None else None,
                        'shift_tm1': curr_shift_tm1[st_lig:ed_lig].cpu().clone() if curr_shift_tm1 is not None else None
                    })


        # ========== 仅修改本部分：单Vina择优 → 帕累托多目标择优 ==========
        valid_records = []  # (原始轨迹k, vina, qed, sa)
        for k in range(num_parallel):
            try:
                final_pos = all_trajectories[k][-1]['pos'].to(device)
                final_v = all_trajectories[k][-1]['v'].to(device)
                pos_real = final_pos + offset[batch_ligand]
                pos_np = pos_real.cpu().numpy().astype(np.float64)
                v_np = final_v.cpu().numpy().astype(np.float64)

                atom_types = transforms.get_atomic_number_from_index(v_np, mode='add_aromatic')
                aromatic = transforms.is_aromatic_from_index(v_np, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pos_np, atom_types, aromatic)

                if mol is None or mol.GetNumAtoms() < 5:
                    continue
                # 同步计算三项指标
                vina_task = VinaDockingTask.from_generated_mol(mol, data.ligand_filename, protein_root='./data/test_set')
                vina_score = vina_task.run(mode='minimize', exhaustiveness=16)[0]['affinity']
                chem_results = scoring_func.get_chem(mol)
                qed_score = chem_results['qed']
                sa_score = chem_results['sa']
                valid_records.append((k, vina_score, qed_score, sa_score))
            except Exception:
                continue

        # 兜底逻辑 + 帕累托选优
        if len(valid_records) == 0:
            best_idx = 0
            best_vina = 100.0
            reward = torch.tensor(0.0, device=device)
        else:
            pareto_scores = []
            global_k_list = []
            for k_global, v, q, s in valid_records:
                # Vina极小化：r_v = σ(2*(ε_v - v))
                r_v = torch.sigmoid(2 * torch.tensor(ε_v - v, dtype=torch.float32, device=device))
                # QED/SA极大化：r_i = σ(8*(score - ε))
                r_q = torch.sigmoid(8 * torch.tensor(q - ε_q, dtype=torch.float32, device=device))
                r_s = torch.sigmoid(8 * torch.tensor(s - ε_s, dtype=torch.float32, device=device))
                # 乘积形式ε-约束帕累托标量化
                R_pareto = r_v * r_q * r_s
                pareto_scores.append(R_pareto.item())
                global_k_list.append(k_global)

            local_best_idx = np.argmax(pareto_scores)
            best_idx = global_k_list[local_best_idx]
            _, best_vina, _, _ = valid_records[local_best_idx]
            # 扰动权重：R越高则扰动越小，与原逻辑兼容
            best_R = torch.tensor(pareto_scores[local_best_idx], dtype=torch.float32, device=device)
            reward = 1.0 - best_R

        change = threshold_change * reward

        # 解析最优轨迹
        best_traj = all_trajectories[best_idx]
        pos_list = [s['pos'] for s in best_traj]
        v_list = [s['v'] for s in best_traj]
        v0_list = [s['v0'] for s in best_traj]
        hbap_l_list = [s['hbap_lig'] for s in best_traj]
        hbap_p_list = [s['hbap_prot'] for s in best_traj]
        st_list = [s['shift_t'] for s in best_traj]
        stm1_list = [s['shift_tm1'] for s in best_traj]

        print(f"✅ Rejection 帕累托多目标完成 | 最优Vina: {best_vina:.2f} | Perturb Reward: {reward.item():.2f}")

        return {
            'ligand_pos_traj_r': pos_list,
            'ligand_v_traj_r': v_list,
            'v0_pred_traj_r': v0_list,
            'vt_pred_traj_r': v0_list,  # 改成用 v0_list 而不是空列表
            # 或者如果不需要，创建一个占位列表
            # 'vt_pred_traj_r': [None] * len(pos_list),
            'hbap_ligand_traj': hbap_l_list,
            'hbap_protein_traj': hbap_p_list,
            'shift_cond_t_traj_r': st_list,
            'shift_cond_t_minus1_traj_r': stm1_list,
            'best_vina': best_vina,
            'reward': reward,
            'change': change
        }
    def sample_diffusion_Estep(self,protein_pos, protein_v, batch_protein,ligand_pos,ligand_v,net_cond,
                         batch_ligand,i,v0_pred_traj,
                         vt_pred_traj,offset,pos_traj,
                         v_traj,hbap_protein,hbap_ligand,
                         t_minus1,shift_cond_t,shift_cond_t_minus1,
                         hbap_ligand_traj, hbap_protein_traj,hbap_other_proteins,
                         hbap_other_ligands,other_protein_features,shift_cond_t_traj,shift_cond_t_minus1_traj,
                         pos_only=False,gamma1=5.5, gamma2=0.0): 
                num_graphs = batch_protein.max().item() + 1     
                    
               
               
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
                preds = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,

                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t,

                        hbap_protein=hbap_protein,
                        hbap_ligand=hbap_ligand
                    )

                pos0_cond = preds['pred_ligand_pos']  # 条件x0预测



                    
                    # --------------------------
                    # 2. 计算无条件噪声 u_theta(xt, ∅, t)（外部蛋白质特征的平均）
                    # --------------------------
                pos0_uncond_list = []
                for idx, ((prot_pos, prot_v, prot_batch), hbap_other_prot, hbap_other_lig) in enumerate(
                    zip(other_protein_features, hbap_other_proteins, hbap_other_ligands)
                ):
                        # 用外部蛋白质特征预测噪声（模拟"无提示"条件）
                    preds_uncond = self(
                            protein_pos=prot_pos,
                            protein_v=prot_v,
                            batch_protein=prot_batch,  # 复用批次信息
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            time_step=t,
                            hbap_protein=hbap_other_prot,
                            hbap_ligand=hbap_ligand                    
                        )
                    pos0_uncond = preds_uncond['pred_ligand_pos']
                    pos0_uncond_list.append(pos0_uncond)

            
                

                pos_uncond = torch.mean(torch.stack(pos0_uncond_list), dim=0)  # 平均作为无条件噪声

                    
                    # --------------------------
                    # 3. 公式（5）：融合条件与无条件噪声（带指导尺度）
                    #    e_theta(xt) = (1+gamma1)*u_cond - gamma1*u_uncond
                    # --------------------------
                e_theta = (1 + gamma1) * pos0_cond - gamma1 * pos_uncond
                    
                    # --------------------------
                    # 4. 去噪步骤（公式2）：x_{t-1} = Phi^t(xt | c, gamma1)
                    # --------------------------
                alpha_t = extract(self.alphas_cumprod, t, batch_ligand)
                alpha_t_prev = extract(self.alphas_cumprod_prev, t, batch_ligand)
                # 计算去噪均值
                term1 = torch.sqrt(alpha_t_prev) * (ligand_pos - torch.sqrt(1 - alpha_t) * e_theta) / torch.sqrt(alpha_t)
                term2 = torch.sqrt(1 - alpha_t_prev) * e_theta
                pos_mean = term1 + term2  # 公式（2）的去噪均值
                    # 加入噪声（根据 posterior variance）
            
                ligand_pos_denoise = pos_mean

                
                # --------------------------
                # 5. Zigzag反演+再去噪（仅对前lambda_zigzag步执行）
                # --------------------------
                preds_prev = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,

                        init_ligand_pos=ligand_pos_denoise,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t-1,

                        hbap_protein=hbap_protein,
                        hbap_ligand=hbap_ligand
                    )

                pos0_cond_prev = preds_prev['pred_ligand_pos']  # 条件x0预测

                    
                    # --------------------------
                    # 2. 计算无条件噪声 u_theta(xt, ∅, t)（外部蛋白质特征的平均）
                    # --------------------------
                pos0_uncond_prev_list = []
                for idx, ((prot_pos, prot_v, prot_batch), hbap_other_prot, hbap_other_lig) in enumerate(
                    zip(other_protein_features, hbap_other_proteins, hbap_other_ligands)
                ):
                        # 用外部蛋白质特征预测噪声（模拟"无提示"条件）
                    preds_uncond_prev = self(
                            protein_pos=prot_pos,
                            protein_v=prot_v,
                            batch_protein=prot_batch,  # 复用批次信息
                            init_ligand_pos=ligand_pos_denoise,
                            init_ligand_v=ligand_v,
                            batch_ligand=batch_ligand,
                            time_step=t-1,
                            hbap_protein=hbap_other_prot,
                            hbap_ligand=hbap_ligand                  
                        )
                    pos0_uncond_prev = preds_uncond_prev['pred_ligand_pos']
                    pos0_uncond_prev_list.append(pos0_uncond_prev)
            
                

                
                pos_uncond_prev = torch.mean(torch.stack(pos0_uncond_prev_list), dim=0)  # 平均作为无条件噪声
              
                
                h_t = torch.sqrt(1/alpha_t - 1) - torch.sqrt(1/alpha_t_prev - 1)
                e_theta_inv = (1 + gamma2) * pos0_cond_prev - gamma2 * pos_uncond_prev
                xt_inv = torch.sqrt(alpha_t / alpha_t_prev) * ligand_pos_denoise + torch.sqrt(alpha_t) * h_t * e_theta_inv
                

                

                # 重新计算条件噪声
                preds_rev_cond = self(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    init_ligand_pos=xt_inv,  # 使用反演后的状态
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t,
                    
                    hbap_protein=hbap_protein,
                    hbap_ligand=hbap_ligand  
                )
                pos0_rev_cond = preds_rev_cond['pred_ligand_pos']
                pos0_from_e = pos0_rev_cond
                v0_from_e = preds_rev_cond['pred_ligand_v']
                
                
                # 重新计算无条件噪声
                pos0_rev_uncond_list = []
                for idx, ((prot_pos, prot_v, prot_batch), hbap_other_prot, hbap_other_lig) in enumerate(
                    zip(other_protein_features, hbap_other_proteins, hbap_other_ligands)
                ):
                    preds_rev_uncond = self(
                        protein_pos=prot_pos,
                        protein_v=prot_v,
                        batch_protein=prot_batch,
                        init_ligand_pos=xt_inv,  # 使用反演后的状态
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t,
                        hbap_protein=hbap_other_prot,
                        hbap_ligand=hbap_ligand 
                    )
                    pos0_rev_uncond = preds_rev_uncond['pred_ligand_pos']
                    pos0_rev_uncond_list.append(pos0_rev_uncond)
                pos_rev_uncond = torch.mean(torch.stack(pos0_rev_uncond_list), dim=0)
                # 释放不再需要的列表

                
                # 融合噪声进行再去噪
                e_theta_rev = (1 + gamma1) * pos0_rev_cond - gamma1 * pos_rev_uncond
                
                # 再去噪步骤（公式2）：x_{t-1} = Phi^t(\tilde{x}_t | c, gamma1)
                term1_rev = torch.sqrt(alpha_t_prev) * (xt_inv - torch.sqrt(1 - alpha_t) * e_theta_rev) / torch.sqrt(alpha_t)
                term2_rev = torch.sqrt(1 - alpha_t_prev) * e_theta_rev
                pos_model_mean_rev = term1_rev + term2_rev
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand) 
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_temp = pos_model_mean_rev + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)




            


                gt_protein_pos = protein_pos.detach()
                gt_protein_v = protein_v.detach()
                gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
                gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)
                pred_ligand_pos = ligand_pos_temp.detach()
                pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)

                hbap_ligand, hbap_protein = net_cond.extract_features(pred_ligand_pos, gt_protein_pos, pred_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
                hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()

                    # 逐个更新外部蛋白质的hbap
                for idx, ((prot_pos, prot_v, prot_batch), hbap_other_prot, hbap_other_lig) in enumerate(
                        zip(other_protein_features, hbap_other_proteins, hbap_other_ligands)
                ):
                        # 外部蛋白质的真实特征
                        gt_other_prot_pos = prot_pos.detach()
                        gt_other_prot_v = prot_v.detach()
                        gt_other_prot_a_h = torch.argmax(gt_other_prot_v[:, :6], dim=1)
                        gt_other_prot_r_h = torch.argmax(gt_other_prot_v[:, 6:26], dim=1)
                        # 用相同的预测配体特征更新外部蛋白质的hbap
                        hbap_other_lig, hbap_other_prot = net_cond.extract_features(
                            pred_ligand_pos, gt_other_prot_pos, pred_lig_a_h, gt_other_prot_a_h, 
                            gt_other_prot_r_h, batch_ligand, prot_batch  # 注意传入外部蛋白质的batch
                        )
                        hbap_other_lig, hbap_other_prot = hbap_other_lig.detach(), hbap_other_prot.detach()
                        # 更新列表中的hbap
                        hbap_other_proteins[idx] = hbap_other_prot
                        hbap_other_ligands[idx] = hbap_other_lig

                shift_cond_t = shift_cond_t_minus1
                if t_minus1 is not None:
                    shift_cond_t_minus1 = self.shift_t_mlp_pos(torch.cat([hbap_ligand, t_minus1[batch_ligand].unsqueeze(-1)], -1))

                if not pos_only:
                    log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                    log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                    log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                    ligand_v_next = log_sample_categorical(log_model_prob)

                    v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                    vt_pred_traj.append(log_model_prob.clone().cpu())
                    ligand_v = ligand_v_next

                ori_ligand_pos = ligand_pos_temp + offset[batch_ligand]
                pos_traj.append(ori_ligand_pos.clone().cpu())
                v_traj.append(ligand_v.clone().cpu())
                hbap_ligand_traj.append(hbap_ligand.clone().cpu())
                hbap_protein_traj.append(hbap_protein.clone().cpu())
                shift_cond_t_traj.append(shift_cond_t)
                shift_cond_t_minus1_traj.append(shift_cond_t_minus1)




                
                return ligand_pos,ligand_v,v0_pred_traj,vt_pred_traj,pos_traj,v_traj,hbap_protein,hbap_ligand,shift_cond_t,shift_cond_t_minus1,hbap_ligand_traj, hbap_protein_traj, hbap_other_proteins,hbap_other_ligands,shift_cond_t_traj,shift_cond_t_minus1_traj    
    
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

    def reward_diffusion(self, data, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v,batch_ligand,net_cond,shift_cond_t,shift_cond_t_minus1,
                         i, v0_pred_traj, vt_pred_traj,hbap_protein,hbap_ligand, offset, pos_traj=None, v_traj=None,num_reward=4, num_steps=None,
                        pos_only=False, enable_grad=False):
        if num_steps is None:
            num_steps = self.num_timesteps
        if num_steps is None:
            num_steps = self.num_timesteps
        # 提前初始化，避免 finally 中报错
        traj_ligand_pos = None
        traj_ligand_v = None
        traj_ligand_pos_final = None



        repeated_protein_pos = protein_pos.repeat(num_reward, 1) # [n*num_reward, 3]
        repeated_protein_v = protein_v.repeat(num_reward, 1)     # [n*num_reward, feat_dim]
        batch_indices = []
        for idx in range(num_reward):
                    batch_indices.append(torch.full((protein_pos.size(0),), idx, dtype=torch.long))
        repeated_protein_batch = torch.cat(batch_indices, dim=0)  # 形状: [num_reward*n]
                
        repeated_ligand_pos = ligand_pos.repeat(num_reward, 1) # [n*num_reward, 3]
        repeated_ligand_v = ligand_v.repeat(num_reward)     # [n*num_reward, feat_dim]
        repeated_hbap_ligand = hbap_ligand.repeat(num_reward, 1)  # [n*num_reward, hbap_dim]
        repeated_hbap_protein = hbap_protein.repeat(num_reward, 1)
        batch_indices = []
        for idx in range(num_reward):
                    batch_indices.append(torch.full((ligand_pos.size(0),), idx, dtype=torch.long))
        repeated_ligand_batch = torch.cat(batch_indices, dim=0)  # 形状: [num_reward*n]
        
        repeated_offset = offset.repeat(num_reward, 1)  # [n*num_reward, 3]
        
        repeated_protein_batch=repeated_protein_batch.to(protein_pos.device)
        repeated_ligand_batch=repeated_ligand_batch.to(ligand_pos.device)
        repeated_shift_cond_t = shift_cond_t.repeat(num_reward, 1).to(protein_pos.device)
        repeated_shift_cond_t_minus1 = shift_cond_t_minus1.repeat(num_reward, 1).to(protein_pos.device)
        traj_ligand_pos = repeated_ligand_pos
        traj_ligand_v = repeated_ligand_v
        time_seq = list(reversed(range(self.num_timesteps - num_steps, i + 1)))
        protein_pos=repeated_protein_pos.to(protein_pos.device)
        protein_v=repeated_protein_v.to(protein_pos.device)
        batch_protein=repeated_protein_batch.to(protein_pos.device)
        ligand_pos=repeated_ligand_pos.to(protein_pos.device)
        ligand_v=repeated_ligand_v.to(protein_pos.device)
        batch_ligand=repeated_ligand_batch.to(protein_pos.device)
        offset=repeated_offset.to(protein_pos.device)
        hbap_ligand=repeated_hbap_ligand.to(protein_pos.device)
        hbap_protein=repeated_hbap_protein.to(protein_pos.device)
        shift_cond_t=repeated_shift_cond_t.to(protein_pos.device)
        shift_cond_t_minus1=repeated_shift_cond_t_minus1.to(protein_pos.device)
        for step in tqdm(time_seq, desc='reward_diffusion (real-time traj)', total=len(time_seq)):
                    t = torch.full(size=(num_reward,), fill_value=step, dtype=torch.long, device=protein_pos.device)
                    t_minus1 = torch.full(size=(num_reward,), fill_value=step - 1, dtype=torch.long, device=protein_pos.device)
                    # 无梯度扩散，不存储中间张量，仅保留最终结果
                    preds = self(
                        protein_pos=protein_pos,
                        protein_v=protein_v,
                        batch_protein=batch_protein,

                        init_ligand_pos=ligand_pos,
                        init_ligand_v=ligand_v,
                        batch_ligand=batch_ligand,
                        time_step=t,

                        hbap_protein=hbap_protein,
                        hbap_ligand=hbap_ligand
                    )
                    if self.model_mean_type == 'noise':
                        pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                        pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                        v0_from_e = preds['pred_ligand_v']
                    elif self.model_mean_type == 'C0':
                        pos0_from_e = preds['pred_ligand_pos']
                        v0_from_e = preds['pred_ligand_v']
                    else:
                        raise ValueError

                    pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, t_minus1=t_minus1, batch=batch_ligand, shift=shift_cond_t, shift_minus1=shift_cond_t_minus1)
                    pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                    nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

                    ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                    ligand_pos = ligand_pos_next

                    gt_protein_pos = protein_pos.detach()
                    gt_protein_v = protein_v.detach()
                    gt_protein_a_h = torch.argmax(gt_protein_v[:, :6], dim=1)
                    gt_protein_r_h = torch.argmax(gt_protein_v[:, 6:26], dim=1)
                    pred_ligand_pos = pos0_from_e.detach()
                    pred_lig_a_h = torch.argmax(v0_from_e.detach(), dim=1)

                    hbap_ligand, hbap_protein = net_cond.extract_features(pred_ligand_pos, gt_protein_pos, pred_lig_a_h, gt_protein_a_h, gt_protein_r_h, batch_ligand, batch_protein)
                    hbap_ligand, hbap_protein = hbap_ligand.detach(), hbap_protein.detach()

                    shift_cond_t = shift_cond_t_minus1
                    if t_minus1 is not None:
                        shift_cond_t_minus1 = self.shift_t_mlp_pos(torch.cat([hbap_ligand, t_minus1[batch_ligand].unsqueeze(-1)], -1))

                    if not pos_only:
                        log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                        log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
                        log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                        ligand_v_next = log_sample_categorical(log_model_prob)
                        ligand_v = ligand_v_next

                    ori_ligand_pos = ligand_pos + offset[batch_ligand]


        ligand_pos = ligand_pos + offset[batch_ligand]
                
        torch.cuda.empty_cache()
            # -------------------------- 2. 梯度局部关联（仅绑定最终步与 ligand_pos_pred）--------------------------
        reward = []
        batch_ligand_tensor = repeated_ligand_batch  # 保留 tensor 避免截断梯度
        particle_indices = torch.unique(batch_ligand_tensor)
            
        for ligand_idx in particle_indices:
                atom_mask = (batch_ligand_tensor == ligand_idx)  # 粒子原子掩码
                
                # 2.1 用无梯度轨迹结果计算奖励数值（Vina对接、稳定性检查）
                particle_pos_np =  ligand_pos[atom_mask].cpu().numpy().astype(np.float64)
                particle_v_np = ligand_v[atom_mask].cpu().numpy().astype(np.float64)
                # 原子类型转换与分子重构（无梯度，正常执行）
                try:
                    pred_atom_type = transforms.get_atomic_number_from_index(particle_v_np, mode='add_aromatic')
                    pred_aromatic = transforms.is_aromatic_from_index(particle_v_np, mode='add_aromatic')
                    mol = reconstruct.reconstruct_from_generated(particle_pos_np, pred_atom_type, pred_aromatic)
                    chem_results = scoring_func.get_chem(mol)
                    qed = chem_results['qed']
                    sa = chem_results['sa']
                    affinity_1 = -1 * sa + 0.6
                    if affinity_1 < 0:
                        affinity_1 = 0
                    # Vina对接计算奖励数值（无梯度）
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, data.ligand_filename, protein_root='./data/test_set')
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=16)
                    affinity = score_only_results[0]['affinity']
                except Exception as e:
                    # 捕获所有异常：重构失败、对接失败等
                    print(f"处理粒子时发生错误: {str(e)}，将亲和力设为0")
                    affinity = 100
                    affinity_1 = 0.6
                
                finally:
                    # 强制删除Vina相关对象，避免隐式引用GPU张量（关键！）
                    del particle_pos_np, particle_v_np, pred_atom_type, pred_aromatic
                    vina_reward =  1.0/(1 + np.exp(-(2*(affinity+6))))
                    SA_reward = affinity_1
                    trans_reward = vina_reward*0.4 + SA_reward*0.6
                    reward_tensor = torch.tensor(trans_reward, device=ligand_pos.device, dtype=torch.float32)
                    reward.append(reward_tensor)
                    if 'mol' in locals():
                        del mol
                    if 'vina_task' in locals():
                        del vina_task
                    if 'score_only_results' in locals():
                        del score_only_results

        del batch_ligand_tensor, particle_indices, atom_mask
        return sum(reward) / len(reward)
    def lookforward_fn(self, rewards):
        """
        将vina对接分数（rewards）转换为粒子权重调整项（log_twist）
        
        参数:
            rewards: 张量，形状为[num_particles]，每个元素为粒子的vina_score（值越低越好）
        
        返回:
            log_twist: 张量，形状同rewards，值越大表示粒子越优质，用于更新log_weights
        """
        # 基础逻辑：vina_score越低（结合越好），权重调整项越大
        # 直接取负号（因rewards是vina_score，值越小越优）
        log_twist = -rewards  # 核心转换
        

        
        return log_twist

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

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
