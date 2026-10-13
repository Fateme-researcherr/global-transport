from functools import partial

import torch
import numpy as np

from optimal_transport import OTPlanSampler
from samplers import null_cond


# Built once at import time so it's ready as soon as training starts.
_OT_SAMPLER = OTPlanSampler()


def interpolant(x, t, path_type="linear", use_ot=False, use_ccot=False, c=None, accelerator=None):
    eps = torch.randn_like(x)
    if use_ot:
        if accelerator is not None and accelerator.num_processes > 1:
            # Solve the exact OT coupling over the whole global batch, not just
            # this GPU's batch.
            eps = _OT_SAMPLER.sample_plan_with_scipy_global(x, eps, accelerator)
        else:
            j = _OT_SAMPLER.sample_plan_with_scipy(x, eps)
            eps = eps[j]
    elif use_ccot:
        # class-conditional OT: couple each datapoint to noise carrying the same class
        if accelerator is not None and accelerator.num_processes > 1:
            eps = _OT_SAMPLER.sample_conditional_plan_with_scipy_global(x, eps, c, accelerator)
        else:
            j = _OT_SAMPLER.sample_conditional_plan_with_scipy(x, eps, c)
            eps = eps[j]
    if path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = (-np.pi / 2) * torch.sin(t * np.pi / 2)
        d_sigma_t = (np.pi / 2) * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError()
    x_t = alpha_t * x + sigma_t * eps
    v_t = d_alpha_t * x + d_sigma_t * eps
    return x_t, v_t


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def log_lv_loss(x, y, lv, eps=0.01):
    err = ((x - y) ** 2)
    mse_loss = torch.sum(err, dim=list(range(1, len(x.shape))))
    mean_loss = torch.mean(err, dim=list(range(1, len(x.shape))))
    lv = lv.reshape(-1)
    log_loss = torch.log((1 / lv.exp()) * mean_loss + eps) + lv * 0.5
    return mse_loss, log_loss


def mse_loss(x, y):
    err = (x - y)**2
    mse_loss = torch.sum(err, dim=list(range(1, len(x.size()))))
    loss = torch.mean(err, dim=list(range(1, len(x.size()))))
    return mse_loss, loss

def repa_loss(zs, zs_tilde):
    proj_loss = 0.
    bsz = zs[0].shape[0]
    for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
        for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
            z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
            z_j = torch.nn.functional.normalize(z_j, dim=-1) 
            proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
    proj_loss /= (len(zs) * bsz)
    return proj_loss


def compute_guidance(model, x, t, y, v_t, drop_ids, omega=0.5, g_min=0.0, g_max=1.0, g_type="default"):
    if g_type == "default":
        return v_t
    elif g_type == "mg":
        drop_ids = drop_ids.reshape(-1, 1, 1, 1)
        cfg_ids = (t >= g_min) & (t <= g_max) & (~drop_ids)
        y_null = null_cond(y)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_t + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_t)
        return v_tgt
    elif g_type == "distill":
        cfg_ids = (t >= g_min) & (t <= g_max)
        y_null = null_cond(y)
        with torch.no_grad():
            x_2 = torch.cat([x, x], dim=0)
            t_2 = torch.cat([t, t], dim=0)
            y_2 = torch.cat([y, y_null], dim=0)
            v = model(x_2, t_2, y_2).to(torch.float32)
            v_cond, v_uncond = v.chunk(2, dim=0)
        v_tgt = v_uncond + omega * (v_cond - v_uncond)
        v_tgt = torch.where(cfg_ids, v_tgt, v_cond)  # no class dropout for distillation
        return v_tgt
    else:
        raise NotImplementedError(f"Unknown guidance type: {g_type}")


def ln_sampler(z, path_type="linear"):
    if path_type == "linear":
        return torch.sigmoid(z)
    elif path_type == "cosine":
        return 2 / np.pi * torch.atan(z)
    else:
        raise NotImplementedError(f"Unknown path_type: {path_type}")


class DMFLoss:
    def __init__(
        self,
        P_mean:         float = -0.4,
        P_std:          float = 1.0,
        P_mean_t:       float = -0.2,
        P_std_t:        float = 1.0,
        P_mean_r:       float = 0.2,
        P_std_r:        float = 1.0,
        cfg_prob:       float = 0.1,
        omega:          float = 0.0,
        g_min:          float = 0.0,
        g_max:          float = 1.0,
        path_type:      str   = "linear",
        g_type:         str   = "default",
        use_ot:         bool  = False,
        use_ccot:       bool  = False,
    ):
        if use_ccot and use_ot:
            raise ValueError("use_ccot is mutually exclusive with use_ot.")
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.use_ot = use_ot          # OT coupling for both branches
        self.use_ccot = use_ccot      # class-conditional OT for both branches
        self.guidance_kwargs = dict(
            omega=omega,
            g_min=g_min,
            g_max=g_max,
            g_type=g_type,
        )

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids.reshape(-1, *[1] * (y.ndim - 1)), null_cond(y), y)
        return y, drop_ids

    def sample_times(self, bsz):
        t_fm = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std + self.P_mean, self.path_type)
        ln_1 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_t + self.P_mean_t, self.path_type)
        ln_2 = ln_sampler(torch.randn((bsz, 1, 1, 1)) * self.P_std_r + self.P_mean_r, self.path_type)
        t_mf, r_mf = torch.maximum(ln_1, ln_2), torch.minimum(ln_1, ln_2)
        return t_fm, t_mf, r_mf

    def __call__(self, model, raw_model, x, y, model_t=None, accelerator=None, **kwargs):
        if model_t is None:
            model_t = model
        c = y
        # class dropout
        y, drop_ids = self.cond_drop(y)
        # sample timesteps
        t_fm, t_mf, r_mf = self.sample_times(x.shape[0])
        t_fm, t_mf, r_mf = t_fm.to(x.device), t_mf.to(x.device), r_mf.to(x.device)

        interp_kwargs = dict(
            path_type=self.path_type,
            use_ccot=self.use_ccot,
            c=c,
            accelerator=accelerator,
        )

        # flow matching loss
        x_t_fm, v_t_fm = interpolant(x, t_fm, use_ot=self.use_ot, **interp_kwargs)
        v_fm, lv_fm = model(x_t_fm, t_fm, t_fm, y, return_logvar=True)
        v_tgt_fm = compute_guidance(
            model_t, x_t_fm, t_fm, y, v_t_fm, drop_ids, **self.guidance_kwargs
        ).detach()
        fm_mse_loss, fm_lp_loss = log_lv_loss(v_fm, v_tgt_fm, lv_fm)

        # MeanFlow loss
        x_t_mf, v_t_mf = interpolant(x, t_mf, use_ot=self.use_ot, **interp_kwargs)
        v_tgt_mf = compute_guidance(
            model_t, x_t_mf, t_mf, y, v_t_mf, drop_ids, **self.guidance_kwargs
        ).detach()

        unwrapped_model = raw_model.module if hasattr(raw_model, "module") else raw_model
        model_fn = partial(unwrapped_model, y=y, return_logvar=True)
        primals = (x_t_mf, t_mf, r_mf)
        tangents = (v_tgt_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf))
        (u, lv_mf), (du_dt, _) = torch.func.jvp(model_fn, primals, tangents)
        u_tgt = v_tgt_mf + (r_mf - t_mf) * du_dt
        u_tgt = u_tgt.detach()
        mf_mse_loss, mf_lp_loss = log_lv_loss(u, u_tgt, lv_mf)
        loss = 0.5 * (fm_lp_loss + mf_lp_loss)
        loss_dict = {
            "fm_loss": fm_mse_loss,
            "fm_p_loss": fm_lp_loss,
            "mf_p_loss": mf_lp_loss,
            "mf_loss": mf_mse_loss,
        }
        return loss, loss_dict


class FMLoss:
    def __init__(
        self,
        P_mean:     float = 0.0,
        P_std:      float = 1.0,
        cfg_prob:   float = 0.1,
        path_type:  str = "linear",
        proj_coeff: float = 0.5,
        use_ot:     bool = False,
        use_ccot:   bool = False,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.cfg_prob = cfg_prob
        self.path_type = path_type
        self.proj_coeff = proj_coeff
        self.use_ot = use_ot
        self.use_ccot = use_ccot

    def cond_drop(self, y):
        bsz = y.shape[0]
        drop_ids = torch.rand(bsz, device=y.device) < self.cfg_prob
        y = torch.where(drop_ids.reshape(-1, *[1] * (y.ndim - 1)), null_cond(y), y)
        return y

    def sample_time(self, bsz, P_mean=0.0, P_std=1.0):
        return ln_sampler(torch.randn((bsz, 1, 1, 1)) * P_std + P_mean, self.path_type)

    def __call__(self, model, x, y, zs=None, accelerator=None, **kwargs):
        # true labels for class-conditional OT (kept before cfg dropout maps some to null)
        c = y
        # class dropout
        y = self.cond_drop(y)
        # sample timesteps
        t = self.sample_time(x.shape[0], self.P_mean, self.P_std).to(x.device)
        x_t, v_t = interpolant(
            x, t, self.path_type, use_ot=self.use_ot, use_ccot=self.use_ccot, c=c, accelerator=accelerator
        )
        if zs is not None and self.proj_coeff > 0.:
            v_fm, zs_tilde = model(x_t, t, y, return_feat=True)
            fm_loss, denoising_loss = mse_loss(v_fm, v_t)
            zs_tilde = [zs_tilde_.to(dtype=torch.float32) for zs_tilde_ in zs_tilde]
            proj_loss = repa_loss(zs, zs_tilde)
            loss = denoising_loss + self.proj_coeff * proj_loss
            loss_dict = {
                "fm_loss": fm_loss,
                "proj_loss": proj_loss,
            }
        else:
            v_fm = model(x_t, t, y)
            fm_loss, loss = mse_loss(v_fm, v_t)
            loss_dict = {
                "fm_loss": fm_loss,
            }
        return loss, loss_dict
