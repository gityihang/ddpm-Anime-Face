
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
    

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # a_t
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T] # a_t-1

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        # var = self.posterior_var

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


class DDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_timesteps, eta=0.0):
        super().__init__()

        self.model = model
        self.T = T
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # a_t

        # 不能使用prev来计算，因为时间t不是连续的, 要通过时间计算prev 
        self.register_buffer('alphas_bar', alphas_bar)

        step = self.T // self.ddim_timesteps
        self.ddim_timestep_seq = np.asarray(list(range(0, self.T, step))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])
        

    def predict_xt_prev_mean_from_eps(self, x_t, t, prev_t):
        eps = self.model(x_t, t)
        assert x_t.shape == eps.shape
        alphas_bar = extract(self.alphas_bar, t, x_t.shape) 
        alphas_bar_prev = extract(self.alphas_bar, prev_t, x_t.shape)

        pred_x0 = (x_t - torch.sqrt(1. - alphas_bar) * eps) / torch.sqrt(alphas_bar)
        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
        sigma_t = self.eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar) * (1 - alphas_bar / alphas_bar_prev))
        pred_dir_xt = torch.sqrt(1. - alphas_bar_prev - sigma_t ** 2) * eps
        x_prev = torch.sqrt(alphas_bar_prev) * pred_x0 + pred_dir_xt + sigma_t * torch.randn_like(x_t)
        return x_prev
        

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.ddim_timesteps)):  # [T, T-1, T-2, ..., 0]
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_seq[time_step]   # batch_size个t构成一个数组
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_prev_seq[time_step]
            mean = self.predict_xt_prev_mean_from_eps(x_t, t, prev_t)
            x_t = mean
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            # 保存去噪全过程
            torchvision.utils.save_image(torch.clip(x_t, -1, 1)*0.5 + 0.5, "./image-1/{}.jpg".format(time_step))
        x_0 = x_t
        return torch.clip(x_0, -1, 1)




class MyDDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_timesteps, eta=0.0):
        super().__init__()

        self.model = model
        self.T = T
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # a_t

        # 不能使用prev来计算，因为时间t不是连续的, 要通过时间计算prev 
        self.register_buffer('alphas_bar', alphas_bar)

        step = self.T // self.ddim_timesteps
        self.ddim_timestep_seq = np.asarray(list(range(0, self.T, step))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])
        

    def predict_xt_prev_mean_from_eps(self, x_t, t, prev_t):
        eps = self.model(x_t, t)
        assert x_t.shape == eps.shape
        alphas_bar = extract(self.alphas_bar, t, x_t.shape) 
        alphas_bar_prev = extract(self.alphas_bar, prev_t, x_t.shape)

        pred_x0 = (x_t - torch.sqrt(1. - alphas_bar) * eps) / torch.sqrt(alphas_bar)
        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
        sigma_t = self.eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar) * (1 - alphas_bar / alphas_bar_prev))
        pred_dir_xt = torch.sqrt(1. - alphas_bar_prev - sigma_t ** 2) * eps
        x_prev = torch.sqrt(alphas_bar_prev) * pred_x0 + pred_dir_xt + sigma_t * torch.randn_like(x_t)
        return x_prev
        

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.ddim_timesteps)):  # [T, T-1, T-2, ..., 0]
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_seq[time_step]   # batch_size个t构成一个数组
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_prev_seq[time_step]
            mean = self.predict_xt_prev_mean_from_eps(x_t, t, prev_t)
            x_t = mean
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            # 保存去噪全过程
            torchvision.utils.save_image(torch.clip(x_t, -1, 1)*0.5 + 0.5, "./image-1/{}.jpg".format(time_step))
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
