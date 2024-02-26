# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
from tqdm import tqdm
import contextlib
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.video.io.bindings import mplfig_to_npimage
from pathlib import Path


class ImprovedDDPM(nn.Module):
    def get_cos_beta_schedule(self, s=0.008):
        # "we selected $s$ such that $\sqrt{\beta_{0}}$ was slightly smaller than the pixel bin size
        # $1 / 127.5$, which gives $s = 0.008$."
        diffusion_step = torch.linspace(0, self.n_diffusion_steps - 1, self.n_diffusion_steps)
        f = torch.cos(((diffusion_step / self.n_diffusion_steps) + s) / (1 + s) * torch.pi / 2) ** 2
        self.alpha_bar = f / f[0]
        self.prev_alpha_bar = torch.cat([torch.ones(size=(1,)), self.alpha_bar], dim=0)[: -1]
        beta = 1 - (self.alpha_bar / self.prev_alpha_bar)
        # "We clip $\beta_{t}$ to be no larger than $0.999$ to prevent singularities at the end
        # of the diffusion process near $t = T$."
        self.beta = torch.clip(beta, 0, 0.999)

    def __init__(
        self,
        model,
        img_size,
        device,
        # "100 sampling steps is sufficient to achieve near-optimal FIDs for our fully trained models."
        n_subsequence_steps=100, # "$K$"
        image_channels=3,
        # "We could get a boost in log-likelihood by increasing $T$ from 1000 to 4000."
        n_diffusion_steps=4000, # "$T$"
        # "We set $\lambda = 0.001$ to prevent $L_{\text{vlb}}$ from overwhelming $L_{\text{simple}}$."
        vlb_weight=0.001,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_subsequence_steps = n_subsequence_steps
        self.n_diffusion_steps = n_diffusion_steps
        self.vlb_weight = vlb_weight

        self.model = model.to(device)

        self.get_cos_beta_schedule()
        # print(self.beta.shape, self.alpha_bar.shape, self.prev_alpha_bar.shape)

        # "$\tilde{\beta_{t}} = \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_{t}}\beta_{t}$"
        # self.beta_tilde = ((1 - self.prev_alpha_bar) / (1 - self.alpha_bar)) * self.beta
        
        # "To reduce the number of sampling steps from $$ to $K$,
        # we use $K$ evenly spaced real numbers between $1$ and $T$ (inclusive),
        # and then round each resulting number to the nearest integer."
        self.subsequence_step = torch.linspace(
            0, self.n_diffusion_steps - 1, self.n_subsequence_steps, dtype=torch.long, device=self.device
        )

    @staticmethod
    def index(x, diffusion_step):
        return torch.index_select(
            x,
            dim=0,
            index=torch.maximum(diffusion_step, torch.zeros_like(diffusion_step)),
        )[:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def get_model_var(self, v, diffusion_step):
        return self.index(
            v * torch.log(self.beta) + (1 - v) * torch.log(self.beta_tilde),
            diffusion_step=diffusion_step,
        )

    def forward(self, noisy_image, diffusion_step):
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    def get_mu_tilde(self, ori_image, noisy_image, diffusion_step):
        return self.index(
            (self.prev_alpha_bar ** 0.5) / (1 - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * ori_image + self.index(
            ((self.alpha_bar ** 0.5) * (1 - self.prev_alpha_bar)) / (1  - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * noisy_image

    @torch.inference_mode()
    def sample(self, batch_size):
        x = self.sample_noise(batch_size=batch_size)
        for subsequence_idx in tqdm(range(self.n_subsequence_steps - 1, 0, -1)):
            batched_subsequence_idx = self.batchify_diffusion_steps(diffusion_step_idx=subsequence_idx, batch_size=batch_size)
            cur_step = self.index(self.subsequence_step, batched_subsequence_idx)
            prev_step = self.index(self.subsequence_step, batched_subsequence_idx - 1)

            alpha_bar_t = self.alpha_bar.to(self.device)[cur_step]
            prev_alpha_bar_t = self.alpha_bar.to(self.device)[prev_step]
            beta_t = 1 - alpha_bar_t / prev_alpha_bar_t

            # print(x.shape, cur_step.shape)
            pred_noise = self(noisy_image=x.detach(), diffusion_step=cur_step[:, 0, 0, 0])
            model_mean = (1 / ((1 - beta_t) ** 0.5)) * (
                x - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
            )
            model_var = beta_t
            rand_noise = self.sample_noise(batch_size=batch_size)
            x = model_mean + (model_var ** 0.5) * rand_noise
        return x


if __name__ == "__main__":
    n_diffusion_steps = 4000
    init_beta = 0.0001
    fin_beta = 0.02
    linear_beta = get_linear_beta_schdule(
        init_beta=init_beta, fin_beta=fin_beta, n_diffusion_steps=n_diffusion_steps,
    )
    cos_beta = get_cos_beta_schedule(n_diffusion_steps=4000)

    linear_alpha = 1 - linear_beta
    linear_alpha_bar = torch.cumprod(linear_alpha, dim=0)

    cos_alpha = 1 - cos_beta
    cos_alpha_bar = torch.cumprod(cos_alpha, dim=0)
    # linear_alpha_bar[0]
    # cos_alpha_bar[0]

    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    line2 = axes.plot(linear_alpha_bar.numpy(), label="Linear")
    line2 = axes.plot(cos_alpha_bar.numpy(), label="Cosine")
    # line2 = axes.plot((linear_alpha_bar.numpy() ** 0.5))
    # line2 = axes.plot((cos_alpha_bar.numpy() ** 0.5))
    axes.legend(fontsize=6)
    axes.tick_params(labelsize=7)
    fig.tight_layout()
    image = plt_to_pil(fig)
    # image.show()
    save_image(image, path="/Users/jongbeomkim/Desktop/workspace/Dhariwal-and-Nichol/beta_schedules.jpg")
