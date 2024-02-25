# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion
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
    def get_linear_beta_schdule(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )

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
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.get_cos_beta_schedule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.beta_tilde = ((1 - self.prev_alpha_bar) / (1 - self.alpha_bar)) * self.beta

        self.model = model.to(device)

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

    def perform_diffusion_process(self, ori_image, diffusion_step, random_noise=None):
        alpha_bar_t = self.index(self.new_alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image
        var = 1 - alpha_bar_t
        if random_noise is None:
            random_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * random_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    @torch.inference_mode()
    def predict_ori_image(self, noisy_image, noise, alpha_bar_t):
        return (noisy_image - ((1 - alpha_bar_t) ** 0.5) * noise) / (alpha_bar_t ** 0.5)

    def get_mu_tilde(self, ori_image, noisy_image, diffusion_step):
        return self.index(
            (self.prev_alpha_bar ** 0.5) / (1 - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * ori_image + self.index(
            ((self.alpha_bar ** 0.5) * (1 - self.prev_alpha_bar)) / (1  - self.alpha_bar),
            diffusion_step=diffusion_step,
        ) * noisy_image

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        pred_noise = self(noisy_image=noisy_image, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_ori_image  = self.predict_ori_image(
            noisy_image=noisy_image, noise=pred_noise, alpha_bar_t=alpha_bar_t,
        )


if __name__ == "__main__":
    n_diffusion_steps = 1000
    init_beta = 0.0001
    fin_beta = 0.02
    linear_beta = get_linear_beta_schdule(
        init_beta=init_beta, fin_beta=fin_beta, n_diffusion_steps=n_diffusion_steps,
    )
    cos_beta = get_cos_beta_schedule(n_diffusion_steps=1000)

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
