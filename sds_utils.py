from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

from utils import sd_weight_policy

device = torch.device('cuda')

@torch.no_grad()
def get_text_embeddings(tokenizer, text_encoder, text):
    tokens = tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                   return_tensors="pt", return_overflowing_tokens=True).input_ids.to(device)
    return text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent, vae, im_cat= None):
    image = vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)

def init_pipe(device, dtype, unet, scheduler):

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas


class SDSLoss:

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32, t_min=50, t_max=950, alpha_exp=0, sigma_exp=0):
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_exp = alpha_exp
        self.sigma_exp = sigma_exp
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.prediction_type

         
    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        # print('#####time step:', timestep)
        if eps is None:
            eps = torch.randn_like(z, dtype=self.dtype)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t, timestep, text_embeddings, alpha_t, sigma_t, get_raw=False,
                           guidance_scale=7.5):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        e_t = self.unet(latent_input, timestep, embedd).sample
        # breakpoint()
        if self.prediction_type == 'v_prediction':
            e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
        e_t_uncond, e_t = e_t.chunk(2)
        if get_raw:
            return e_t_uncond, e_t
        norm1 = e_t_uncond.norm()
        norm2 = (e_t - e_t_uncond).norm()
        # print('uncond norm:', norm1, 'guidance norm:', norm2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        e_t_discri = guidance_scale * (e_t - e_t_uncond)
        # e_t = guidance_scale * (e_t - e_t_uncond)
        # breakpoint()
        assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, e_t_uncond, e_t_discri, pred_z0

    def get_sds_loss(self, z, text_embeddings, eps=None, mask=None, t=None,
                 timestep: Optional[int] = None, guidance_scale=7.5, use_policy=False):
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            # breakpoint()
            e_t, e_t_uncond, e_t_discri, _ = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            w_t = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * 2
            if use_policy:
                w_t = sd_weight_policy(timestep.item(), self.t_min, self.t_max)
            grad_z = w_t * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
            if mask is not None:
                grad_z = grad_z * mask
            log_loss = (grad_z ** 2).mean()
        sds_loss = grad_z.clone() * z
        del grad_z
        return sds_loss.sum() / (z.shape[2] * z.shape[3]), log_loss, e_t_uncond, e_t_discri
