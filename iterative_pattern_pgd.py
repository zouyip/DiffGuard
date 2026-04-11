import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PretrainedConfig


class TwoBlockMasker(nn.Module):
    def __init__(self, block_size=128):
        super().__init__()
        self.block_size = block_size

    def forward(self, delta):
        if not self.training:
            return delta

        bsz, _, height, width = delta.shape
        grid_h = height // self.block_size
        grid_w = width // self.block_size
        num_blocks = grid_h * grid_w

        if num_blocks < 3:
            return delta

        mask_flat = torch.ones((bsz, num_blocks), device=delta.device)
        rand_vals = torch.rand((bsz, num_blocks), device=delta.device)
        mask_indices = torch.argsort(rand_vals, dim=1)[:, :3]
        mask_flat.scatter_(1, mask_indices, 0.0)
        mask_grid = mask_flat.view(bsz, 1, grid_h, grid_w)
        mask = F.interpolate(mask_grid, size=(height, width), mode="nearest")
        return delta * mask


class FrequencyCarrier(nn.Module):
    def __init__(self, size=512, freqs=(20, 40, 80)):
        super().__init__()
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid_x", grid_x[None, None])
        self.register_buffer("grid_y", grid_y[None, None])
        self.freqs = tuple(freqs)

    def get_carriers(self, batch_size):
        carriers = []
        for omega in self.freqs:
            wx = omega * torch.pi * self.grid_x
            wy = omega * torch.pi * self.grid_y
            carriers.extend([torch.sin(wx), torch.cos(wx), torch.sin(wy), torch.cos(wy)])
        return torch.cat(carriers, dim=1).expand(batch_size, -1, -1, -1)


def total_variation_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def variance_alignment_loss(latent_clean_dist, latent_adv_dist, latent_pattern_dist, margin_mean=3.0, margin_var=1.0):
    mean_clean = latent_clean_dist.mean.float().detach()
    logvar_clean = latent_clean_dist.logvar.float().detach()
    mean_adv = latent_adv_dist.mean.float()
    logvar_adv = latent_adv_dist.logvar.float()
    mean_pattern = latent_pattern_dist.mean.float().detach()
    logvar_pattern = latent_pattern_dist.logvar.float().detach()

    dist_pos_mean = F.mse_loss(mean_adv, mean_pattern, reduction="none").mean(dim=[1, 2, 3])
    dist_pos_var = F.mse_loss(logvar_adv, logvar_pattern, reduction="none").mean(dim=[1, 2, 3])
    dist_neg_mean = F.mse_loss(mean_adv, mean_clean, reduction="none").mean(dim=[1, 2, 3])
    dist_neg_var = F.mse_loss(logvar_adv, logvar_clean, reduction="none").mean(dim=[1, 2, 3])

    loss_mean = F.relu(dist_pos_mean - dist_neg_mean + margin_mean).mean()
    loss_var = F.relu(dist_pos_var - dist_neg_var + margin_var).mean()
    return loss_mean + loss_var


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    raise ValueError(f"{model_class} is not supported.")


class CustomDiffusionDataset(Dataset):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, instance_data_dir, instance_prompt, tokenizer, size=512, center_crop=False):
        self.tokenizer = tokenizer
        root = Path(instance_data_dir)
        self.instance_images_path = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXT]
        if not self.instance_images_path:
            raise ValueError(f"No images found in {instance_data_dir}")

        random.shuffle(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        image = Image.open(self.instance_images_path[index]).convert("RGB")
        image = self.image_transforms(image)
        input_ids = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return {"instance_images": image, "instance_prompt_ids": input_ids}


@torch.enable_grad()
def build_semantic_mask(
    images,
    input_ids,
    vae,
    unet,
    text_encoder,
    noise_scheduler,
    weight_dtype,
    topk_ratio=0.35,
):
    probe = images.detach().clone().requires_grad_(True)
    latents = vae.encode(probe.to(dtype=weight_dtype) * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
    probe_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    grad = torch.autograd.grad(probe_loss, probe)[0]

    score = grad.abs().mean(dim=1, keepdim=True)
    score = F.avg_pool2d(score, kernel_size=15, stride=1, padding=7)
    score = score / (score.amax(dim=(2, 3), keepdim=True) + 1e-8)
    flat = score.flatten(2)
    keep_num = max(1, int(flat.shape[-1] * topk_ratio))
    thresh = flat.topk(keep_num, dim=-1).values[:, :, -1:]
    hard_mask = (flat >= thresh).float().view_as(score)
    soft_mask = F.avg_pool2d(hard_mask, kernel_size=21, stride=1, padding=10)
    return soft_mask.clamp(0.0, 1.0).detach()
