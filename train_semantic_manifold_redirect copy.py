"""Semantic Manifold Redirect PGD.

Instead of creating fragile perturbations near the original image's semantics,
stably redirect the sample to a perturbation-dominated, naturally coherent
target semantic manifold within L_inf budget.  The diffusion model treats the
result as valid data, so purification (DiffPure, JPEG, denoising) PRESERVES
and even STRENGTHENS the redirect.

Core innovation: multi-timestep score redirect
-----------------------------------------------
At each timestep t the UNet prediction target is analytically computed so
that the model denoises x_adv toward x_target:

  epsilon mode:  eps_target = (z_noisy - sqrt(ab_t) * z_target) / sqrt(1 - ab_t)
  v-pred mode:   v_target   = (sqrt(ab_t) * z_noisy - z_target) / sqrt(1 - ab_t)

When z_adv -> z_target the target converges to the standard DSM target,
giving a smooth transition from "redirect" to "on-manifold".

Why purification STRENGTHENS the attack
-----------------------------------------
Standard PGD: x_adv sits in the void between manifolds.
DiffPure projects back to the source manifold -> attack removed.

This method: x_adv sits ON the target manifold.
DiffPure projects onto the SAME target manifold -> attack PRESERVED.

Losses (all minimised, gradient-descent PGD)
---------------------------------------------
1. Semantic anchor  — multi-timestep UNet score redirect toward z_target
2. Latent redirect  — direct VAE latent push toward target distribution
3. Source escape    — hinge loss ensuring distance from clean exceeds margin
4. TV              — spatial smoothness of the perturbation

Example
-------
python train_semantic_manifold_redirect.py \
    --input_dir /usr/zou/DiffGuard/data/instance_1 \
    --target_image_path /usr/zou/DiffGuard/data/target/0.png \
    --output_dir outputs/semantic_redirect \
    --eps 0.05 \
    --inner_steps 100 \
    --init_from_target \
    --smooth_grad \
    --smooth_delta

python train_semantic_manifold_redirect.py \
    --input_dir /usr/zou/DiffGuard/data/instance_1 \
    --target_image_path /usr/zou/DiffGuard/data/target/0.png \
    --output_dir outputs/semantic_redirect_strong \
    --eps 0.05 \
    --inner_steps 200 \
    --step_size 0.00392156862745098 \
    --lambda_anchor 1.5 \
    --lambda_redirect 2.0 \
    --lambda_escape 1.0 \
    --num_t_per_step 8 \
    --diffpure_fraction 0.6 \
    --init_from_target \
    --smooth_grad \
    --smooth_delta
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from iterative_pattern_pgd import (
    TwoBlockMasker,
    total_variation_loss,
    import_model_class_from_model_name_or_path,
)


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════


class ImageDirectoryDataset(Dataset):
    def __init__(self, input_dir, prompt, tokenizer, resolution=512, center_crop=False):
        self.input_dir = Path(input_dir)
        self.paths = sorted(
            p for p in self.input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXT
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.input_dir}")
        self.prompt_ids = tokenizer(
            prompt, truncation=True, padding="max_length",
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        if center_crop:
            self.transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return {
            "instance_images": self.transform(Image.open(path).convert("RGB")),
            "instance_prompt_ids": self.prompt_ids.clone(),
            "image_path": str(path),
        }


class TargetImageBank:
    def __init__(self, resolution=512, center_crop=False,
                 target_dir=None, target_image_path=None):
        if target_image_path is not None:
            p = Path(target_image_path)
            if not p.is_file():
                raise ValueError(f"Target image not found: {p}")
            self.paths = [p]
        elif target_dir is not None:
            self.paths = sorted(
                p for p in Path(target_dir).rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXT
            )
            if not self.paths:
                raise ValueError(f"No target images in {target_dir}")
        else:
            raise ValueError("Provide target_dir or target_image_path")
        self.cursor = 0
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def sample(self, batch_size, device):
        tensors, paths = [], []
        for _ in range(batch_size):
            p = self.paths[self.cursor % len(self.paths)]
            self.cursor += 1
            paths.append(str(p))
            tensors.append(self.transform(Image.open(p).convert("RGB")))
        return torch.stack(tensors).to(device), paths


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════════════════════


def _gaussian_kernel_2d(k, sigma):
    c = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-c ** 2 / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum())[None, None]


class GaussianSmoother:
    def __init__(self, k=15, sigma=3.0):
        self._k = _gaussian_kernel_2d(k, sigma)
        self._pad = k // 2

    def __call__(self, x):
        C = x.shape[1]
        w = self._k.to(device=x.device, dtype=x.dtype).expand(C, 1, -1, -1)
        return F.conv2d(x, w, padding=self._pad, groups=C)


# ═══════════════════════════════════════════════════════════════════════════════
# Core PGD attacker
# ═══════════════════════════════════════════════════════════════════════════════


class SemanticManifoldRedirectPGD:
    """Redirect x+delta onto target semantic manifold via multi-timestep score
    alignment.  All losses are minimised (gradient-descent PGD).
    """

    def __init__(
        self,
        eps=0.05,
        inner_steps=100,
        num_restarts=1,
        step_size=2.0 / 255.0,
        lambda_anchor=1.0,
        lambda_redirect=1.0,
        lambda_escape=0.5,
        lambda_tv=0.0,
        num_t_per_step=4,
        diffpure_fraction=0.5,
        t_min=50,
        t_max=250,
        margin_mean=3.0,
        margin_var=1.0,
        random_init=False,
        init_from_target=False,
        smooth_grad=False,
        smooth_delta=False,
        smooth_kernel_size=15,
        smooth_sigma=3.0,
    ):
        self.eps = eps
        self.inner_steps = inner_steps
        self.num_restarts = max(1, num_restarts)
        self.step_size = step_size
        self.lambda_anchor = lambda_anchor
        self.lambda_redirect = lambda_redirect
        self.lambda_escape = lambda_escape
        self.lambda_tv = lambda_tv
        self.num_t_per_step = max(1, num_t_per_step)
        self.diffpure_fraction = diffpure_fraction
        self.t_min = t_min
        self.t_max = t_max
        self.margin_mean = margin_mean
        self.margin_var = margin_var
        self.random_init = random_init
        self.init_from_target = init_from_target
        self.smooth_grad = smooth_grad
        self.smooth_delta = smooth_delta
        self._smoother = GaussianSmoother(smooth_kernel_size, smooth_sigma)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _sample_timesteps(self, n, device):
        """Mix of DiffPure-range and full-range timesteps."""
        n_dp = max(1, int(n * self.diffpure_fraction))
        n_full = n - n_dp
        t_dp = torch.randint(self.t_min, self.t_max + 1, (n_dp,), device=device)
        t_full = torch.randint(0, 1000, (n_full,), device=device)
        return torch.cat([t_dp, t_full]).long()

    @staticmethod
    def _redirect_target(z_noisy, z_target, timesteps, scheduler):
        """Analytic noise-prediction target that makes the UNet denoise
        z_noisy toward z_target.

        Derivation (epsilon mode):
            z_noisy = sqrt(ab_t) * z_adv + sqrt(1-ab_t) * noise
            We want the model's denoised estimate z_0_hat = z_target:
                z_0_hat = (z_noisy - sqrt(1-ab_t) * eps_pred) / sqrt(ab_t)
            Setting z_0_hat = z_target and solving:
                eps_target = (z_noisy - sqrt(ab_t) * z_target) / sqrt(1-ab_t)

        When z_adv == z_target  =>  eps_target == noise  (standard DSM).
        """
        ab = scheduler.alphas_cumprod[timesteps].to(z_noisy.device)
        s1 = torch.sqrt(ab).view(-1, 1, 1, 1)
        s2 = torch.sqrt(1.0 - ab).view(-1, 1, 1, 1)
        if getattr(scheduler.config, "prediction_type", "epsilon") == "v_prediction":
            return (s1 * z_noisy - z_target) / s2
        return (z_noisy - s1 * z_target) / s2

    def _init_delta(self, images, target_images, use_ri):
        if self.init_from_target and target_images is not None:
            delta = (target_images - images).clamp(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(images)
        if use_ri:
            noise = torch.zeros_like(images).uniform_(-self.eps, self.eps)
            if self.smooth_delta:
                noise = self._smoother(noise)
            delta = (delta + noise).clamp(-self.eps, self.eps)
        return delta

    # ------------------------------------------------------------------ #
    #  Loss                                                               #
    # ------------------------------------------------------------------ #

    def _compute_loss(
        self, images, x_adv, delta,
        clean_dist, target_latent_mean,
        vae, unet, scheduler, weight_dtype, encoder_hs,
    ):
        bsz = images.shape[0]
        sc = vae.config.scaling_factor

        # encode adversarial image
        with torch.cuda.amp.autocast(enabled=(weight_dtype != torch.float32)):
            adv_dist = vae.encode(x_adv.to(dtype=weight_dtype) * 2.0 - 1.0).latent_dist

        z_adv = adv_dist.sample().float() * sc
        z_target = target_latent_mean.float() * sc          # deterministic mode

        # ── (1) Semantic anchor: multi-timestep score redirect ──
        total_t = bsz * self.num_t_per_step
        timesteps = self._sample_timesteps(total_t, images.device)

        if self.num_t_per_step > 1:
            z_t = z_adv.repeat(self.num_t_per_step, 1, 1, 1)
            z_tgt_t = z_target.repeat(self.num_t_per_step, 1, 1, 1)
            hs = encoder_hs.repeat(self.num_t_per_step, 1, 1)
        else:
            z_t = z_adv
            z_tgt_t = z_target
            hs = encoder_hs

        noise = torch.randn_like(z_t)
        z_noisy = scheduler.add_noise(z_t, noise, timesteps)
        eps_tgt = self._redirect_target(z_noisy, z_tgt_t, timesteps, scheduler).detach()

        with torch.cuda.amp.autocast(enabled=(weight_dtype != torch.float32)):
            eps_pred = unet(z_noisy, timesteps, hs).sample.float()

        anchor_loss = F.mse_loss(eps_pred, eps_tgt)

        # ── (2) Latent redirect: push distribution mean toward target ──
        redirect_loss = F.mse_loss(
            adv_dist.mean.float(), target_latent_mean.float().detach(),
        )

        # ── (3) Source escape: hinge loss away from clean ──
        d_m = F.mse_loss(
            adv_dist.mean.float(), clean_dist.mean.float().detach(),
            reduction="none",
        ).mean(dim=[1, 2, 3])
        d_v = F.mse_loss(
            adv_dist.logvar.float(), clean_dist.logvar.float().detach(),
            reduction="none",
        ).mean(dim=[1, 2, 3])
        escape_loss = (
            F.relu(self.margin_mean - d_m).mean()
            + F.relu(self.margin_var - d_v).mean()
        )

        # ── (4) TV ──
        tv_loss = (
            total_variation_loss(delta) if self.lambda_tv > 0
            else delta.new_zeros(())
        )

        total = (
            self.lambda_anchor * anchor_loss
            + self.lambda_redirect * redirect_loss
            + self.lambda_escape * escape_loss
            + self.lambda_tv * tv_loss
        )
        return total, anchor_loss, redirect_loss, escape_loss, tv_loss

    # ------------------------------------------------------------------ #
    #  Main optimisation loop                                             #
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        images, input_ids,
        target_images,
        vae, unet, text_encoder, scheduler, weight_dtype,
        block_masker=None,
    ):
        images = images.float()
        device = images.device

        with torch.no_grad():
            clean_dist = vae.encode(
                images.to(dtype=weight_dtype) * 2.0 - 1.0,
            ).latent_dist
            target_dist = vae.encode(
                target_images.to(dtype=weight_dtype) * 2.0 - 1.0,
            ).latent_dist
            target_latent_mean = target_dist.mean
            encoder_hs = text_encoder(input_ids)[0].to(dtype=weight_dtype)

        best_outputs, best_score = None, None

        for ri in range(self.num_restarts):
            delta_state = self._init_delta(
                images, target_images, self.random_init or ri > 0,
            )
            ri_best_out, ri_best_score = None, None

            for step in range(self.inner_steps):
                dv = delta_state.detach().clone().requires_grad_(True)
                fd = dv.clamp(-self.eps, self.eps)
                df = block_masker(fd) if block_masker else fd
                xa = (images + df).clamp(0.0, 1.0)
                xp = (images + fd).clamp(0.0, 1.0)

                L, aL, rL, eL, tL = self._compute_loss(
                    images, xa, fd, clean_dist, target_latent_mean,
                    vae, unet, scheduler, weight_dtype, encoder_hs,
                )

                g = torch.autograd.grad(
                    L, dv, retain_graph=False, create_graph=False,
                )[0]

                with torch.no_grad():
                    if self.smooth_grad:
                        g = self._smoother(g)
                    delta_state.add_(-self.step_size * g.sign())
                    if self.smooth_delta:
                        delta_state.data = self._smoother(delta_state)
                    delta_state.clamp_(-self.eps, self.eps)

                s = float(L.item())
                out = {
                    "x_protected": xp.detach(),
                    "final_delta": fd.detach(),
                    "stats": {
                        "restart": ri, "step": step, "total": s,
                        "anchor": float(aL), "redirect": float(rL),
                        "escape": float(eL), "tv": float(tL),
                        "linf": float(fd.abs().amax().item()),
                    },
                }
                if ri_best_out is None or s < ri_best_score:
                    ri_best_out, ri_best_score = out, s

            # final evaluation of this restart
            with torch.no_grad():
                fd = delta_state.clamp(-self.eps, self.eps)
                df = block_masker(fd) if block_masker else fd
                xa = (images + df).clamp(0.0, 1.0)
                xp = (images + fd).clamp(0.0, 1.0)

                L, aL, rL, eL, tL = self._compute_loss(
                    images, xa, fd, clean_dist, target_latent_mean,
                    vae, unet, scheduler, weight_dtype, encoder_hs,
                )
                fs = float(L.item())
                fout = {
                    "x_protected": xp.detach(),
                    "final_delta": fd.detach(),
                    "stats": {
                        "restart": ri, "step": self.inner_steps, "total": fs,
                        "anchor": float(aL), "redirect": float(rL),
                        "escape": float(eL), "tv": float(tL),
                        "linf": float(fd.abs().amax().item()),
                    },
                }

            if ri_best_out is None or fs < ri_best_score:
                ri_best_out, ri_best_score = fout, fs
            if best_outputs is None or ri_best_score < best_score:
                best_outputs, best_score = ri_best_out, ri_best_score

        if best_outputs is None:
            raise RuntimeError("PGD produced no outputs")
        return best_outputs


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def tensor_to_image(t):
    return Image.fromarray(
        (t.detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8),
    )


def delta_to_vis(d):
    s = max(float(d.detach().float().abs().amax().item()), 1e-8)
    v = ((d.detach().float().cpu().permute(1, 2, 0).numpy() / (2 * s)) + 0.5).clip(0, 1)
    return Image.fromarray((v * 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════════
# Saving
# ═══════════════════════════════════════════════════════════════════════════════


def save_batch(output_dir, input_root, batch_paths, clean_images,
               target_images, outputs, save_clean=False, save_raw=False):
    out = Path(output_dir)
    prot_root = out / "protected"
    delta_root = out / "delta_vis"
    target_root = out / "target_ref"
    for d in (prot_root, delta_root, target_root):
        d.mkdir(parents=True, exist_ok=True)

    clean_root = out / "clean_resized"
    raw_root = out / "delta_tensors"
    if save_clean:
        clean_root.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_root.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, pstr in enumerate(batch_paths):
        rel = Path(pstr).relative_to(input_root)
        pp = (prot_root / rel).with_suffix(".png")
        dp = (delta_root / rel).with_suffix(".png")
        tp = (target_root / rel).with_suffix(".png")
        for p in (pp, dp, tp):
            p.parent.mkdir(parents=True, exist_ok=True)

        tensor_to_image(outputs["x_protected"][idx]).save(pp)
        delta_to_vis(outputs["final_delta"][idx]).save(dp)
        tensor_to_image(target_images[idx]).save(tp)

        cp = None
        if save_clean:
            cp = (clean_root / rel).with_suffix(".png")
            cp.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_image(clean_images[idx]).save(cp)

        rp = None
        if save_raw:
            rp = (raw_root / rel).with_suffix(".pt")
            rp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"delta": outputs["final_delta"][idx:idx+1].cpu()}, rp)

        records.append({
            "source": str(Path(pstr)),
            "protected": str(pp),
            "target_ref": str(tp),
            "clean_resized": str(cp) if cp else None,
            "delta_vis": str(dp),
            "delta_tensor": str(rp) if rp else None,
            "stats": outputs["stats"],
        })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Manifold Redirect PGD")

    # model
    p.add_argument("--pretrained_model", type=str, default="/usr/zou/CAAT/sd1.5/")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--input_dir", type=str, default="/usr/zou/DiffGuard/data/instance_1")
    p.add_argument("--target_dir", type=str, default=None)
    p.add_argument("--target_image_path", type=str, default="/usr/zou/DiffGuard/data/target/0.png")
    p.add_argument("--instance_prompt", type=str, default="a photo of a sks person")
    p.add_argument("--output_dir", type=str, default="outputs/semantic_manifold_redirect")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--sample_batch_size", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="fp16")
    p.add_argument("--seed", type=int, default=42)

    # PGD
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--inner_steps", type=int, default=100)
    p.add_argument("--num_restarts", type=int, default=1)
    p.add_argument("--step_size", type=float, default=2.0 / 255.0)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--init_from_target", action="store_true",
                   help="Initialise delta as (target - source) clamped to eps ball")

    # loss weights
    p.add_argument("--lambda_anchor", type=float, default=1.0,
                   help="Weight for multi-timestep semantic score redirect")
    p.add_argument("--lambda_redirect", type=float, default=1.0,
                   help="Weight for direct VAE latent redirect toward target")
    p.add_argument("--lambda_escape", type=float, default=0.5,
                   help="Weight for source-escape hinge loss")
    p.add_argument("--lambda_tv", type=float, default=0.0,
                   help="Weight for total-variation regularisation")

    # timestep config
    p.add_argument("--num_t_per_step", type=int, default=4,
                   help="Timestep samples per PGD inner step")
    p.add_argument("--diffpure_fraction", type=float, default=0.5,
                   help="Fraction of timesteps sampled from DiffPure range [t_min, t_max]")
    p.add_argument("--t_min", type=int, default=50)
    p.add_argument("--t_max", type=int, default=250)

    # margins
    p.add_argument("--margin_mean", type=float, default=3.0)
    p.add_argument("--margin_var", type=float, default=1.0)

    # smoothing
    p.add_argument("--smooth_grad", action="store_true",
                   help="Gaussian-blur PGD gradient (TI-PGD style)")
    p.add_argument("--smooth_delta", action="store_true",
                   help="Gaussian-blur delta after each PGD step")
    p.add_argument("--smooth_kernel_size", type=int, default=15)
    p.add_argument("--smooth_sigma", type=float, default=3.0)

    # block mask
    p.add_argument("--disable_block_mask", action="store_true")
    p.add_argument("--block_size", type=int, default=128)

    # output
    p.add_argument("--save_clean_copy", action="store_true")
    p.add_argument("--save_raw_delta", action="store_true")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Models (all frozen) ──
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", use_fast=False)
    te_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model, args.revision)
    text_encoder = te_cls.from_pretrained(
        args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()

    wd = torch.float32
    if args.mixed_precision == "fp16":
        wd = torch.float16
    elif args.mixed_precision == "bf16":
        wd = torch.bfloat16
    vae.to(device, dtype=wd)
    text_encoder.to(device, dtype=wd)
    unet.to(device, dtype=wd)

    # ── Block masker ──
    bm = None
    if not args.disable_block_mask:
        bm = TwoBlockMasker(block_size=args.block_size).to(device)
        bm.train()

    # ── Target bank ──
    target_bank = TargetImageBank(
        resolution=args.resolution,
        center_crop=args.center_crop,
        target_dir=args.target_dir,
        target_image_path=args.target_image_path,
    )

    # ── Dataset ──
    dataset = ImageDirectoryDataset(
        input_dir=args.input_dir,
        prompt=args.instance_prompt,
        tokenizer=tokenizer,
        resolution=args.resolution,
        center_crop=args.center_crop,
    )
    dl = DataLoader(
        dataset, batch_size=args.sample_batch_size,
        shuffle=False, num_workers=args.dataloader_num_workers,
        pin_memory=True, drop_last=False,
    )

    # ── Attacker ──
    attacker = SemanticManifoldRedirectPGD(
        eps=args.eps,
        inner_steps=args.inner_steps,
        num_restarts=args.num_restarts,
        step_size=args.step_size,
        lambda_anchor=args.lambda_anchor,
        lambda_redirect=args.lambda_redirect,
        lambda_escape=args.lambda_escape,
        lambda_tv=args.lambda_tv,
        num_t_per_step=args.num_t_per_step,
        diffpure_fraction=args.diffpure_fraction,
        t_min=args.t_min,
        t_max=args.t_max,
        margin_mean=args.margin_mean,
        margin_var=args.margin_var,
        random_init=args.random_init,
        init_from_target=args.init_from_target,
        smooth_grad=args.smooth_grad,
        smooth_delta=args.smooth_delta,
        smooth_kernel_size=args.smooth_kernel_size,
        smooth_sigma=args.smooth_sigma,
    )

    # ── Run ──
    print("***** Semantic Manifold Redirect PGD *****")
    print(f"  eps              : {args.eps}")
    print(f"  inner_steps      : {args.inner_steps}")
    print(f"  lambda_a/r/e/tv  : {args.lambda_anchor}/{args.lambda_redirect}"
          f"/{args.lambda_escape}/{args.lambda_tv}")
    print(f"  t_per_step       : {args.num_t_per_step}"
          f"  (diffpure_frac={args.diffpure_fraction})")
    print(f"  init_from_target : {args.init_from_target}")
    print(f"  Images           : {len(dataset)}")

    records = []
    pbar = tqdm(dl, desc="SemanticRedirect PGD")

    for batch in pbar:
        images = batch["instance_images"].to(device)
        input_ids = batch["instance_prompt_ids"].to(device)
        paths = batch["image_path"]

        target_images, target_paths = target_bank.sample(images.shape[0], device)

        outputs = attacker.optimize(
            images=images,
            input_ids=input_ids,
            target_images=target_images,
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            scheduler=scheduler,
            weight_dtype=wd,
            block_masker=bm,
        )

        pbar.set_postfix(
            total=f"{outputs['stats']['total']:.4f}",
            anchor=f"{outputs['stats']['anchor']:.4f}",
            redir=f"{outputs['stats']['redirect']:.4f}",
            linf=f"{outputs['stats']['linf']:.3f}",
        )

        records.extend(save_batch(
            output_dir=out_dir,
            input_root=Path(args.input_dir),
            batch_paths=paths,
            clean_images=images.detach().cpu(),
            target_images=target_images.detach().cpu(),
            outputs=outputs,
            save_clean=args.save_clean_copy,
            save_raw=args.save_raw_delta,
        ))

    # ── Manifest ──
    manifest = {
        "input_dir": args.input_dir,
        "target_image_path": args.target_image_path,
        "target_dir": args.target_dir,
        "output_dir": str(out_dir),
        "num_images": len(dataset),
        "prompt": args.instance_prompt,
        "settings": vars(args),
        "files": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} protected images to {out_dir / 'protected'}")


if __name__ == "__main__":
    main(parse_args())
