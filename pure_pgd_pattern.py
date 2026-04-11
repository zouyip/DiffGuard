import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from iterative_pattern_pgd import (
    CustomDiffusionDataset,
    FrequencyCarrier,
    TwoBlockMasker,
    build_semantic_mask,
    import_model_class_from_model_name_or_path,
    variance_alignment_loss,
)


def project_fixed_pattern(delta, semantic_mask, carrier_bank, eps, mode="batch_shared"):
    """
    把任意 delta 投影到“有语义 + 固定 pattern 模式”的集合 S：

    S = { delta | delta_i = M_i * P, P 属于固定频率基底张成的子空间 }

    这里唯一优化变量始终是 delta，本函数只是 PGD 的投影算子 Proj_S。
    """
    bsz, channels, height, width = delta.shape
    carriers = carrier_bank.get_carriers(bsz).to(delta.device)
    mask = semantic_mask.expand(-1, channels, -1, -1)
    masked_delta = delta * mask

    if mode == "batch_shared":
        base = masked_delta.mean(dim=1, keepdim=True)
        coeff_num = (base * carriers).flatten(2).mean(dim=2, keepdim=True)
        coeff_den = (carriers * carriers).flatten(2).mean(dim=2, keepdim=True) + 1e-8
        coeff = (coeff_num / coeff_den).view(bsz, -1, 1, 1).mean(dim=0, keepdim=True)
        pattern = (carriers[:1] * coeff).sum(dim=1, keepdim=True)
        pattern = pattern.repeat(bsz, channels, 1, 1)
    elif mode == "samplewise":
        base = masked_delta.mean(dim=1, keepdim=True)
        coeff_num = (base * carriers).flatten(2).mean(dim=2, keepdim=True)
        coeff_den = (carriers * carriers).flatten(2).mean(dim=2, keepdim=True) + 1e-8
        coeff = (coeff_num / coeff_den).view(bsz, -1, 1, 1)
        pattern = (carriers * coeff).sum(dim=1, keepdim=True).repeat(1, channels, 1, 1)
    else:
        raise ValueError(f"Unsupported projection mode: {mode}")

    projected = mask * pattern
    return projected.clamp(-eps, eps), pattern.clamp(-eps, eps)


class PurePGDSemanticPattern:
    """
    严格的 PGD：
    1. 优化变量只有 delta
    2. 每步按 sign gradient 更新 delta
    3. 每步投影到结构集合 S

    更新公式：
    delta_{t+1} = Proj_S(delta_t + alpha * sign(grad_delta L))
    """

    def __init__(
        self,
        resolution=512,
        eps=0.05,
        alpha=1.0 / 255.0,
        pgd_steps=10,
        freqs=(20, 40, 80),
        lambda_diff=1000.0,
        lambda_var=1.0,
        semantic_topk=0.35,
        projection_mode="batch_shared",
    ):
        self.resolution = resolution
        self.eps = eps
        self.alpha = alpha
        self.pgd_steps = pgd_steps
        self.lambda_diff = lambda_diff
        self.lambda_var = lambda_var
        self.semantic_topk = semantic_topk
        self.projection_mode = projection_mode
        self.carrier_bank = FrequencyCarrier(size=resolution, freqs=freqs)

    def attack_batch(
        self,
        images,
        input_ids,
        vae,
        unet,
        text_encoder,
        noise_scheduler,
        weight_dtype,
        block_masker=None,
    ):
        device = images.device
        bsz = images.shape[0]

        semantic_mask = build_semantic_mask(
            images=images,
            input_ids=input_ids,
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            topk_ratio=self.semantic_topk,
        )

        delta = torch.zeros_like(images, device=device)
        delta, pattern = project_fixed_pattern(
            delta=delta,
            semantic_mask=semantic_mask,
            carrier_bank=self.carrier_bank,
            eps=self.eps,
            mode=self.projection_mode,
        )

        with torch.no_grad():
            latent_clean_dist = vae.encode(images.to(dtype=weight_dtype) * 2 - 1).latent_dist

        last_diff = None
        last_var = None

        for _ in range(self.pgd_steps):
            delta_var = delta.detach().clone().requires_grad_(True)
            delta_for_attack = block_masker(delta_var) if block_masker is not None else delta_var
            x_adv = (images + delta_for_attack.to(images.dtype)).clamp(0.0, 1.0)

            latent_adv_dist = vae.encode(x_adv.to(dtype=weight_dtype) * 2 - 1).latent_dist
            poison_latents = latent_adv_dist.sample() * vae.config.scaling_factor

            pattern_latent_dist = vae.encode((delta_var * 2.0 - 1.0).to(dtype=weight_dtype)).latent_dist
            pattern_latents = pattern_latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(poison_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(pattern_latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(
                poison_latents, noise, timesteps
            )

            diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            var_loss = variance_alignment_loss(latent_clean_dist, latent_adv_dist, pattern_latent_dist)
            loss = self.lambda_diff * diffusion_loss + self.lambda_var * var_loss

            grad = torch.autograd.grad(loss, delta_var, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                delta = delta + self.alpha * grad.sign()
                delta = delta.clamp(-self.eps, self.eps)
                delta, pattern = project_fixed_pattern(
                    delta=delta,
                    semantic_mask=semantic_mask,
                    carrier_bank=self.carrier_bank,
                    eps=self.eps,
                    mode=self.projection_mode,
                )

            last_diff = diffusion_loss.detach()
            last_var = var_loss.detach()

        with torch.no_grad():
            final_delta = block_masker(delta) if block_masker is not None else delta
            x_adv = (images + final_delta.to(images.dtype)).clamp(0.0, 1.0)

        stats = {
            "diffusion_loss": float(last_diff.item()),
            "var_loss": float(last_var.item()),
            "mask_mean": float(semantic_mask.mean().item()),
            "delta_linf": float(final_delta.abs().amax().item()),
        }
        return {
            "x_adv": x_adv,
            "delta": final_delta,
            "pattern": pattern,
            "semantic_mask": semantic_mask,
            "stats": stats,
        }


def save_outputs(output_dir, step, outputs, eps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_adv = outputs["x_adv"].detach().float().cpu()
    delta = outputs["delta"].detach().float().cpu()
    semantic_mask = outputs["semantic_mask"].detach().float().cpu()
    pattern = outputs["pattern"].detach().float().cpu()

    for i in range(x_adv.shape[0]):
        adv = (x_adv[i].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
        delta_vis = ((delta[i].permute(1, 2, 0).numpy() / (2 * eps)) + 0.5).clip(0, 1)
        mask_vis = semantic_mask[i, 0].numpy().clip(0, 1)
        Image.fromarray(adv).save(output_dir / f"adv_{step}_{i}.png")
        Image.fromarray((delta_vis * 255).astype("uint8")).save(output_dir / f"delta_{step}_{i}.png")
        Image.fromarray((mask_vis * 255).astype("uint8")).save(output_dir / f"mask_{step}_{i}.png")

    pattern_vis = ((pattern[0].permute(1, 2, 0).numpy() / (2 * eps)) + 0.5).clip(0, 1)
    Image.fromarray((pattern_vis * 255).astype("uint8")).save(output_dir / f"pattern_{step}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/usr/zou/CAAT/sd1.5/")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default="/usr/zou/dataset/cele_label/celebahq_512x512/")
    parser.add_argument("--instance_prompt", type=str, default="a photo of a sks person")
    parser.add_argument("--output_dir", type=str, default="outputs/pure_pgd_pattern")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--lambda_diff", type=float, default=1000.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--semantic_topk", type=float, default=0.35)
    parser.add_argument("--projection_mode", type=str, default="batch_shared", choices=["batch_shared", "samplewise"])
    parser.add_argument("--num_batches", type=int, default=3)
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    dataset = CustomDiffusionDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    attacker = PurePGDSemanticPattern(
        resolution=args.resolution,
        eps=args.eps,
        alpha=args.alpha,
        pgd_steps=args.pgd_steps,
        lambda_diff=args.lambda_diff,
        lambda_var=args.lambda_var,
        semantic_topk=args.semantic_topk,
        projection_mode=args.projection_mode,
    )
    block_masker = TwoBlockMasker(block_size=128).to(device)
    block_masker.train()

    for step, batch in enumerate(tqdm(dataloader, desc="pure PGD semantic pattern")):
        images = batch["instance_images"].to(device, dtype=weight_dtype)
        input_ids = batch["instance_prompt_ids"].squeeze(1).to(device)
        outputs = attacker.attack_batch(
            images=images,
            input_ids=input_ids,
            vae=vae,
            unet=unet,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            block_masker=block_masker,
        )
        save_outputs(output_dir, step, outputs, args.eps)
        print(
            f"step={step} d_loss={outputs['stats']['diffusion_loss']:.4f} "
            f"v_loss={outputs['stats']['var_loss']:.4f} "
            f"mask={outputs['stats']['mask_mean']:.4f} "
            f"linf={outputs['stats']['delta_linf']:.4f}"
        )
        if step + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    main(parse_args())
