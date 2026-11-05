import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers


class CaptionImageDataset(Dataset):
    def __init__(self, data_dir, resolution):
        self.data_dir = Path(data_dir)
        metadata_path = self.data_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing {metadata_path}. Run notebooks/01_dataset_preparation.ipynb first."
            )

        metadata = pd.read_csv(metadata_path)
        required_columns = {"filename", "caption", "split"}
        missing_columns = required_columns - set(metadata.columns)
        if missing_columns:
            raise ValueError(f"Dataset metadata is missing columns: {sorted(missing_columns)}")

        self.rows = metadata.loc[metadata["split"] == "train", ["filename", "caption"]].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError("The dataset has no rows with split='train'.")

        self.image_dir = self.data_dir / "images"
        missing_images = [
            filename for filename in self.rows["filename"]
            if not (self.image_dir / filename).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} training image(s) are missing from {self.image_dir}; "
                f"first missing file: {missing_images[0]}"
            )
        self.resolution = resolution

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.image_dir / row["filename"]) as image:
            image = image.convert("RGB").resize(
                (self.resolution, self.resolution), Image.Resampling.BICUBIC
            )
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        return pixel_values, str(row["caption"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion 1.5 U-Net attention with LoRA."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/flickr30k_subset"))
    parser.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", type=Path, default=Path("models/sd15-flickr30k-lora"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA fine-tuning requires a CUDA GPU; use a Colab T4 runtime.")
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8.")
    if min(args.train_steps, args.batch_size, args.gradient_accumulation_steps, args.lora_rank) < 1:
        raise ValueError("Training steps, batch size, accumulation steps, and LoRA rank must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    weight_dtype = torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {weight_dtype}")

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    dataset = CaptionImageDataset(args.data_dir, args.resolution)
    loader_generator = torch.Generator().manual_seed(args.seed)

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        captions = [example[1] for example in examples]
        tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return pixel_values, tokens.input_ids

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=loader_generator,
    )

    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=weight_dtype
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    if text_encoder.config.hidden_size != unet.config.cross_attention_dim:
        raise ValueError(
            "CLIP text hidden size does not match the U-Net cross-attention dimension."
        )

    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
    unet.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    cast_training_params(unet, dtype=torch.float32)
    trainable_params = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were added to the U-Net.")
    trainable_count = sum(parameter.numel() for parameter in trainable_params)
    total_count = sum(parameter.numel() for parameter in unet.parameters())
    print(f"Training {trainable_count:,} / {total_count:,} U-Net parameters with LoRA")
    print(f"Training examples: {len(dataset)} | optimizer updates: {args.train_steps}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    use_attention_mask = getattr(text_encoder.config, "use_attention_mask", False)
    global_step = 0
    loss_sum = 0.0
    loss_count = 0
    loss_history = []

    unet.train()
    while global_step < args.train_steps:
        for batch_index, (pixel_values, input_ids) in enumerate(dataloader):
            if global_step >= args.train_steps:
                break

            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(dataloader) - group_start)
            pixel_values = pixel_values.to(device, dtype=weight_dtype, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    attention_mask = (
                        input_ids.ne(tokenizer.pad_token_id)
                        if use_attention_mask else None
                    )
                    encoder_hidden_states = text_encoder(
                        input_ids, attention_mask=attention_mask
                    ).last_hidden_state

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=weight_dtype):
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unsupported scheduler prediction type: {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at optimizer step {global_step}.")
            scaler.scale(loss / group_size).backward()
            loss_sum += loss.detach().item()
            loss_count += 1

            is_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(dataloader)
            )
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 25 == 0 or global_step == args.train_steps:
                    mean_loss = loss_sum / max(loss_count, 1)
                    loss_history.append({"step": global_step, "mean_loss": mean_loss})
                    print(f"step {global_step:04d}/{args.train_steps} | mean loss {mean_loss:.5f}")
                    loss_sum = 0.0
                    loss_count = 0

                if global_step >= args.train_steps:
                    break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(args.output_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )

    config = {
        "base_model": args.model_id,
        "training_split": "train",
        "training_examples": len(dataset),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "resolution": args.resolution,
        "seed": args.seed,
        "trained_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "frozen_components": ["text_encoder", "vae", "base_unet_weights"],
        "precision": str(weight_dtype),
    }
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    pd.DataFrame(loss_history).to_csv(args.output_dir / "loss_history.csv", index=False)
    print(f"Saved LoRA adapter and training metadata to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
