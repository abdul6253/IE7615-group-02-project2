import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_MODEL = "openai/clip-vit-large-patch14"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Stable Diffusion prompt conditioning with an empty-prompt control."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/flickr30k_subset"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/sd15-flickr30k-lora"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/sd15-flickr30k-lora/conditioning_eval")
    )
    parser.add_argument("--num-prompts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_images(pipeline, prompts, seeds, steps, output_dir, condition_name, guidance_scale):
    images = []
    for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipeline(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        image.save(output_dir / f"{condition_name}_{index:02d}.png")
        images.append(image)
        print(f"{condition_name:13s} {index + 1}/{len(prompts)} seed={seed}")
    return images


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Prompt-conditioning evaluation requires a CUDA GPU.")
    if args.num_prompts < 2 or args.steps < 1:
        raise ValueError("--num-prompts must be at least 2 and --steps must be positive.")
    if not args.adapter_dir.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {args.adapter_dir}")

    metadata_path = args.data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing validation metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {"filename", "caption", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Dataset metadata is missing columns: {sorted(missing)}")
    validation = metadata.loc[metadata["split"] == "val", ["filename", "caption"]]
    if len(validation) < 2:
        raise ValueError("At least two validation captions are required for retrieval evaluation.")
    prompts_df = validation.sample(
        n=min(args.num_prompts, len(validation)), random_state=args.seed
    ).reset_index(drop=True)
    prompts = prompts_df["caption"].astype(str).tolist()
    seeds = [args.seed + index for index in range(len(prompts))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_df.assign(seed=seeds).to_csv(args.output_dir / "prompts.csv", index=False)

    print(f"Evaluating {len(prompts)} fixed validation captions with matching seeds.")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generated = {}
    adapter_loaded = False
    try:
        generated["pretrained"] = generate_images(
            pipeline, prompts, seeds, args.steps, args.output_dir, "pretrained", 7.5
        )
        pipeline.load_lora_weights(str(args.adapter_dir))
        adapter_loaded = True
        generated["lora"] = generate_images(
            pipeline, prompts, seeds, args.steps, args.output_dir, "lora", 7.5
        )
        generated["empty_prompt"] = generate_images(
            pipeline, [""] * len(prompts), seeds, args.steps,
            args.output_dir, "empty_prompt", 1.0
        )
    finally:
        if adapter_loaded:
            pipeline.unload_lora_weights()
        pipeline.to("cpu")
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(
        CLIP_MODEL, torch_dtype=torch.float16
    ).to("cuda").eval()

    @torch.inference_mode()
    def encode_texts(texts):
        tokens = processor.tokenizer(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to("cuda")
        projected = clip.text_projection(clip.text_model(**tokens).pooler_output)
        return F.normalize(projected.float(), dim=-1)

    @torch.inference_mode()
    def encode_images(images):
        embeddings = []
        for start in range(0, len(images), 4):
            batch = processor(
                images=images[start:start + 4], return_tensors="pt"
            )["pixel_values"].to("cuda", dtype=clip.dtype)
            projected = clip.visual_projection(
                clip.vision_model(pixel_values=batch).pooler_output
            )
            embeddings.append(F.normalize(projected.float(), dim=-1))
        return torch.cat(embeddings)

    text_embeddings = encode_texts(prompts)
    rows = []
    similarities = {}
    for condition, images in generated.items():
        image_embeddings = encode_images(images)
        matrix = (100 * image_embeddings @ text_embeddings.T).cpu().numpy()
        similarities[condition] = matrix
        diagonal = np.diag(matrix)
        shuffled = matrix[np.arange(len(prompts)), np.roll(np.arange(len(prompts)), 1)]
        rows.append({
            "condition": condition,
            "mean_matching_clip": float(diagonal.mean()),
            "mean_mismatched_clip": float(shuffled.mean()),
            "matching_minus_mismatched": float((diagonal - shuffled).mean()),
            "caption_retrieval_top1_percent": float(
                100 * np.mean(matrix.argmax(axis=1) == np.arange(len(prompts)))
            ),
        })

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output_dir / "clip_conditioning_scores.csv", index=False)
    lora_row = scores.loc[scores["condition"] == "lora"].iloc[0]
    empty_row = scores.loc[scores["condition"] == "empty_prompt"].iloc[0]
    conditioning_pass = (
        lora_row["mean_matching_clip"] > empty_row["mean_matching_clip"]
        and lora_row["matching_minus_mismatched"] > 0
    )

    print("\nCLIP cosine similarity x 100 (higher is better):")
    print(scores.round(2).to_string(index=False))
    print(
        "\nLoRA conditioning check:",
        "PASS" if conditioning_pass else "REVIEW",
        "| compared with empty-prompt outputs and cyclically mismatched captions.",
    )
    print(
        "This is an automatic semantic check, not a guarantee of exact object counts, "
        "attributes, or image quality; review the saved grid."
    )

    fig, axes = plt.subplots(len(prompts), 4, figsize=(17, 3.8 * len(prompts)))
    if len(prompts) == 1:
        axes = np.expand_dims(axes, axis=0)
    for index, (caption, row_axes) in enumerate(zip(prompts, axes)):
        row_axes[0].text(
            0.02, 0.98, caption, ha="left", va="top", wrap=True, fontsize=11,
            transform=row_axes[0].transAxes,
        )
        row_axes[0].axis("off")
        for axis, condition, title in zip(
            row_axes[1:],
            ("pretrained", "lora", "empty_prompt"),
            ("Pretrained + caption", "LoRA + caption", "Empty prompt control"),
        ):
            axis.imshow(generated[condition][index])
            axis.set_title(
                f"{title}\nCLIP {similarities[condition][index, index]:.1f}", fontsize=10
            )
            axis.axis("off")
    fig.tight_layout()
    grid_path = args.output_dir / "conditioning_comparison.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved prompt/image grid and scores to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
