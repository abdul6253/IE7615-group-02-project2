# Milestone 2: Stable Diffusion integration and fine-tuning

## Architecture

```text
Caption
  -> Stable Diffusion CLIP tokenizer (77 token IDs)
  -> pretrained CLIP text encoder (77 x 768 hidden states)
  -> U-Net cross-attention at each denoising step
  -> denoised 4 x 64 x 64 latent
  -> pretrained VAE decoder
  -> 512 x 512 RGB image
```

The generation notebook passes the token-level CLIP hidden states to the pipeline as
`prompt_embeds`. An empty-caption embedding is passed as `negative_prompt_embeds` for
classifier-free guidance. The pooled 768-dimensional vectors in `embeddings/` are for
retrieval/scoring and are not the diffusion conditioning tensor.

## Fine-tuning path

`scripts/train_sd_lora.py`, launched by `notebooks/04_sd_lora_finetuning.ipynb`, adapts
the pretrained Stable Diffusion 1.5 U-Net using the paired Flickr30k training split:

1. Load a training image and its caption; tokenize the caption with the model's CLIP
   tokenizer.
2. Encode the image into a latent with the frozen VAE. Encode the caption into token
   hidden states with the frozen CLIP text encoder.
3. Add a randomly sampled timestep of Gaussian noise to the image latent.
4. Condition the U-Net on the noisy latent, timestep, and caption hidden states.
   Cross-attention uses the caption states as keys/values while predicting the noise
   (or velocity, according to the pretrained scheduler configuration).
5. Optimize only rank-8 LoRA matrices attached to U-Net attention projections
   (`to_q`, `to_k`, `to_v`, `to_out.0`). Keep the pretrained U-Net weights, text
   encoder, and VAE frozen.
6. Save a portable U-Net LoRA adapter, training configuration, and loss history.

At inference, load the saved adapter into the same pretrained Stable Diffusion pipeline.
The existing baseline notebook remains unchanged, so baseline and fine-tuned generations
can be compared using the same model family, prompts, seeds, and evaluation.

### Verifying that text conditioning is active

The fine-tuning notebook's final evaluation uses a fixed, seeded sample of validation
captions. It generates images with the pretrained model and correct captions, with the
LoRA adapter and correct captions, and with the LoRA adapter but empty prompts. Each
caption uses the same initial noise seed across conditions. CLIP reports matching-caption
similarity, cyclically mismatched-caption similarity, and top-1 retrieval across the
caption set. The saved comparison grid supports human review. Higher matching scores for
caption-conditioned images than for the empty-prompt control are evidence that generation
uses the text condition; CLIP and a small sample do not guarantee exact counts or attributes.

## Reproduction and data boundaries

Run `notebooks/01_dataset_preparation.ipynb` first so that
`data/flickr30k_subset/metadata.csv` and its `images/` directory exist. The training code
uses only rows marked `train`; validation and test images are not used for optimization.
Run `notebooks/03_baseline_sd_generation.ipynb` before fine-tuning to retain the
pre-adaptation comparison. Then run `notebooks/04_sd_lora_finetuning.ipynb` on a CUDA
GPU (Colab T4 recommended).

The default run uses 800 optimizer updates, batch size 1, 512-pixel images, and rank-8
LoRA adapters. It is an experiment configuration, not a claim that the resulting model
will outperform the pretrained baseline. Record the generated adapter and loss outputs
outside version control; the underlying Flickr30k images remain excluded from the repo.
