# Project 2 — Text-to-Image Generation Using Transformer and Diffusion Models

IE 7615 Deep Learning for AI · Generative Project · Group 02

A text-to-image pipeline that turns a caption into a 512×512 image: a Transformer text encoder (CLIP ViT-L/14) conditions a latent diffusion model (Stable Diffusion 1.5). Built on a curated 2,000-pair subset of Flickr30k.

## Milestone 1 deliverables

| Deliverable | Where |
|---|---|
| Milestone 1 report (front page + 6 pages) | `docs/Project2_M1_Report.pdf` |
| Project proposal | `docs/Project2_Proposal.pdf` |
| Dataset memo | `docs/Project2_Dataset_Memo.pdf` |
| Baseline gallery and memo | `docs/Project2_Baseline_Gallery_Memo.pdf`, `baseline_gallery/` |
| Baseline evaluation review | `docs/m1_baseline_evaluation_review.md` |
| Dataset-preparation notebook | `notebooks/01_dataset_preparation.ipynb` |
| CLIP / text-embedding notebook | `notebooks/02_clip_embeddings.ipynb` |
| Baseline Stable Diffusion notebook | `notebooks/03_baseline_sd_generation.ipynb` |

## Milestone 1 results

| Result | Value |
|---|---|
| Image–caption pairs kept (from 2,031 checked) | 2,000 (1,600 train / 200 val / 200 test) |
| CLIP caption → image matching, top 1 / top 5 (500 pairs) | 78.4% / 95.4% (random guess: 0.2%) |
| Mean CLIP score, generated vs. real photos (15 test captions) | 27.0 vs. 24.4 |
| Generation time per image (Colab T4) | 4.9 s |

## Milestone 2 deliverables (Model integration and baseline results)

| Deliverable | Where |
|---|---|
| Milestone 2 report (front page + 6 pages) | `docs/Project2_M2_Report.pdf` |
| Integration write-up (architecture, text → embedding → conditioning → image) | `docs/Project2_M2_Integration_Writeup.pdf` |
| Generated samples gallery (baseline, guidance, sampler, noise schedule) | `docs/Project2_M2_Samples_Gallery.pdf` |
| Training logs and summary | `docs/Project2_M2_Training_Logs_Summary.pdf` |
| Integration + training notebook | `notebooks/04_integration_training.ipynb` |
| Conditional generation + parameter sweeps notebook | `notebooks/05_conditional_generation_sweeps.ipynb` |
| Stable Diffusion LoRA fine-tuning notebook | `notebooks/04_sd_lora_finetuning.ipynb` |
| LoRA training loop and prompt-conditioning evaluation | `scripts/train_sd_lora.py`, `scripts/evaluate_sd_conditioning.py` |
| LoRA integration notes (architecture, training objective) | `docs/m2_integration_architecture.md` |
| Integration outputs and logs (loss curves, diagnostics, checks) | `m2_integration/` |
| Generation outputs (202 images, `results.csv` with prompt, seed and settings per image) | `m2_generation/` |

## Milestone 2 results

| Result | Value |
|---|---|
| Our own denoising loop vs. the official pipeline (same prompt and seed) | identical (mean pixel difference 0.0) |
| Same seed, 6 different captions: images matching their own caption best | 6 of 6 (CLIP 27.4 right vs. 7.8 wrong) |
| BERT → CLIP adapter (17M parameters, 15 epochs, 5.1 min, 6.35 GB) | validation loss 0.52 (average-only reference 1.00) |
| Mean CLIP score: CLIP / raw BERT / BERT + adapter | 28.8 / 9.9 / 21.4 |
| Baseline (15 test prompts × 2 seeds) | mean CLIP 27.3, 5.0 s per image |
| Guidance scale 1 / 3 / 5 / 7.5 / 10 / 15 (mean CLIP) | 20.8 / 24.8 / 25.8 / 26.0 / 25.6 / 25.9 |
| Fastest sampler with the same look as the baseline | DPM-Solver++, 20 steps (3.8 s vs. 5.7 s) |
| LoRA fine-tuning of the U-Net (rank 8, 1.6M of 861M parameters, 800 steps) | loss 0.10–0.24, no clear drop yet; held-out comparison in Milestone 3 |

## Repository structure

```
.
├── notebooks/
│   ├── 01_dataset_preparation.ipynb     # Flickr30k → 2,000 clean 512×512 pairs
│   ├── 02_clip_embeddings.ipynb         # environment check, CLIP tokenization + embeddings
│   ├── 03_baseline_sd_generation.ipynb  # SD 1.5 baseline, gallery, CLIP scores, diversity test
│   ├── 04_integration_training.ipynb    # M2: own text->embedding->U-Net->image loop, checks, BERT adapter training
│   ├── 04_sd_lora_finetuning.ipynb      # M2: U-Net LoRA fine-tuning on Flickr30k train images and captions
│   └── 05_conditional_generation_sweeps.ipynb  # M2: baseline prompts, guidance / sampler / noise-schedule sweeps
├── scripts/
│   ├── train_sd_lora.py                 # frozen CLIP/VAE + U-Net attention LoRA training loop
│   └── evaluate_sd_conditioning.py      # pretrained vs. LoRA vs. empty-prompt comparison with CLIP
├── data/flickr30k_subset/
│   ├── metadata.csv                     # filename, caption, split, alternates, original size
│   ├── dataset_config.json              # every setting + cleaning counts
│   └── sample_pairs.png
├── embeddings/                          # CLIP retrieval results and figures
├── baseline_gallery/                    # generated images, gallery pages, results.csv, summary.json
├── m2_integration/                      # M2 checks, sampler diagnostics, adapter loss logs, encoder comparison
├── m2_generation/                       # M2 202 images (JPEG copies), results.csv, summary.json, grids
├── docs/                                # M1 and M2 reports, memos, write-up, gallery, training summary, LoRA notes
└── requirements.txt                     # package versions used for the reported results
```

The 2,000 dataset images are **not** in the repo (Flickr30k images are licensed for non-commercial research only and stay under their owners' copyright). The notebooks rebuild them exactly from Hugging Face.

## Reproduce the environment

**Option A: Google Colab (recommended).** Open each notebook in Colab. Notebooks 02–05 need a GPU: *Runtime → Change runtime type → T4 GPU*. The first cell of each notebook installs what it needs (it also removes Colab's old `torchao`, which breaks new versions of Diffusers).

**Option B: local machine** (NVIDIA GPU with at least 8 GB memory for notebooks 02–03):

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook          # run notebooks from the repository root
```

Versions used for the reported results: Python 3.13, PyTorch 2.11, Transformers 5.20, Diffusers 0.41 (Milestone 1) and Transformers 5.22, Diffusers 0.42.1 (Milestone 2), on an NVIDIA Tesla T4. LoRA also needs `peft`.

## Rerun the baseline

| Step | Notebook | GPU | Time | Produces |
|---|---|---|---|---|
| 1 | `01_dataset_preparation` | no | ~10 min | `data/flickr30k_subset/` + `flickr30k_subset.zip` |
| 2 | `02_clip_embeddings` | T4 | ~10 min | `embeddings/`, `requirements.txt` |
| 3 | `03_baseline_sd_generation` | T4 | ~15 min | `baseline_gallery/` |
| 4 | `04_integration_training` | T4 | ~11 min | `m2_integration/` + `m2_integration.zip` |
| 5 | `05_conditional_generation_sweeps` | T4 | ~21 min | `m2_generation/` + `m2_generation.zip` |
| 6 | `04_sd_lora_finetuning` | T4 | varies with GPU | `models/sd15-flickr30k-lora/` (adapter, config, loss history; ignored by Git) |

Notebooks 02 and 03 do **not** need an upload: if `flickr30k_subset.zip` is in the Colab file panel they unzip it; if not, they rebuild the same 2,000 pairs from Hugging Face (about 5 minutes). Their data cell prints a **check code**; `477c680542` means the dataset is identical to the one used in our report. Notebooks 04 and 05 only need the captions, so if there is no zip they download `metadata.csv` from this repository (no rebuild). Milestone 2 results were made with Diffusers 0.42.1 and Transformers 5.22. The 202 images in `m2_generation/` are stored as JPEG copies (quality 90) to keep the repository small; the notebook saves PNG.

Notebook `04_sd_lora_finetuning` trains on the photos themselves, so it needs `data/flickr30k_subset/images/` (run notebook 01 first, or unzip `flickr30k_subset.zip`) and the `peft` package. Notebook 03 encodes each prompt into the 77-token CLIP hidden-state sequence and passes it as `prompt_embeds`, with the empty-prompt sequence as `negative_prompt_embeds`; the pooled vectors from notebook 02 are for retrieval and scoring only. See `docs/m2_integration_architecture.md` for the LoRA training objective and component boundaries.

**Milestone 2 sweeps:** guidance 1, 3, 5, 7.5, 10, 15 · samplers DDPM, DDIM, PNDM, Euler, DPM-Solver++ × 10, 20, 30, 50 steps · noise schedules scaled linear, linear, cosine (DDIM) and default vs. Karras sigmas (DPM-Solver++) · seed 42 + prompt id. **LoRA:** rank 8 on `to_q`, `to_k`, `to_v`, `to_out.0` · 800 steps · batch 1 · learning rate 1e-4 · train split only.

**Fixed settings for the baseline:** seed 42 · 2,000 pairs (1,600 / 200 / 200) · 512×512 · CLIP `openai/clip-vit-large-patch14` · Stable Diffusion `stable-diffusion-v1-5/stable-diffusion-v1-5` · PNDM scheduler · 30 steps · guidance scale 7.5 · 15 prompts from the test split, seed 42 + prompt index.

## Data and model sources

- **Flickr30k:** Young, Lai, Hodosh & Hockenmaier, "From image descriptions to visual denotations," TACL 2014. Loaded from [`lmms-lab-encoder/flickr30k`](https://huggingface.co/datasets/lmms-lab-encoder/flickr30k). Non-commercial research and educational use only.
- **CLIP ViT-L/14:** [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14).
- **Stable Diffusion 1.5:** [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), CreativeML OpenRAIL-M license.

## Team (Group 02)

| Member | Role | Contribution and responsibility |
|---|---|---|
| Abdellah Faleh | Technical lead | Built the three notebooks (data, CLIP, first images), ran them in Colab, and wrote the proposal, dataset memo, gallery memo and report; in Milestone 2 built and ran notebooks 04–05 (integration, adapter training, sweeps) and wrote the M2 documents |
| Jin-Woo Hong | Experiments | Reran the notebooks to check the baseline and added the recommendation to the evaluation review; in Milestone 2 built and ran the LoRA fine-tuning of the U-Net (notebook 04_sd_lora_finetuning, training and evaluation scripts) and updated notebook 03 to pass explicit CLIP embeddings; leads the parameter-sensitivity study in Milestone 3 |
| Samuel Tong | Evaluation | Reran notebooks 02 and 03 in Colab to check they reproduce, and wrote the Milestone 1 evaluation review (`docs/m1_baseline_evaluation_review.md`); leads FID, Inception Score and CLIP score evaluation in Milestone 3 |
| Tristan Lyons | Documentation | Keeps the GitHub repository organized, submits on Canvas, and is the main contact with the teaching staff |
