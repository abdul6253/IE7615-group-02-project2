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

## Milestone 2 implementation

| Deliverable | Where |
|---|---|
| Stable Diffusion LoRA fine-tuning notebook | `notebooks/04_sd_lora_finetuning.ipynb` |
| LoRA training loop and prompt-conditioning evaluation | `scripts/train_sd_lora.py`, `scripts/evaluate_sd_conditioning.py` |
| Text-to-image integration architecture | `docs/m2_integration_architecture.md` |

## Milestone 1 results

| Result | Value |
|---|---|
| Image–caption pairs kept (from 2,031 checked) | 2,000 (1,600 train / 200 val / 200 test) |
| CLIP caption → image matching, top 1 / top 5 (500 pairs) | 78.4% / 95.4% (random guess: 0.2%) |
| Mean CLIP score, generated vs. real photos (15 test captions) | 27.0 vs. 24.4 |
| Generation time per image (Colab T4) | 4.9 s |

## Repository structure

```
.
├── notebooks/
│   ├── 01_dataset_preparation.ipynb     # Flickr30k → 2,000 clean 512×512 pairs
│   ├── 02_clip_embeddings.ipynb         # environment check, CLIP tokenization + embeddings
│   ├── 03_baseline_sd_generation.ipynb  # pretrained SD 1.5 baseline generation and evaluation
│   └── 04_sd_lora_finetuning.ipynb     # U-Net LoRA fine-tuning on Flickr30k captions/images
├── scripts/
│   └── train_sd_lora.py                # frozen CLIP/VAE + U-Net attention LoRA training loop
├── data/flickr30k_subset/
│   ├── metadata.csv                     # filename, caption, split, alternates, original size
│   ├── dataset_config.json              # every setting + cleaning counts
│   └── sample_pairs.png
├── embeddings/                          # CLIP retrieval results and figures
├── baseline_gallery/                    # generated images, gallery pages, results.csv, summary.json
├── docs/                                # reports, memos, and Milestone 2 integration architecture
└── requirements.txt                     # package versions used for the reported results
```

The 2,000 dataset images are **not** in the repo (Flickr30k images are licensed for non-commercial research only and stay under their owners' copyright). The notebooks rebuild them exactly from Hugging Face.

## Reproduce the environment

**Option A: Google Colab (recommended).** Clone/open the repository and run notebooks from its root. Notebooks 02, 03, and 04 need a GPU: *Runtime → Change runtime type → T4 GPU*. The first cell of each notebook installs what it needs (notebooks 02 and 03 also remove Colab's old `torchao`, which breaks new versions of Diffusers).

**Option B: local machine** (NVIDIA GPU with at least 8 GB memory for notebooks 02–03):

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook          # run notebooks from the repository root
```

Versions used for the reported results: Python 3.13, PyTorch 2.11, Transformers 5.20, Diffusers 0.41, on an NVIDIA Tesla T4.

## Run the baseline and fine-tune

| Step | Notebook | GPU | Time | Produces |
|---|---|---|---|---|
| 1 | `01_dataset_preparation` | no | ~10 min | `data/flickr30k_subset/` + `flickr30k_subset.zip` |
| 2 | `02_clip_embeddings` | T4 | ~10 min | `embeddings/`, `requirements.txt` |
| 3 | `03_baseline_sd_generation` | T4 | ~15 min | `baseline_gallery/` |
| 4 | `04_sd_lora_finetuning` | T4 | varies with GPU | `models/sd15-flickr30k-lora/` (adapter, config, loss history) |

Notebooks 02 and 03 do **not** need an upload: if `flickr30k_subset.zip` is in the Colab file panel they unzip it; if not, they rebuild the same 2,000 pairs from Hugging Face (about 5 minutes). Their data cell prints a **check code**; `477c680542` means the dataset is identical to the one used in our report. Notebook 04 uses the prepared `data/flickr30k_subset/` files, so run notebook 01 first (or make that directory available in the runtime).

**Fixed settings for the baseline:** seed 42 · 2,000 pairs (1,600 / 200 / 200) · 512×512 · CLIP `openai/clip-vit-large-patch14` · Stable Diffusion `stable-diffusion-v1-5/stable-diffusion-v1-5` · PNDM scheduler · 30 steps · guidance scale 7.5 · 15 prompts from the test split, seed 42 + prompt index.

Notebook 03 explicitly encodes each prompt into the 77-token hidden-state sequence expected by Stable Diffusion's U-Net and supplies it as `prompt_embeds`; the empty-prompt sequence is supplied as `negative_prompt_embeds` for classifier-free guidance. The pooled CLIP vectors saved by notebook 02 are for retrieval and scoring, not diffusion conditioning.

Notebook 04 fine-tunes rank-8 LoRA adapters on the Stable Diffusion U-Net attention projections, using only the Flickr30k training split. The pretrained CLIP text encoder, VAE, and base U-Net weights stay frozen. It saves the adapter and run metadata under `models/` (ignored by Git), generates a validation-prompt preview, and compares fixed held-out captions under pretrained, LoRA-conditioned, and empty-prompt control generation. CLIP matching/mismatched-caption scores and a review grid help check whether prompt conditioning is active. Run notebook 03 first to preserve the unchanged pretrained baseline. See [the Milestone 2 integration architecture](docs/m2_integration_architecture.md) for the text-to-image flow, training objective, and component boundaries.

## Data and model sources

- **Flickr30k:** Young, Lai, Hodosh & Hockenmaier, "From image descriptions to visual denotations," TACL 2014. Loaded from [`lmms-lab-encoder/flickr30k`](https://huggingface.co/datasets/lmms-lab-encoder/flickr30k). Non-commercial research and educational use only.
- **CLIP ViT-L/14:** [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14).
- **Stable Diffusion 1.5:** [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), CreativeML OpenRAIL-M license.

## Team (Group 02)

| Member | Role | Contribution and responsibility |
|---|---|---|
| Abdellah Faleh | Technical lead | Built the three notebooks (data, CLIP, first images), ran them in Colab, and wrote the proposal, dataset memo, gallery memo and report; connects the text encoder in Milestone 2 |
| Jin-Woo Hong | Experiments | Reran the notebooks to check the baseline and added the recommendation to the evaluation review; leads the guidance scale, scheduler and step-count tests in Milestones 2 and 3 |
| Samuel Tong | Evaluation | Reran notebooks 02 and 03 in Colab to check they reproduce, and wrote the Milestone 1 evaluation review (`docs/m1_baseline_evaluation_review.md`); leads FID, Inception Score and CLIP score evaluation in Milestone 3 |
| Tristan Lyons | Documentation | Keeps the GitHub repository organized, submits on Canvas, and is the main contact with the teaching staff |
