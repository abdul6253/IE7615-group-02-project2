# Milestone 1 — Canvas Submission Checklist

- [x] Select image-caption dataset. (`Flickr30k`)
- [x] Curate a working subset of 1-5k image-caption pairs applying consistent rezizing, normalization, and caption-cleaning pipeline. (`notebooks/01_dataset_preparation.ipynb`)
- [x] Setup and configure enviornment.
- [x] Build a notebook that loads CLIP, tokenizes captions, computes image and text embeddings for a sample of pairs, and verifies the pipeline runs end-to-end. (`notebooks/02_clip_embeddings.ipynb`)
- [x] Generate images for 10-20 representative prompts drawn from curated dataset using stable diffusion. (`notebooks/03_baseline_sd_generation.ipynb`)
- [x] Write intial project proposal. (`docs/Project2_Proposal.pdf`)
- [x] Initialize and setup GitHub repository for project 2 including all files described above and a README documenting how to setup/reproduce the enviornment and rerun the baseline. (`README.md`)
- [x] Repo set to public per direct confirmation from Priyanka/TA
- [X] All team members have an individual commit in `main`
- [X] Final commit hash recorded for submission
- [X] Submitted via Canvas: repo link + commit hash

---

# Milestone 2 — Canvas Submission Checklist

- [x] Integrate CLIP text embeddings into a diffusion pipeline as the conditioning signal. Document the architecture and text → embedding → conditioning → image workflow.
- [x] Build or configure a text-conditional diffusion model (unconditional DDPM extended with text conditioning or a pretrained pipeline such as Stable Diffusion). Document any modifications to the conditioning pathway.
- [x] Generate baseline conditional images for a fixed set of representative prompts from the curated dataset. Save generated images, prompts, and random seeds.
- [x] Implement a classifier-free guidance scale experiment using at least three different guidance settings on the same baseline prompts. Compare output quality, prompt alignment, and diversity.
- [x] Save representative outputs from the guidance-scale experiment for the final gallery.
- [x] Implement a noise schedule or sampling experiment by varying at least one of the following: linear vs. cosine noise schedule, number of sampling steps, or DDPM vs. DDIM sampler.
- [x] Compare image quality and computational trade-offs across noise schedule/sampling configurations. Save representative outputs and parameter settings.
- [x] Capture training or inference diagnostics, including loss curves (or sampler diagnostics for pretrained inference-only pipelines), training/inference time, GPU memory usage, and any instability encountered.
- [x] Create an integration write-up (PDF, 1–2 pages) containing an architecture diagram, explanation of how text embeddings flow into the diffusion model, and any modifications made to a pretrained pipeline. (`docs/Project2_M2_Integration_Writeup.pdf`)
- [x] Create a generated samples gallery (PDF or notebook export) containing baseline conditional outputs, guidance-scale comparisons, and noise-schedule/sampling comparisons, with prompts and parameter settings labeled. (`docs/Project2_M2_Samples_Gallery.pdf`)
- [x] Create a training logs and summary document (PDF or Markdown, 1–2 pages) containing loss curves, runtime metrics, parameter-tuning observations, and lessons learned. (`docs/Project2_M2_Training_Logs_Summary.pdf`)
- [x] Create an integration/training notebook documenting the model setup, text-conditioning integration, and training or configuration process. (`notebooks/04_integration_training.ipynb`)
- [x] Create a conditional-generation notebook implementing baseline generation and guidance-scale/noise-schedule parameter sweeps. (`notebooks/05_conditional_generation_sweeps.ipynb`)
- [x] Update the GitHub README with environment setup instructions, notebook descriptions, model architecture, and steps to reproduce the experiments. (`README.md`)
- [x] Ensure all generated samples, prompts, random seeds, experiment configurations, and relevant logs are saved in the repository.
- [x] Verify that all notebooks run successfully and that the experiments can be reproduced using the documented instructions.
- [x] Push all final notebooks, documentation, generated samples, and supporting artifacts to the team's GitHub repository.
- [x] Record the final Git commit hash or release tag for submission.
- [x] Submit via Canvas: updated GitHub repository link + commit hash/release tag, integration write-up, generated samples gallery, training logs and summary, and notebook artifacts.