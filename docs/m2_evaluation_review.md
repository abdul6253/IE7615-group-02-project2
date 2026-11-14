# Milestone 2 Evaluation Review

## Guidance Scale Results

I reviewed the guidance-scale results and noticed that increasing guidance generally improved prompt alignment up to a certain point. The average CLIP score increased from 20.78 at guidance 1 to 25.97 at guidance 7.5.

However, increasing guidance beyond 7.5 did not lead to much improvement. Saturation also increased from 30.68 at guidance 1 to 57.51 at guidance 15, and the seed-similarity results suggest that higher guidance reduced some of the variation between outputs.

Based on these results, I agree with using guidance 7.5 as a reasonable balance between prompt alignment, image appearance, and diversity. Higher guidance is not necessarily better.

## Sampler and Sampling Steps

The sampler comparison shows that increasing the number of steps takes more time, but it does not always improve the CLIP score.

I also agree with adding the "difference to 50 steps" measure. CLIP score alone does not tell us whether the generated image is visually close to a higher-step output or whether important details are distorted.

DPM-Solver++ at 20 steps looks like a good option for faster generation. It took about 3.75 seconds per image compared with around 5.5 seconds at 30 steps. Its difference from the 50-step reference was also relatively low.

One limitation is that similarity to the 50-step image does not automatically mean better image quality. We should still review the generated images for missing objects, distorted features, and whether the prompts were followed correctly.

## CLIP vs. BERT with Adapter

The text-encoder comparison showed a clear improvement after adding the BERT adapter:

- CLIP: 28.78 average CLIP score
- Raw BERT: 9.88
- BERT with adapter: 21.40

The adapter improved BERT's compatibility with the diffusion pipeline, but CLIP still performed better.

I think this supports continuing with CLIP as the main conditioning method for the project. The BERT adapter was still a useful experiment because it showed that the conditioning embeddings matter and that an adapter can improve an otherwise poorly matched encoder.

## Recommendations for Milestone 3

For Milestone 3, I recommend evaluating the generated images using several measurements rather than relying only on CLIP score.

- **FID:** Compare the distribution of generated images against the real Flickr30k test images to evaluate overall visual realism and distribution similarity.
- **Inception Score:** Evaluate generated image quality and diversity, while recognizing that it does not measure prompt adherence or directly compare with real images.
- **CLIP Score:** Continue measuring how closely the generated images align with their captions.
- **Qualitative review:** Check for missing people or objects, distorted faces and hands, incorrect details, and differences between prompts.

For a fair comparison, we should use the same prompt set, keep seeds consistent where possible, and report generation settings and sample counts. FID should be interpreted carefully when using a small number of images.

Overall, the Milestone 2 results show that the text-to-image pipeline is working, but the best settings depend on balancing prompt alignment, diversity, image appearance, and generation time. I would carry these findings into the Milestone 3 evaluation.
