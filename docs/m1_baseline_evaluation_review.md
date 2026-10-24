# Milestone 1 Baseline Evaluation Review

## Reproduction check

The CLIP embedding pipeline and Stable Diffusion baseline generation notebook were
independently rerun in Google Colab using a T4 GPU. Both notebooks completed without
errors and reproduced the expected Milestone 1 workflow.

The Stable Diffusion baseline generated images for 15 test prompts and produced the
expected gallery and CLIP-score outputs.

## Qualitative review

During the qualitative review of the Stable Diffusion baseline outputs, several common
issues were observed:

- **Sample 2:** The generated image did not produce the correct jersey number, and the
  players' hands were severely distorted.
- **Sample 4:** The people's feet and standing positions were noticeably distorted.
  Similar body-shape issues were also visible in other people in the image.
- **Sample 6:** The generated image did not fully follow the prompt and was missing
  one person.
- **Sample 7:** Several people described in the prompt were missing from the generated
  image.
- **Sample 8:** One person was missing, and some of the faces were slightly distorted.
- **Sample 9:** The brown dog described in the prompt was missing.
- **Sample 10:** Several faces were distorted, and parts of the background also
  appeared unrealistic.
- **Sample 11:** One person was missing, and the generated faces were distorted.
- **Sample 12:** The model did not appear to correctly understand or represent the
  roller coaster described in the prompt.
- **Sample 13:** The children's faces were noticeably distorted.
- **Sample 14:** The man, vehicles, and dog all showed visible distortion.

## Baseline analysis

The baseline results show that Stable Diffusion was generally able to produce images
related to the input captions, but the quality of the conditional sampling was
inconsistent. Several generated images were missing people, animals, or other objects
described in the prompt, showing that the model did not always fully follow the text
condition. Human anatomy was also a repeated weakness, especially with faces, hands,
feet, and body positioning.

The results also show the difference between generative and discriminative modeling.
A discriminative model would mainly predict a label or class, while the diffusion model
has to generate a complete image from a latent representation conditioned on the text
prompt. Because of this, it can create a visually realistic scene while still getting
important details wrong, such as the jersey number in Sample 2 or the missing dog in
Sample 9.

There was no clear evidence of complete mode collapse because the generated images were
visually different across prompts. However, the model still struggled with more complex
scenes containing multiple people, animals, vehicles, or specific objects. These
results suggest that the baseline captures the general meaning of many captions, but
detailed prompt adherence and realistic human features remain important limitations.

## Recommendation

Overall, the baseline is solid and reproducible enough to proceed to the next milestones.
The model successfully captures the general content of many captions, the notebooks can be
rerun consistently, and the outputs align with the expected Milestone 1 workflow. The
issues observed are not signs of a broken pipeline; rather, they reflect the known
challenges of text-to-image generation for complex scenes and human anatomy.

The main recommendation is to keep the current baseline as a valid starting point, while
prioritizing improvements in three areas for Milestone 2 and 3: prompt fidelity for
multiple-entity scenes, improved realism of faces and hands, and stronger control over
specific objects and counts. In practice, this means tuning sampling settings such as
guidance scale, scheduler choice, and step count, while also evaluating generation quality
with CLIP, FID, and human inspection. With these targeted refinements, the project should
be able to move beyond a general baseline toward a more reliable and visually consistent
text-to-image model.
