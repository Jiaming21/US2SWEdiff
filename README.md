# US2SWEdiff

Shear Wave Elastography (SWE) offers elasticity maps that reflect tumor stiffness, assisting clinicians in determining tumor malignancy. However, unlike conventional Ultrasound (US) imaging, SWE is relatively expensive, and there is currently no publicly available SWE image dataset.

Therefore, in this project, we aim to generate SWE images from US images with the following logistics:

1. Construct baselines with SOTA models designed for modality transfer and a SOTA GAN model used for conditional generation (pix2pixHD).

2. Develop **US2SWEdiff**, a ControlNet-based conditional diffusion model that generates breast tumors’ SWE images from corresponding US images.  
   Specifically, we examine three types of conditional inputs:
   - Canny edge maps  
   - Laplacian edge maps  
   - US images  

3. Further improve the hint input module to shorten convergence time and enhance the quality of the generated SWE images.

4. Build a website for SWE image–based breast tumor malignancy classification, along with a Gradio interface for SWE image generation using **US2SWEdiff**.

Additionally, to enable unconditional generation, we aim to replace explicit prompt assignment with a trained tumor malignancy classifier that assigns prompts based on its predictions.
