# Referee Can Play: An Alternative Approach to Conditional Generation via Model Inversion (ICML2024)
This is the codebase for the paper: [Referee Can Play: An Alternative Approach to Conditional Generation via Model Inversion](https://arxiv.org/abs/2402.16305), accepted by ICML2024

Our work primarily focuses on providing a novel approach to enhance the **controllability** of text-to-image generation. Based on a **new understanding** of the training and inference paradigms of state-of-the-art text-to-image models such as DALL-E 3 and Pixart, we have recognized the significant role of **discriminative Vision Language Models (VLMs)** in text-to-image tasks. We propose a new understanding of the existing diffusion probabilistic models (DPMs) for text-to-image tasks, i.e., a learned inversion of VLM; then based on **VLM inversion**, we have designed an alternative text-to-image generation pipeline. We conducted a series of validation experiments using BLIP2 as a proof of concept to demonstrate the effectiveness of our method.

If you want to know the BLIP-VQA score of the generated image, you need to import LAVIS by
```
git clone https://github.com/salesforce/LAVIS.git
```
and replace the **blip_vqa.py** under the path `lavis/models/blip_models/` with our **blip_vqa.py** under `lavis`.

Then, after importing all the required packages according to **requirements.txt**, you can easily try to generate images by running the code below:

```bash
python generate.py --save
```

Note that you may need a GPU with memory > 30Gb to generate images with resolution 512*512 with at least 10 augmentations to ensure the image quality.
Note that our method is not limited to any specific model! If you are interested, we welcome you to try replacing BLIP2 with other discriminative VLMs. Any discussion or potential collaboration is welcomed! My email address: xliude@connect.ust.hk
