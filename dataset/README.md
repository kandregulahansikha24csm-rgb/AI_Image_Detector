# Dataset Guide

Use this folder structure:

- dataset/real/  -> real photos
- dataset/ai/    -> AI-generated images

## Recommended sizes
- Minimum for a mini project: 100 real + 100 AI
- Better: 300 real + 300 AI
- Stronger: 500+ real + 500+ AI

## Good public sources
- Real photos:
  - Open Images
  - CIFAR-10 / CIFAR-100
  - Kaggle photo datasets
- AI-generated images:
  - Kaggle datasets for AI-generated or synthetic images
  - Stable Diffusion / GAN-generated image collections

## Best practice
- Keep the number of images balanced between real and AI.
- Use different lighting, backgrounds, resolutions, and styles.
- Avoid using the same image in both folders.
- Keep a few images aside for testing only.

## Simple workflow
1. Download or collect images.
2. Put real photos in dataset/real.
3. Put AI-generated images in dataset/ai.
4. Run: python train_model.py

## Note
A larger and more varied dataset will improve accuracy much more than changing the model alone.
