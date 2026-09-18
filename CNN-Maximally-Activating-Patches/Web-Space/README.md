---
title: CNN Maximally Activating Neuron Patches
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.10"
app_file: app.py
---

# CNN Maximally Activating Neuron Patches

Online Gradio version of the VGG16 maximally activating patch experiment.

The app scans CIFAR-10 test images, selects the maximum spatial activation for a
chosen VGG16 convolutional channel, ranks images by activation, and visualizes
the corresponding theoretical receptive-field region.

## Run online

This Space is designed for Hugging Face Spaces. For GPU acceleration, select
**ZeroGPU** in the Space hardware settings.

## Classroom starting settings

- Layer: `features.28 | Conv2d | 512 channels`
- Channel: `0`
- Images: `100`
- Top-K: `12`

Increase the number of images after the basic demonstration works.

## Local version

The original Tkinter/PyTorch desktop application remains in the GitHub
repository:

`cnn_maximally_activating_patches.py`

The first run downloads VGG16 ImageNet weights and CIFAR-10 automatically.
