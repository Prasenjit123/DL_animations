# CNN Filter Visualizer — RGB / AlexNet

This Streamlit application demonstrates the lecture idea: plot the learned first convolutional filters as images.

## Why this looks closer to the lecture slide

The MNIST version used grayscale filters `[16, 1, 5, 5]`. This version uses pretrained AlexNet:

```text
Natural RGB images
        ↓
AlexNet Conv1
        ↓
64 filters × 3 RGB channels × 11 × 11
        ↓
RGB filter visualization
```

This produces colored edge, texture and color-pattern patches much closer to the lecture-slide visualization.

## No MNIST download or training

This project does **not** use MNIST and does not train a model. On the first run, torchvision downloads the pretrained AlexNet weights. They are then cached locally.

## Run locally

```bash
conda activate torch-env
streamlit run app.py
```

## Visualization

- 64 Conv1 filters in an 8 × 8 grid
- selected filter enlarged as an RGB image
- R/G/B weight planes shown separately
- exact tensor shape: `[64, 3, 11, 11]`

## GitHub

```bash
git init
git add .
git commit -m "Add RGB CNN filter visualization"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cnn-filter-visualizer-rgb.git
git push -u origin main
```

For Streamlit Community Cloud use `app.py` as the main file.
