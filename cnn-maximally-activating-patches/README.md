# CNN Maximally Activating Neuron Patches

Interactive VGG16 experiment for demonstrating visualizing image patches that maximally activate a CNN neuron/channel.

## What it does

1. Loads ImageNet-pretrained VGG16.
2. Lets you select a VGG16 layer and channel.
3. Scans CIFAR-10 test images or a local image folder.
4. Finds the strongest spatial activation for the selected channel in each image.
5. Ranks images by activation strength.
6. Maps the activation location to the theoretical receptive field.
7. Displays the Top-K activating patches.
8. Exports patches and activation results as CSV.

## Install

Use Python 3.10-3.13. From Anaconda Prompt:

```bat
conda create -n cnn-interpretability python=3.11 -y
conda activate cnn-interpretability
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build appropriate for the computer. Follow the official PyTorch installation instructions rather than assuming the CUDA version in this requirements file.

## Run

```bat
python cnn_maximally_activating_patches.py
```

Or open the `.py` file in Spyder and run it.

On the first run, internet access is required to download VGG16 ImageNet weights and CIFAR-10. CIFAR-10 is stored locally in `data/`, which is excluded from GitHub.

## Recommended classroom settings

- Dataset: CIFAR-10 Test
- Layer: `features.28 | Conv2d | 512 channels`
- Channel: `0`
- Images: `1000`
- Batch size: `16`
- Top-K: `12`
- Patch mode: `Exact theoretical receptive field`

Then change the channel and compare the Top-K patches.

## Interpretation

For a selected channel:

```text
Input images -> VGG16 -> selected layer -> selected channel
             -> activation map -> maximum activation
             -> rank images -> map activation to input patch
```

A high activation means that the selected learned feature responded strongly at that spatial location. The CIFAR-10 class label should not automatically be treated as the semantic meaning of the channel; inspect repeated visual patterns across the Top-K patches.

## Important for GitHub

Do not upload the downloaded CIFAR-10 data, virtual environments, large model checkpoints, API keys, or passwords. The `.gitignore` file excludes the runtime data and common local files.

## Classroom note

A GitHub repository link shares the source code; it does not run this Tkinter desktop GUI in the browser. Students need Python installed locally (and optionally a CUDA-capable NVIDIA GPU for faster execution).
