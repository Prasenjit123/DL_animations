"""
Online Gradio version of the CNN Maximally Activating Neuron Patches experiment.

Designed for Hugging Face Spaces / Gradio.
It preserves the core experiment:
CIFAR-10 -> pretrained VGG16 -> selected convolutional layer/channel
-> maximum spatial activation per image -> rank -> theoretical receptive-field crop.

For Hugging Face ZeroGPU, select ZeroGPU as the Space hardware.
"""

import spaces  # Keep this before torch for ZeroGPU compatibility.

import math
import os
from typing import List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Load model once. On ZeroGPU, CUDA is emulated at startup and the real GPU
# is allocated when a @spaces.GPU function is called.
print("Loading VGG16...")
MODEL = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    MODEL = MODEL.to(DEVICE)
except Exception:
    DEVICE = torch.device("cpu")
    MODEL = MODEL.to(DEVICE)
print("Model device:", DEVICE)


def calculate_receptive_fields(model):
    rf = 1.0
    jump = 1.0
    start = 0.5
    info = {}

    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            k = layer.kernel_size[0]
            s = layer.stride[0]
            p = layer.padding[0]
        elif isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            if s is None:
                s = k
            p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        else:
            info[i] = {"rf": rf, "jump": jump, "start": start}
            continue

        start += ((k - 1) / 2.0 - p) * jump
        rf += (k - 1) * jump
        jump *= s
        info[i] = {"rf": rf, "jump": jump, "start": start}

    return info


RF_INFO = calculate_receptive_fields(MODEL)

CONV_LAYERS = []
for i, layer in enumerate(MODEL.features):
    if isinstance(layer, nn.Conv2d):
        CONV_LAYERS.append((i, layer.out_channels))

LAYER_CHOICES = [
    f"features.{i} | Conv2d | {channels} channels"
    for i, channels in CONV_LAYERS
]
LAYER_MAP = {text: i for text, (i, _) in zip(LAYER_CHOICES, CONV_LAYERS)}

DEFAULT_LAYER = next(x for x in LAYER_CHOICES if x.startswith("features.28"))


class ActivationHook:
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output.detach()

    def close(self):
        self.handle.remove()


def channel_count(layer_text):
    idx = LAYER_MAP[layer_text]
    for i, channels in CONV_LAYERS:
        if i == idx:
            return channels
    return 1


def receptive_field_crop(fx, fy, rf_info, image_size=224):
    center_x = rf_info["start"] + fx * rf_info["jump"]
    center_y = rf_info["start"] + fy * rf_info["jump"]
    size = int(round(rf_info["rf"]))
    half = size / 2.0

    left = max(0, int(round(center_x - half)))
    top = max(0, int(round(center_y - half)))
    right = min(image_size, int(round(center_x + half)))
    bottom = min(image_size, int(round(center_y + half)))

    return (left, top, right, bottom), (center_x, center_y)


def make_visual(original, crop_box, label, activation, rank):
    """Create an educational side-by-side: original with activation box + patch."""
    image = original.resize((INPUT_SIZE, INPUT_SIZE))
    marked = image.copy()
    draw = ImageDraw.Draw(marked)

    left, top, right, bottom = crop_box
    # Draw a simple high-contrast box without depending on a fixed color palette.
    draw.rectangle([left, top, right, bottom], outline="white", width=4)
    draw.rectangle([left + 2, top + 2, right - 2, bottom - 2], outline="black", width=2)

    patch = image.crop(crop_box)

    canvas = Image.new("RGB", (INPUT_SIZE * 2 + 12, INPUT_SIZE), "white")
    canvas.paste(marked, (0, 0))
    patch.thumbnail((INPUT_SIZE, INPUT_SIZE))
    px = INPUT_SIZE + 12 + (INPUT_SIZE - patch.width) // 2
    py = (INPUT_SIZE - patch.height) // 2
    canvas.paste(patch, (px, py))

    return canvas


@spaces.GPU(duration=120)
def run_experiment(layer_text: str, channel: int, n_images: int, top_k: int):
    try:
        layer_idx = LAYER_MAP[layer_text]
        n_channels = channel_count(layer_text)
        channel = int(channel)
        n_images = int(n_images)
        top_k = int(top_k)

        if not 0 <= channel < n_channels:
            raise gr.Error(f"Channel must be between 0 and {n_channels - 1}.")
        if n_images < 1:
            raise gr.Error("Images to scan must be at least 1.")
        if top_k < 1:
            raise gr.Error("Top-K must be at least 1.")

        dataset = datasets.CIFAR10(
            root=DATA_ROOT,
            train=False,
            download=True,
            transform=TRANSFORM,
        )
        n_images = min(n_images, len(dataset))
        top_k = min(top_k, n_images)

        # Ensure model is on the available device in hosted/local environments.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MODEL
        try:
            model = model.to(device)
        except Exception:
            device = torch.device("cpu")
            model = model.to(device)
        model.eval()

        batch_size = 32 if device.type == "cuda" else 8
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, range(n_images)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        hook = ActivationHook(model.features[layer_idx])
        records = []

        try:
            processed = 0
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(device, non_blocking=True)
                    model(x)
                    act = hook.activation

                    if act is None or act.ndim != 4:
                        raise gr.Error("Selected layer did not produce a 4-D activation map.")

                    selected = act[:, channel]
                    values, flat = selected.reshape(selected.shape[0], -1).max(dim=1)
                    h, w = selected.shape[-2:]

                    ys = (flat // w).cpu().numpy()
                    xs = (flat % w).cpu().numpy()
                    values = values.cpu().numpy()

                    for j, value in enumerate(values):
                        idx = processed + j
                        records.append({
                            "dataset_index": idx,
                            "activation": float(value),
                            "feature_x": int(xs[j]),
                            "feature_y": int(ys[j]),
                            "label": dataset.classes[dataset.targets[idx]],
                        })
                    processed += len(values)
        finally:
            hook.close()

        results = pd.DataFrame(records).sort_values(
            "activation", ascending=False
        ).reset_index(drop=True)

        rf = RF_INFO[layer_idx]
        gallery = []

        for rank, row in results.head(top_k).iterrows():
            idx = int(row["dataset_index"])
            original = Image.fromarray(dataset.data[idx]).convert("RGB")
            crop_box, center = receptive_field_crop(
                int(row["feature_x"]),
                int(row["feature_y"]),
                rf,
                image_size=INPUT_SIZE,
            )

            visual = make_visual(
                original,
                crop_box,
                row["label"],
                row["activation"],
                rank + 1,
            )

            caption = (
                f"Rank {rank + 1} | {row['label']} | "
                f"activation={row['activation']:.3f} | "
                f"feature=({int(row['feature_x'])}, {int(row['feature_y'])})"
            )
            gallery.append((visual, caption))

        mean_activation = float(results["activation"].mean())
        top_mean = float(results.head(top_k)["activation"].mean())
        ratio = top_mean / mean_activation if mean_activation != 0 else float("inf")

        summary = f"""
### Experiment complete

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 Test |
| Images scanned | {n_images} |
| Layer | `features.{layer_idx}` |
| Channel | `{channel}` |
| Device | `{device}` |
| Theoretical receptive field | **{rf['rf']:.0f} × {rf['rf']:.0f}** |
| Feature-map jump | {rf['jump']:.0f} pixels |
| Mean activation | {mean_activation:.4f} |
| Top-{top_k} mean activation | {top_mean:.4f} |
| Top-K / all-images mean | {ratio:.3f} |

**Interpretation:** Each gallery item shows the original 224×224 input with the theoretical receptive-field region marked, alongside the corresponding extracted patch. The ranking is based on the maximum spatial activation of the selected VGG16 channel for each image.
"""

        # Move back to CPU after the request so the next request can acquire
        # a fresh ZeroGPU allocation cleanly.
        try:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return summary, gallery

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Experiment failed: {type(exc).__name__}: {exc}")


def update_channel_range(layer_text):
    n = channel_count(layer_text)
    return gr.update(maximum=n - 1, value=0)


with gr.Blocks(title="CNN Maximally Activating Neuron Patches") as demo:
    gr.Markdown(
        """
# CNN Maximally Activating Neuron Patches

**Interactive VGG16 interpretability experiment**

This demonstration finds real CIFAR-10 images that produce the strongest spatial
activation for a selected VGG16 convolutional channel and maps that activation
back to its theoretical input receptive field.

### What to look for

1. Select a VGG16 convolutional layer.
2. Select a channel (feature detector).
3. Scan a set of CIFAR-10 test images.
4. For every image, find the spatial maximum activation.
5. Rank the images by activation.
6. Visualize the corresponding receptive-field patches.

**Online demo default:** 100 images and Top-12 patches. Increase the image count
for a broader search, but larger scans take longer.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            layer = gr.Dropdown(
                choices=LAYER_CHOICES,
                value=DEFAULT_LAYER,
                label="VGG16 convolutional layer",
            )
            channel = gr.Number(
                value=0,
                precision=0,
                minimum=0,
                maximum=511,
                label="Neuron / Channel",
            )
            n_images = gr.Slider(
                minimum=10,
                maximum=1000,
                value=100,
                step=10,
                label="Images to scan",
            )
            top_k = gr.Slider(
                minimum=4,
                maximum=20,
                value=12,
                step=1,
                label="Top-K patches",
            )
            run = gr.Button("RUN EXPERIMENT", variant="primary")

        with gr.Column(scale=2):
            summary = gr.Markdown(
                "Choose a layer and channel, then click **RUN EXPERIMENT**."
            )

    gallery = gr.Gallery(
        label="Top maximally activating patches — original image + receptive-field patch",
        columns=3,
        rows=4,
        height="auto",
        object_fit="contain",
    )

    layer.change(
        update_channel_range,
        inputs=layer,
        outputs=channel,
    )

    run.click(
        run_experiment,
        inputs=[layer, channel, n_images, top_k],
        outputs=[summary, gallery],
    )

if __name__ == "__main__":
    demo.launch()
