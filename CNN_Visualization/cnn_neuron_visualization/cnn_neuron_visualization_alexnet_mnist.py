
# -*- coding: utf-8 -*-
"""
CNN NEURON / FILTER VISUALIZATION — ALEXNET-STYLE MNIST
-------------------------------------------------------
Purpose:
    Select ANY convolutional layer and ANY filter/channel.
    Find the Top-K MNIST images that produce the strongest
    response in that filter, then show WHERE in each image
    the strongest response occurs.

Important terminology:
    A "filter/channel" is a learned convolution kernel.
    A "neuron" is one spatial position inside that channel.
    For an image, the image-level filter score is:
        max over (H,W) of the selected feature map.
    Therefore Top-K = images in which the selected filter
    reaches its largest activation.

Activation choice:
    PRE-RELU  = convolution output before ReLU
    POST-RELU = activation after ReLU (default)

Receptive field:
    The red box is the theoretical receptive field of the
    selected spatial neuron mapped back to the original 28x28
    MNIST image.

This is intentionally a single-file Tkinter program so it is
easy to run from Spyder and later upload to GitHub.
"""

import os
import time
import threading
import traceback
from pathlib import Path

# Prevent common Intel OpenMP duplicate-runtime crashes in Anaconda/Spyder.
# This is a compatibility safeguard, NOT the solution to the Tkinter issue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT = RESULTS_DIR / "alexnet_mnist.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4-GB GPU friendly
BATCH_SIZE = 64
TRAIN_EPOCHS = 15
DEFAULT_SCAN = 5000
DEFAULT_TOPK = 8

torch.backends.cudnn.benchmark = True

# ============================================================
# MODEL
# ============================================================

class AlexNetMNIST(nn.Module):
    """
    AlexNet-style CNN adapted for 28x28 MNIST.

    Five convolutional stages are exposed individually:
        conv1, conv2, conv3, conv4, conv5

    Each stage is:
        Conv2d -> ReLU -> MaxPool

    This is NOT the ImageNet AlexNet checkpoint.
    It is an AlexNet-style architecture TRAINED on MNIST.
    This is preferable for teaching "what activates a filter"
    because the learned filters are actually trained for digits.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.relu5 = nn.ReLU(inplace=False)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 28 -> 14 -> 7 -> 3 -> 1 -> 0 would be invalid if all
        # five pools are used. Therefore pool only through conv4.
        # conv5 remains at 1x1 after conv4 pooling.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x, return_features=False):
        feats = {}

        x = self.conv1(x)
        feats["conv1_pre"] = x
        x = self.relu1(x)
        feats["conv1_post"] = x
        x = self.pool1(x)

        x = self.conv2(x)
        feats["conv2_pre"] = x
        x = self.relu2(x)
        feats["conv2_post"] = x
        x = self.pool2(x)

        x = self.conv3(x)
        feats["conv3_pre"] = x
        x = self.relu3(x)
        feats["conv3_post"] = x
        x = self.pool3(x)

        x = self.conv4(x)
        feats["conv4_pre"] = x
        x = self.relu4(x)
        feats["conv4_post"] = x
        x = self.pool4(x)

        x = self.conv5(x)
        feats["conv5_pre"] = x
        x = self.relu5(x)
        feats["conv5_post"] = x

        logits = self.classifier(x)

        if return_features:
            return logits, feats
        return logits


# ============================================================
# DATA
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

DISPLAY_TRANSFORM = transforms.ToTensor()


def get_datasets():
    print("Checking MNIST dataset...")
    print(f"Data directory: {DATA_DIR}")

    train_ds = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=TRANSFORM
    )

    test_ds = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=TRANSFORM
    )

    print(f"MNIST train: {len(train_ds)}")
    print(f"MNIST test : {len(test_ds)}")
    return train_ds, test_ds


# ============================================================
# TRAIN / LOAD
# ============================================================

def train_model(progress=None):
    train_ds, test_ds = get_datasets()

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # safest for Spyder/Windows
        pin_memory=(DEVICE.type == "cuda")
    )

    model = AlexNetMNIST().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("=" * 70)
    print("TRAINING ALEXNET-STYLE MNIST MODEL")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for b, (images, labels) in enumerate(loader, 1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if b % 100 == 0:
                acc = 100.0 * correct / total
                msg = (
                    f"Epoch {epoch+1}/{TRAIN_EPOCHS} | "
                    f"batch {b}/{len(loader)} | "
                    f"loss {loss.item():.4f} | "
                    f"train acc {acc:.2f}%"
                )
                print(msg)
                if progress:
                    progress(msg)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        elapsed = time.time() - start

        msg = (
            f"Epoch {epoch+1}/{TRAIN_EPOCHS} complete | "
            f"loss {epoch_loss:.4f} | "
            f"train acc {epoch_acc:.2f}% | "
            f"{elapsed:.1f}s"
        )
        print(msg)
        if progress:
            progress(msg)

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in DataLoader(
            test_ds, batch_size=256, shuffle=False, num_workers=0,
            pin_memory=(DEVICE.type == "cuda")
        ):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = 100.0 * correct / total
    print(f"MNIST test accuracy: {test_acc:.2f}%")
    if progress:
        progress(f"Test accuracy: {test_acc:.2f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "AlexNetMNIST",
            "test_accuracy": test_acc,
        },
        CHECKPOINT
    )
    print(f"Saved model: {CHECKPOINT}")

    return model, test_ds


def load_or_train(progress=None):
    test_ds = None
    model = AlexNetMNIST().to(DEVICE)

    if CHECKPOINT.exists():
        try:
            payload = torch.load(
                CHECKPOINT,
                map_location=DEVICE,
                weights_only=True
            )
            model.load_state_dict(payload["model_state"])
            model.eval()

            _, test_ds = get_datasets()

            acc = payload.get("test_accuracy", None)
            print(f"Loaded trained model: {CHECKPOINT}")
            if acc is not None:
                print(f"Stored MNIST test accuracy: {acc:.2f}%")
            return model, test_ds

        except Exception as e:
            print("Existing checkpoint could not be loaded.")
            print("Reason:", e)
            print("A fresh model will be trained.")

    return train_model(progress)


# ============================================================
# LAYER INFORMATION
# ============================================================

LAYER_INFO = {
    "conv1": {
        "channels": 64,
        "description": "Early layer: edges, stroke directions, simple curves.",
        "pool_after": True,
    },
    "conv2": {
        "channels": 128,
        "description": "Mid-early layer: combinations of strokes and corners.",
        "pool_after": True,
    },
    "conv3": {
        "channels": 256,
        "description": "Middle layer: more structured digit parts.",
        "pool_after": True,
    },
    "conv4": {
        "channels": 256,
        "description": "Deep layer: larger combinations of digit structure.",
        "pool_after": True,
    },
    "conv5": {
        "channels": 256,
        "description": "Deepest convolution: high-level digit patterns.",
        "pool_after": False,
    },
}

# Theoretical receptive fields for this architecture.
# Every convolution: k=3, s=1, p=1
# Pool: k=2, s=2
#
# Track:
# conv1: rf 3, jump 1
# pool1: rf 4, jump 2
# conv2: rf 8, jump 2
# pool2: rf 10, jump 4
# conv3: rf 18, jump 4
# pool3: rf 22, jump 8
# conv4: rf 38, jump 8
# pool4: rf 46, jump 16
# conv5: rf 78, jump 16
RF_INFO = {
    "conv1": (3, 1),
    "conv2": (8, 2),
    "conv3": (18, 4),
    "conv4": (38, 8),
    "conv5": (78, 16),
}


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def tensor_to_uint8(t):
    """Convert normalized MNIST tensor back to grayscale uint8."""
    x = t.detach().cpu().squeeze().numpy()
    x = x * 0.3081 + 0.1307
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def crop_rf(img28, fmap_y, fmap_x, layer):
    """
    Map a feature-map spatial coordinate to its theoretical RF
    in the original 28x28 image.

    Uses the standard center-coordinate recurrence.
    """
    rf, jump = RF_INFO[layer]

    # Need layer-specific start offset.
    # For odd k=3 conv with padding 1, center unchanged.
    # Each 2x2 pool shifts center by 0.5 * jump.
    start = 0.5

    # Calculate layer start positions explicitly.
    starts = {
        "conv1": 0.5,
        "conv2": 1.0,
        "conv3": 2.0,
        "conv4": 4.0,
        "conv5": 8.0,
    }

    start = starts[layer]

    cy = start + fmap_y * jump
    cx = start + fmap_x * jump

    half = rf / 2.0
    x0 = int(np.floor(cx - half))
    y0 = int(np.floor(cy - half))
    x1 = int(np.ceil(cx + half))
    y1 = int(np.ceil(cy + half))

    x0 = max(0, min(27, x0))
    y0 = max(0, min(27, y0))
    x1 = max(x0 + 1, min(28, x1))
    y1 = max(y0 + 1, min(28, y1))

    crop = img28[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1), (cx, cy)


def make_original_image(img28, box, point, size=220):
    im = Image.fromarray(img28, mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    ).convert("RGB")

    draw = ImageDraw.Draw(im)

    x0, y0, x1, y1 = box
    sx = size / 28.0
    sy = size / 28.0

    draw.rectangle(
        [x0*sx, y0*sy, x1*sx, y1*sy],
        outline=(255, 0, 0),
        width=4
    )

    px, py = point
    r = 5
    draw.ellipse(
        [(px*sx-r, py*sy-r), (px*sx+r, py*sy+r)],
        fill=(255, 255, 0),
        outline=(0, 0, 0),
        width=2
    )

    return im


def make_crop_image(crop, size=220):
    im = Image.fromarray(crop, mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    ).convert("RGB")
    return im


# ============================================================
# SCANNING
# ============================================================

def scan_topk(model, dataset, layer, channel, scan_n, topk,
              activation_mode, progress=None):
    """
    Correct definition of "what activates a filter":

    For each image:
        feature_map = selected channel
        score = max(feature_map)

    Then rank images by score.

    Also stores the spatial location of that maximum.
    """
    model.eval()

    scan_n = min(scan_n, len(dataset))
    topk = min(topk, scan_n)

    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(scan_n)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    results = []

    # GUI labels use "POST-RELU"/"PRE-RELU", while the model feature
    # dictionary uses the canonical keys "layer_post"/"layer_pre".
    # Keep this mapping explicit so the selected activation mode can
    # never produce a KeyError.
    mode_key = {
        "POST-RELU": "post",
        "PRE-RELU": "pre",
    }.get(activation_mode)

    if mode_key is None:
        raise ValueError(
            f"Unknown activation mode: {activation_mode!r}. "
            "Expected POST-RELU or PRE-RELU."
        )

    key = f"{layer}_{mode_key}"

    with torch.inference_mode():
        done = 0

        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)

            _, feats = model(images, return_features=True)
            fmap = feats[key][:, channel, :, :]

            # One scalar per image: maximum response of this filter.
            scores, flat_idx = torch.max(
                fmap.reshape(fmap.size(0), -1), dim=1
            )

            h, w = fmap.shape[-2:]

            ys = torch.div(flat_idx, w, rounding_mode="floor")
            xs = flat_idx % w

            for i in range(images.size(0)):
                results.append({
                    "dataset_index": done + i,
                    "label": int(labels[i].item()),
                    "score": float(scores[i].item()),
                    "fy": int(ys[i].item()),
                    "fx": int(xs[i].item()),
                })

            done += images.size(0)

            if progress and done % 512 < 64:
                progress(f"Scanning images: {done}/{scan_n}")

    results.sort(key=lambda z: z["score"], reverse=True)
    return results[:topk]


# ============================================================
# GUI
# ============================================================

class App:
    def __init__(self, root, model, dataset):
        self.root = root
        self.model = model
        self.dataset = dataset
        self.current_results = []

        self.root.title("CNN Neuron Visualization — AlexNet-style + MNIST")
        self.root.geometry("1450x900")
        self.root.minsize(1200, 760)

        self.build_ui()

    def build_ui(self):
        # Main horizontal split.
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="ns")

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(
            left,
            text="EXPERIMENT CONTROLS",
            font=("Arial", 13, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        ttk.Label(left, text="Dataset").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(
            left,
            text="MNIST test set • 10,000 handwritten digits"
        ).grid(row=2, column=0, sticky="w", pady=(0, 12))

        ttk.Label(left, text="Network").grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(
            left,
            text="AlexNet-style CNN\n5 convolutional layers\ntrained on MNIST"
        ).grid(row=4, column=0, sticky="w", pady=(0, 12))

        ttk.Label(left, text="Convolutional layer").grid(
            row=5, column=0, sticky="w"
        )

        self.layer_var = tk.StringVar(value="conv1")
        self.layer_combo = ttk.Combobox(
            left,
            textvariable=self.layer_var,
            values=list(LAYER_INFO.keys()),
            state="readonly",
            width=20
        )
        self.layer_combo.grid(row=6, column=0, sticky="ew", pady=4)
        self.layer_combo.bind("<<ComboboxSelected>>", self.update_channel_range)

        ttk.Label(left, text="Filter / channel").grid(
            row=7, column=0, sticky="w", pady=(10, 0)
        )

        self.channel_var = tk.IntVar(value=0)
        self.channel_spin = tk.Spinbox(
            left,
            from_=0,
            to=63,
            textvariable=self.channel_var,
            width=10
        )
        self.channel_spin.grid(row=8, column=0, sticky="w", pady=4)

        self.channel_info = ttk.Label(left, text="Channel 0 of 64")
        self.channel_info.grid(row=9, column=0, sticky="w")

        ttk.Label(left, text="Activation used for ranking").grid(
            row=10, column=0, sticky="w", pady=(12, 0)
        )

        self.activation_var = tk.StringVar(value="POST-RELU")
        self.activation_combo = ttk.Combobox(
            left,
            textvariable=self.activation_var,
            values=["POST-RELU", "PRE-RELU"],
            state="readonly",
            width=20
        )
        self.activation_combo.grid(row=11, column=0, sticky="ew", pady=4)

        ttk.Label(
            left,
            text="POST-RELU = actual signal after ReLU\n"
                 "PRE-RELU = raw convolution output"
        ).grid(row=12, column=0, sticky="w")

        ttk.Label(left, text="Images to scan").grid(
            row=13, column=0, sticky="w", pady=(12, 0)
        )

        self.scan_var = tk.IntVar(value=DEFAULT_SCAN)
        tk.Spinbox(
            left, from_=100, to=10000,
            increment=100,
            textvariable=self.scan_var,
            width=10
        ).grid(row=14, column=0, sticky="w")

        ttk.Label(left, text="Top-K").grid(
            row=15, column=0, sticky="w", pady=(12, 0)
        )

        self.topk_var = tk.IntVar(value=DEFAULT_TOPK)
        tk.Spinbox(
            left, from_=1, to=12,
            textvariable=self.topk_var,
            width=10
        ).grid(row=16, column=0, sticky="w")

        self.run_button = ttk.Button(
            left,
            text="RUN TOP-K VISUALIZATION",
            command=self.start_experiment
        )
        self.run_button.grid(
            row=17, column=0, sticky="ew", pady=(18, 8)
        )

        self.status_var = tk.StringVar(
            value="Ready. Select layer + filter and run."
        )
        ttk.Label(
            left,
            textvariable=self.status_var,
            wraplength=260
        ).grid(row=18, column=0, sticky="w", pady=(10, 0))

        ttk.Separator(left).grid(
            row=19, column=0, sticky="ew", pady=15
        )

        ttk.Label(
            left,
            text="WHAT STUDENTS SHOULD LEARN",
            font=("Arial", 11, "bold")
        ).grid(row=20, column=0, sticky="w")

        explanation = (
            "1. One filter is shared across every image.\n\n"
            "2. For each image we measure the maximum "
            "activation of that filter.\n\n"
            "3. Top-K = images producing the strongest "
            "responses.\n\n"
            "4. The yellow dot is the strongest spatial "
            "neuron in the selected feature map.\n\n"
            "5. The red box is its theoretical receptive "
            "field in the original 28×28 image.\n\n"
            "6. Different digits can activate the same "
            "filter because the filter detects a visual "
            "pattern, not a class label.\n\n"
            "7. Compare layers: early filters usually "
            "detect simple strokes; deeper filters combine "
            "larger structures."
        )

        ttk.Label(
            left,
            text=explanation,
            wraplength=275,
            justify="left"
        ).grid(row=21, column=0, sticky="w", pady=(6, 0))

        ttk.Label(
            right,
            text="WHAT ACTIVATES A FILTER?",
            font=("Arial", 20, "bold")
        ).grid(row=0, column=0, sticky="n", pady=(0, 8))

        # Scrollable results canvas.
        self.canvas = tk.Canvas(right, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            right, orient="vertical",
            command=self.canvas.yview
        )
        self.results_frame = ttk.Frame(self.canvas)

        self.results_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.results_frame,
            anchor="nw"
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.scrollbar.grid(row=1, column=1, sticky="ns")

        self.canvas.bind(
            "<Configure>",
            self.on_canvas_resize
        )

        self.update_channel_range()

    def on_canvas_resize(self, event):
        self.canvas.itemconfigure(
            self.canvas_window,
            width=event.width
        )

    def update_channel_range(self, event=None):
        layer = self.layer_var.get()
        n = LAYER_INFO[layer]["channels"]

        self.channel_spin.config(to=n - 1)

        try:
            c = int(self.channel_var.get())
        except Exception:
            c = 0

        if c >= n:
            self.channel_var.set(0)

        self.channel_info.config(
            text=f"Channel {self.channel_var.get()} of {n}"
        )

    def clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def start_experiment(self):
        try:
            layer = self.layer_var.get()
            channel = int(self.channel_var.get())
            scan_n = int(self.scan_var.get())
            topk = int(self.topk_var.get())
            activation = self.activation_var.get()

            if channel < 0 or channel >= LAYER_INFO[layer]["channels"]:
                raise ValueError("Invalid channel number.")

            if scan_n < 1 or scan_n > len(self.dataset):
                raise ValueError(
                    f"Images to scan must be between 1 and {len(self.dataset)}."
                )

            if topk < 1:
                raise ValueError("Top-K must be at least 1.")

        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        self.run_button.config(state="disabled")
        self.status_var.set(
            "RUNNING: measuring the selected filter on every image..."
        )
        self.clear_results()

        thread = threading.Thread(
            target=self.worker,
            args=(layer, channel, scan_n, topk, activation),
            daemon=True
        )
        thread.start()

    def worker(self, layer, channel, scan_n, topk, activation):
        try:
            messages = []

            def progress(msg):
                messages.append(msg)

            top = scan_topk(
                self.model,
                self.dataset,
                layer,
                channel,
                scan_n,
                topk,
                activation,
                progress=progress
            )

            self.root.after(
                0,
                lambda: self.show_results(
                    top, layer, channel, activation, messages
                )
            )

        except Exception:
            error = traceback.format_exc()
            self.root.after(
                0,
                lambda: self.show_error(error)
            )

    def show_error(self, error):
        self.run_button.config(state="normal")
        self.status_var.set("ERROR — see message box.")
        messagebox.showerror(
            "Experiment error",
            error
        )

    def show_results(self, top, layer, channel, activation, messages):
        self.run_button.config(state="normal")
        self.current_results = top

        rf, jump = RF_INFO[layer]
        n = LAYER_INFO[layer]["channels"]

        self.status_var.set(
            f"COMPLETE. Top-{len(top)} strongest responses displayed."
        )

        ttk.Label(
            self.results_frame,
            text=(
                f"Selected: {layer} • filter {channel} • "
                f"{activation}\n"
                f"Feature channels: {n} • theoretical RF: {rf}×{rf} px"
            ),
            font=("Arial", 12, "bold"),
            justify="left"
        ).grid(
            row=0, column=0, columnspan=4,
            sticky="w", padx=12, pady=(8, 15)
        )

        ttk.Label(
            self.results_frame,
            text=(
                "Ranking rule: for every image, take the MAX value "
                "over all spatial positions of the selected filter. "
                "Top-K are the images with the largest maxima.\n"
                "POST-RELU = after ReLU (negative responses removed); "
                "PRE-RELU = raw convolution output before ReLU."
            ),
            justify="left"
        ).grid(
            row=1, column=0, columnspan=4,
            sticky="w", padx=12, pady=(0, 12)
        )

        # Four results per row.
        for i, item in enumerate(top):
            grid_row = 2 + (i // 4) * 2
            grid_col = i % 4

            card = ttk.Frame(
                self.results_frame,
                padding=8,
                relief="ridge"
            )
            card.grid(
                row=grid_row,
                column=grid_col,
                padx=8,
                pady=8,
                sticky="n"
            )

            image = self.dataset[item["dataset_index"]][0]
            img28 = tensor_to_uint8(image)

            crop, box, point = crop_rf(
                img28,
                item["fy"],
                item["fx"],
                layer
            )

            original = make_original_image(
                img28, box, point, size=190
            )
            crop_im = make_crop_image(crop, size=190)

            original_tk = ImageTk.PhotoImage(original)
            crop_tk = ImageTk.PhotoImage(crop_im)

            ttk.Label(
                card,
                text=(
                    f"#{i+1}  digit {item['label']}\n"
                    f"activation = {item['score']:.4f}"
                ),
                font=("Arial", 11, "bold")
            ).grid(row=0, column=0, columnspan=2, pady=(0, 5))

            # IMPORTANT: use grid only. No pack/grid mixing.
            lab1 = ttk.Label(card, image=original_tk)
            lab1.image = original_tk
            lab1.grid(row=1, column=0, padx=4)

            lab2 = ttk.Label(card, image=crop_tk)
            lab2.image = crop_tk
            lab2.grid(row=1, column=1, padx=4)

            ttk.Label(
                card,
                text=(
                    f"Feature-map location = "
                    f"({item['fy']}, {item['fx']})\n"
                    f"RF = {rf}×{rf} • jump = {jump}\n"
                    f"LEFT: original + RF\n"
                    f"RIGHT: exact RF crop"
                ),
                justify="left"
            ).grid(
                row=2, column=0, columnspan=2,
                sticky="w", pady=(5, 0)
            )

        for c in range(4):
            self.results_frame.columnconfigure(c, weight=1)

        self.canvas.yview_moveto(0)



# Need ImageTk only after GUI environment is available.
from PIL import ImageTk


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CNN NEURON VISUALIZATION — ALEXNET-STYLE + MNIST")
    print("=" * 70)
    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM    : "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB"
        )
    print(f"Project : {PROJECT_DIR}")
    print(f"Model   : {CHECKPOINT}")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)

    # We need the model before opening the GUI because the GUI
    # should never have a half-initialized model.
    print("Preparing model...")

    model, dataset = load_or_train(
        progress=lambda m: print(m)
    )

    print("Model ready.")
    print("Starting GUI...")

    root = tk.Tk()
    app = App(root, model, dataset)

    root.mainloop()


if __name__ == "__main__":
    main()
