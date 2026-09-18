"""
SPYDER-READY GUI: Visualizing Image Patches That Maximally Activate a CNN Neuron

Run this file directly from Spyder. A Tkinter GUI opens.

Install:
    pip install torch torchvision pillow matplotlib numpy

The first run needs internet access to download VGG16 ImageNet weights and
CIFAR-10. The GUI can also analyze images from a local folder.

Experiment:
    - pretrained VGG16
    - select layer and channel
    - scan images
    - find the maximum spatial activation per image
    - rank images by activation
    - map the activation to the theoretical receptive field
    - display the Top-K maximally activating patches
"""

import os
import math
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, datasets, transforms

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ============================================================
# LOCAL IMAGE DATASET
# ============================================================

class FolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.paths = []

        for base, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    self.paths.append(os.path.join(base, f))
        self.paths.sort()

        if not self.paths:
            raise ValueError("No supported image files were found in the selected folder.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path


# ============================================================
# ACTIVATION HOOK
# ============================================================

class ActivationHook:
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activation = output.detach()

    def close(self):
        self.handle.remove()


# ============================================================
# EXACT THEORETICAL RECEPTIVE FIELD FOR VGG16
# ============================================================

def calculate_receptive_fields(model):
    """Return theoretical receptive field, jump and first-center for VGG16 layers."""
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


def receptive_field_crop(fx, fy, rf_info, image_size=224, fixed_size=None):
    """Map feature-map coordinate to an input crop."""
    center_x = rf_info["start"] + fx * rf_info["jump"]
    center_y = rf_info["start"] + fy * rf_info["jump"]
    size = int(round(rf_info["rf"] if fixed_size is None else fixed_size))
    half = size / 2.0

    left = max(0, int(round(center_x - half)))
    top = max(0, int(round(center_y - half)))
    right = min(image_size, int(round(center_x + half)))
    bottom = min(image_size, int(round(center_y + half)))

    return (left, top, right, bottom), (center_x, center_y)


# ============================================================
# GUI APPLICATION
# ============================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Maximally Activating Neuron Patches - VGG16")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 700)

        self.model = None
        self.rf_info = {}
        self.dataset = None
        self.results = None
        self.layer_map = {}
        self.running = False
        self.last_layer = None
        self.last_channel = None

        self.build_gui()
        self.load_model_async()

    # --------------------------------------------------------
    # GUI CONSTRUCTION
    # --------------------------------------------------------

    def build_gui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="CNN Neuron Visualization", font=("Arial", 16, "bold")).pack(anchor="w")
        ttk.Label(left, text="Maximally activating image patches", foreground="gray").pack(anchor="w", pady=(0, 12))

        ttk.Label(left, text="Dataset").pack(anchor="w")
        self.dataset_var = tk.StringVar(value="CIFAR-10 Test")
        self.dataset_combo = ttk.Combobox(
            left, textvariable=self.dataset_var,
            values=["CIFAR-10 Test", "Image Folder"],
            state="readonly", width=32
        )
        self.dataset_combo.pack(fill="x", pady=(3, 7))
        self.dataset_combo.bind("<<ComboboxSelected>>", self.dataset_changed)

        folder_row = ttk.Frame(left)
        folder_row.pack(fill="x", pady=(0, 10))
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(folder_row, textvariable=self.folder_var, width=25, state="disabled")
        self.folder_entry.pack(side="left", fill="x", expand=True)
        self.browse_btn = ttk.Button(folder_row, text="Browse", command=self.browse_folder, state="disabled")
        self.browse_btn.pack(side="left", padx=(5, 0))

        ttk.Separator(left).pack(fill="x", pady=5)

        ttk.Label(left, text="Model", font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Label(left, text="VGG16 — ImageNet pretrained").pack(anchor="w")
        ttk.Label(
            left, text=f"Device: {DEVICE}",
            foreground="darkgreen" if DEVICE.type == "cuda" else "darkorange"
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(left, text="Layer").pack(anchor="w")
        self.layer_var = tk.StringVar()
        self.layer_combo = ttk.Combobox(left, textvariable=self.layer_var, state="readonly", width=36)
        self.layer_combo.pack(fill="x", pady=(3, 7))
        self.layer_combo.bind("<<ComboboxSelected>>", self.layer_changed)

        ttk.Label(left, text="Neuron / Channel").pack(anchor="w")
        channel_row = ttk.Frame(left)
        channel_row.pack(fill="x", pady=(3, 8))
        self.channel_var = tk.IntVar(value=0)
        self.channel_spin = tk.Spinbox(channel_row, from_=0, to=511, textvariable=self.channel_var, width=10)
        self.channel_spin.pack(side="left")
        self.channel_info = ttk.Label(channel_row, text="")
        self.channel_info.pack(side="left", padx=8)

        ttk.Label(left, text="Images to scan").pack(anchor="w")
        self.images_var = tk.IntVar(value=1000)
        ttk.Spinbox(left, from_=100, to=100000, increment=100, textvariable=self.images_var, width=12).pack(anchor="w", pady=(3, 8))

        ttk.Label(left, text="Top-K patches").pack(anchor="w")
        self.topk_var = tk.IntVar(value=12)
        ttk.Spinbox(left, from_=4, to=40, increment=4, textvariable=self.topk_var, width=12).pack(anchor="w", pady=(3, 8))

        ttk.Label(left, text="Patch mode").pack(anchor="w")
        self.patch_mode_var = tk.StringVar(value="Exact theoretical receptive field")
        ttk.Combobox(
            left, textvariable=self.patch_mode_var,
            values=["Exact theoretical receptive field", "Fixed 96 x 96 crop", "Fixed 128 x 128 crop"],
            state="readonly", width=36
        ).pack(fill="x", pady=(3, 12))

        self.run_btn = ttk.Button(left, text="RUN EXPERIMENT", command=self.run_experiment, state="disabled")
        self.run_btn.pack(fill="x", ipady=7, pady=3)

        self.save_btn = ttk.Button(left, text="SAVE TOP PATCHES", command=self.save_patches, state="disabled")
        self.save_btn.pack(fill="x", ipady=5, pady=3)

        self.csv_btn = ttk.Button(left, text="SAVE RESULTS CSV", command=self.save_csv, state="disabled")
        self.csv_btn.pack(fill="x", ipady=5, pady=3)

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="Experiment information", font=("Arial", 10, "bold")).pack(anchor="w")

        self.info = tk.Text(left, width=40, height=15, wrap="word", state="disabled")
        self.info.pack(fill="both", expand=True, pady=5)

        self.status_var = tk.StringVar(value="Loading VGG16...")
        ttk.Label(left, textvariable=self.status_var, wraplength=330, foreground="navy").pack(anchor="w", pady=(5, 0))

        ttk.Label(right, text="Maximally Activating Image Patches", font=("Arial", 15, "bold")).grid(row=0, column=0, sticky="w")

        plot_frame = ttk.Frame(right)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            right,
            text=("Each patch is centered around the input location corresponding to the "
                  "strongest activation of the selected VGG16 channel. The default mode uses "
                  "the theoretical receptive field of the selected layer."),
            wraplength=1050, foreground="gray"
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.set_info(
            "OBJECTIVE\n\n"
            "Find real image patches that maximally activate a selected CNN neuron/channel.\n\n"
            "METHOD\n\n"
            "Dataset search → activation map → spatial maximum → ranking → receptive-field crop.\n\n"
            "Start with features.28, channel 0, and 1000 images."
        )

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    def load_model_async(self):
        thread = threading.Thread(target=self.load_model_worker, daemon=True)
        thread.start()

    def load_model_worker(self):
        try:
            weights = models.VGG16_Weights.DEFAULT
            model = models.vgg16(weights=weights).to(DEVICE)
            model.eval()
            rf = calculate_receptive_fields(model)
            self.root.after(0, lambda: self.model_loaded(model, rf))
        except Exception:
            error = traceback.format_exc()
            self.root.after(0, lambda: self.model_failed(error))

    def model_loaded(self, model, rf):
        self.model = model
        self.rf_info = rf

        options = []
        self.layer_map.clear()

        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                n = layer.out_channels
                text = f"features.{i} | Conv2d | {n} channels"
                options.append(text)
                self.layer_map[text] = i
            elif isinstance(layer, nn.MaxPool2d):
                n = self.channels_at(i)
                text = f"features.{i} | MaxPool2d | {n} channels"
                options.append(text)
                self.layer_map[text] = i

        self.layer_combo["values"] = options
        preferred = next((x for x in options if x.startswith("features.28")), options[-1])
        self.layer_var.set(preferred)
        self.layer_changed()
        self.run_btn.config(state="normal")
        self.status_var.set(f"VGG16 loaded successfully. Device: {DEVICE}")

    def model_failed(self, error):
        self.status_var.set("VGG16 loading failed.")
        messagebox.showerror(
            "Model loading error",
            "Could not load VGG16.\n\nInstall torch/torchvision and ensure internet is available for the first download.\n\n" + error[-3000:]
        )

    def channels_at(self, index):
        channels = 3
        for i in range(index + 1):
            layer = self.model.features[i]
            if isinstance(layer, nn.Conv2d):
                channels = layer.out_channels
        return channels

    # --------------------------------------------------------
    # GUI INPUTS
    # --------------------------------------------------------

    def dataset_changed(self, event=None):
        folder = self.dataset_var.get() == "Image Folder"
        self.folder_entry.config(state="normal" if folder else "disabled")
        self.browse_btn.config(state="normal" if folder else "disabled")

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.folder_var.set(folder)

    def layer_changed(self, event=None):
        if self.model is None:
            return
        text = self.layer_var.get()
        if text not in self.layer_map:
            return
        idx = self.layer_map[text]
        layer = self.model.features[idx]
        n = layer.out_channels if isinstance(layer, nn.Conv2d) else self.channels_at(idx)
        self.channel_spin.config(from_=0, to=n - 1)
        if self.channel_var.get() >= n:
            self.channel_var.set(n - 1)

        r = self.rf_info[idx]
        self.channel_info.config(text=f"0–{n - 1}")
        self.set_info(
            f"SELECTED LAYER\n\n"
            f"features.{idx}\n"
            f"Type: {type(layer).__name__}\n"
            f"Channels: {n}\n\n"
            f"THEORETICAL RECEPTIVE FIELD\n\n"
            f"{r['rf']:.0f} × {r['rf']:.0f} input pixels\n\n"
            f"Feature-map jump: {r['jump']:.0f} pixels\n"
            f"First-center coordinate: {r['start']:.1f} pixels"
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    def get_dataset(self):
        if self.dataset_var.get() == "CIFAR-10 Test":
            return datasets.CIFAR10(
                root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                train=False,
                download=True,
                transform=TRANSFORM
            )

        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            raise ValueError("Select a valid image folder first.")
        return FolderDataset(folder, TRANSFORM)

    def original_image(self, dataset, index):
        if isinstance(dataset, datasets.CIFAR10):
            image = Image.fromarray(dataset.data[index])
            label = dataset.classes[dataset.targets[index]]
            return image, label

        path = dataset.paths[index]
        image = Image.open(path).convert("RGB")
        relative = os.path.relpath(path, dataset.root)
        parts = relative.split(os.sep)
        label = parts[0] if len(parts) > 1 else ""
        return image, label

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    def read_settings(self):
        text = self.layer_var.get()
        if text not in self.layer_map:
            raise ValueError("Select a layer.")

        layer_idx = self.layer_map[text]
        channel = int(self.channel_var.get())
        n_images = int(self.images_var.get())
        topk = int(self.topk_var.get())

        if n_images < 1:
            raise ValueError("Images to scan must be at least 1.")
        if topk < 1:
            raise ValueError("Top-K must be at least 1.")

        n_channels = self.channels_at(layer_idx)
        if not 0 <= channel < n_channels:
            raise ValueError(f"Channel must be between 0 and {n_channels - 1}.")

        mode = self.patch_mode_var.get()
        fixed = None
        if mode.startswith("Fixed 96"):
            fixed = 96
        elif mode.startswith("Fixed 128"):
            fixed = 128

        return layer_idx, channel, n_images, topk, fixed

    def run_experiment(self):
        if self.running:
            return
        try:
            layer_idx, channel, n_images, topk, fixed = self.read_settings()
            dataset = self.get_dataset()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        n_images = min(n_images, len(dataset))
        topk = min(topk, n_images)
        self.dataset = dataset
        self.running = True
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.csv_btn.config(state="disabled")
        self.status_var.set(f"Scanning {n_images} images...")

        thread = threading.Thread(
            target=self.scan_worker,
            args=(dataset, layer_idx, channel, n_images, topk, fixed),
            daemon=True
        )
        thread.start()

    def scan_worker(self, dataset, layer_idx, channel, n_images, topk, fixed):
        hook = None
        try:
            subset = Subset(dataset, range(n_images))
            batch_size = 32 if DEVICE.type == "cuda" else 8
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda")

            hook = ActivationHook(self.model.features[layer_idx])
            records = []
            processed = 0

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(DEVICE, non_blocking=True)
                    self.model(x)
                    act = hook.activation

                    if act.ndim != 4:
                        raise RuntimeError("Selected layer did not produce a 4-D activation map.")

                    selected = act[:, channel]
                    values, flat = selected.reshape(selected.shape[0], -1).max(dim=1)
                    h, w = selected.shape[-2:]

                    ys = (flat // w).cpu().numpy()
                    xs = (flat % w).cpu().numpy()
                    values = values.cpu().numpy()

                    for j, value in enumerate(values):
                        idx = processed + j
                        source, label = self.source_label(dataset, idx)
                        records.append({
                            "dataset_index": idx,
                            "activation": float(value),
                            "feature_x": int(xs[j]),
                            "feature_y": int(ys[j]),
                            "feature_h": int(h),
                            "feature_w": int(w),
                            "source": source,
                            "label": label,
                        })

                    processed += len(values)
                    if processed % (batch_size * 5) == 0:
                        p = int(100 * processed / n_images)
                        self.root.after(0, lambda p=p: self.status_var.set(f"Scanning images... {p}%"))

            results = pd.DataFrame(records).sort_values("activation", ascending=False).reset_index(drop=True)
            self.root.after(0, lambda: self.finished(results, layer_idx, channel, topk, fixed))
        except Exception:
            error = traceback.format_exc()
            self.root.after(0, lambda: self.failed(error))
        finally:
            if hook is not None:
                hook.close()

    def source_label(self, dataset, index):
        if isinstance(dataset, datasets.CIFAR10):
            label = dataset.classes[dataset.targets[index]]
            return f"CIFAR-10: {label}", label
        path = dataset.paths[index]
        relative = os.path.relpath(path, dataset.root)
        parts = relative.split(os.sep)
        label = parts[0] if len(parts) > 1 else ""
        return path, label

    def finished(self, results, layer_idx, channel, topk, fixed):
        self.running = False
        self.results = results
        self.last_layer = layer_idx
        self.last_channel = channel
        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.csv_btn.config(state="normal")
        self.status_var.set(f"Completed: {len(results)} images scanned; showing top {topk}.")
        self.display_results(results, layer_idx, channel, topk, fixed)

    def failed(self, error):
        self.running = False
        self.run_btn.config(state="normal")
        self.status_var.set("Experiment failed.")
        messagebox.showerror("Experiment error", error[-5000:])

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    def display_results(self, results, layer_idx, channel, topk, fixed):
        self.figure.clear()
        top = results.head(topk)
        ncols = 4
        nrows = math.ceil(len(top) / ncols)
        axes = self.figure.subplots(nrows, ncols, squeeze=False).ravel()
        rf = self.rf_info[layer_idx]

        for rank, (ax, (_, row)) in enumerate(zip(axes, top.iterrows()), 1):
            image, label = self.original_image(self.dataset, int(row["dataset_index"]))
            image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
            box, center = receptive_field_crop(
                int(row["feature_x"]), int(row["feature_y"]), rf, INPUT_SIZE, fixed
            )
            patch = image.crop(box)
            ax.imshow(patch)
            ax.set_title(
                f"Rank {rank}\nActivation = {row['activation']:.3f}\n{label}",
                fontsize=9
            )
            ax.axis("off")

        for ax in axes[len(top):]:
            ax.axis("off")

        self.figure.suptitle(
            f"VGG16 features.{layer_idx} | Channel {channel} | Top {topk}",
            fontsize=15
        )
        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

        top_mean = top["activation"].mean()
        all_mean = results["activation"].mean()
        ratio = top_mean / (all_mean + 1e-12)

        self.set_info(
            f"RESULTS\n\n"
            f"Layer: features.{layer_idx}\n"
            f"Channel: {channel}\n"
            f"Images scanned: {len(results)}\n"
            f"Top-K: {topk}\n\n"
            f"Theoretical RF: {rf['rf']:.0f} × {rf['rf']:.0f}\n\n"
            f"Mean activation: {all_mean:.4f}\n"
            f"Top-{topk} mean: {top_mean:.4f}\n"
            f"Top-K / all mean: {ratio:.3f}\n\n"
            f"INTERPRETATION\n\n"
            f"Inspect whether the top patches share a consistent visual pattern."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_patches(self):
        if self.results is None:
            return
        folder = filedialog.askdirectory(title="Select folder to save top patches")
        if not folder:
            return

        try:
            rf = self.rf_info[self.last_layer]
            count = 0
            for rank, (_, row) in enumerate(self.results.head(50).iterrows(), 1):
                image, _ = self.original_image(self.dataset, int(row["dataset_index"]))
                image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
                box, _ = receptive_field_crop(
                    int(row["feature_x"]), int(row["feature_y"]), rf, INPUT_SIZE, None
                )
                patch = image.crop(box)
                name = f"rank_{rank:02d}_activation_{row['activation']:.4f}.png"
                patch.save(os.path.join(folder, name))
                count += 1
            messagebox.showinfo("Saved", f"Saved {count} patches to:\n{folder}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def save_csv(self):
        if self.results is None:
            return
        filename = filedialog.asksaveasfilename(
            title="Save activation results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            self.results.to_csv(filename, index=False)
            messagebox.showinfo("Saved", f"Results saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def set_info(self, text):
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.config(state="disabled")


def main():
    print("=" * 70)
    print("CNN MAXIMALLY ACTIVATING PATCH VISUALIZATION")
    print("=" * 70)
    print("Device:", DEVICE)
    print("Starting GUI...")

    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
