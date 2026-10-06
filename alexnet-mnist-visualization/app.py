import streamlit as st
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AlexNet CNN Visualization",
    page_icon="🧠",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CHECKPOINT = APP_DIR / "alexnet_mnist.pt"

DEVICE = torch.device("cpu")

BATCH_SIZE = 64
DEFAULT_SCAN = 5000
DEFAULT_TOPK = 8

# ============================================================
# MODEL
# ============================================================

class AlexNetMNIST(nn.Module):
    """AlexNet-style CNN adapted for 28x28 MNIST."""

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
# MODEL / DATA LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = AlexNetMNIST().to(DEVICE)

    payload = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(payload["model_state"])
    model.eval()

    accuracy = payload.get("test_accuracy", None)

    return model, accuracy


@st.cache_data
def load_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )


# ============================================================
# LAYER INFORMATION
# ============================================================

LAYER_INFO = {
    "conv1": {
        "channels": 64,
        "description": "Early layer: edges, stroke directions, simple curves.",
    },
    "conv2": {
        "channels": 128,
        "description": "Mid-early layer: combinations of strokes and corners.",
    },
    "conv3": {
        "channels": 256,
        "description": "Middle layer: more structured digit parts.",
    },
    "conv4": {
        "channels": 256,
        "description": "Deep layer: larger combinations of digit structure.",
    },
    "conv5": {
        "channels": 256,
        "description": "Deepest convolution: high-level digit patterns.",
    },
}

RF_INFO = {
    "conv1": (3, 1),
    "conv2": (8, 2),
    "conv3": (18, 4),
    "conv4": (38, 8),
    "conv5": (78, 16),
}

START_INFO = {
    "conv1": 0.5,
    "conv2": 1.0,
    "conv3": 2.0,
    "conv4": 4.0,
    "conv5": 8.0,
}


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def tensor_to_uint8(tensor):
    x = tensor.detach().cpu().squeeze().numpy()
    x = x * 0.3081 + 0.1307
    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def crop_rf(img28, fmap_y, fmap_x, layer):
    rf, jump = RF_INFO[layer]
    start = START_INFO[layer]

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


def make_original_image(img28, box, point, size=240):
    image = Image.fromarray(img28, mode="L").resize(
        (size, size),
        Image.Resampling.NEAREST,
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    x0, y0, x1, y1 = box
    scale = size / 28.0

    draw.rectangle(
        [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
        outline=(255, 0, 0),
        width=4,
    )

    px, py = point
    cx = px * scale
    cy = py * scale
    radius = 5

    draw.ellipse(
        [
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
        ],
        fill=(255, 255, 0),
        outline=(0, 0, 0),
        width=2,
    )

    return image


def make_crop_image(crop, size=240):
    return Image.fromarray(
        crop,
        mode="L",
    ).resize(
        (size, size),
        Image.Resampling.NEAREST,
    )


# ============================================================
# TOP-K SCAN
# ============================================================

@torch.inference_mode()
def scan_topk(model, dataset, layer, channel, scan_n, topk, activation_mode):
    scan_n = min(scan_n, len(dataset))
    topk = min(topk, scan_n)

    subset = Subset(dataset, range(scan_n))

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    mode_key = {
        "POST-RELU": "post",
        "PRE-RELU": "pre",
    }[activation_mode]

    key = f"{layer}_{mode_key}"

    results = []
    done = 0

    progress = st.progress(0, text="Scanning MNIST images...")

    for images, labels in loader:
        images = images.to(DEVICE)

        _, feats = model(
            images,
            return_features=True,
        )

        fmap = feats[key][:, channel, :, :]

        scores, flat_idx = torch.max(
            fmap.reshape(fmap.size(0), -1),
            dim=1,
        )

        h, w = fmap.shape[-2:]

        ys = torch.div(
            flat_idx,
            w,
            rounding_mode="floor",
        )
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

        progress.progress(
            min(done / scan_n, 1.0),
            text=f"Scanning MNIST images: {done}/{scan_n}",
        )

    progress.empty()

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:topk]


# ============================================================
# PAGE
# ============================================================

st.title("🧠 AlexNet CNN Visualization")
st.caption(
    "Interactive visualization of convolutional filters and "
    "maximally activating MNIST images."
)

try:
    model, stored_accuracy = load_model()
    dataset = load_test_dataset()
except Exception as exc:
    st.error("The model or MNIST dataset could not be loaded.")
    st.exception(exc)
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Experiment Controls")

    st.write("**Dataset:** MNIST test set (10,000 images)")
    st.write("**Network:** AlexNet-style CNN trained on MNIST")

    if stored_accuracy is not None:
        st.metric(
            "Stored test accuracy",
            f"{stored_accuracy:.2f}%",
        )

    layer = st.selectbox(
        "Convolutional layer",
        list(LAYER_INFO.keys()),
        index=0,
    )

    max_channels = LAYER_INFO[layer]["channels"]

    channel = st.number_input(
        "Filter / channel",
        min_value=0,
        max_value=max_channels - 1,
        value=0,
        step=1,
    )

    st.caption(
        f"Channel {channel} of {max_channels - 1}"
    )

    st.info(
        LAYER_INFO[layer]["description"]
    )

    activation_mode = st.selectbox(
        "Activation used for ranking",
        ["POST-RELU", "PRE-RELU"],
        index=0,
    )

    scan_n = st.number_input(
        "Images to scan",
        min_value=100,
        max_value=len(dataset),
        value=min(DEFAULT_SCAN, len(dataset)),
        step=100,
    )

    topk = st.number_input(
        "Top-K",
        min_value=1,
        max_value=12,
        value=DEFAULT_TOPK,
        step=1,
    )

    run = st.button(
        "▶ RUN TOP-K VISUALIZATION",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.subheader("What students should learn")

    st.markdown(
        """
1. One filter is shared across every image.

2. For each image, measure the **maximum activation**
   of that filter.

3. **Top-K** = images producing the strongest responses.

4. The yellow dot marks the strongest spatial neuron.

5. The red box shows its theoretical receptive field.

6. Different digits can activate the same filter because
   the filter detects a visual pattern, not necessarily a
   class label.

7. Compare layers: early filters usually detect simple
   strokes; deeper filters combine larger structures.
        """
    )

# Main information
rf, jump = RF_INFO[layer]

st.subheader("Selected filter")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Layer", layer)

with col2:
    st.metric("Filter", channel)

with col3:
    st.metric("Activation", activation_mode)

with col4:
    st.metric("Theoretical RF", f"{rf} × {rf} px")

if rf > 28:
    st.caption("RF exceeds 28×28 input; display is clipped.")

if rf > 28:
    st.info(
        f"{layer} has a theoretical receptive field of {rf}×{rf} pixels, "
        "which is larger than the 28×28 MNIST input. "
        "Therefore, the red RF box is clipped at the image boundaries."
    )

st.write(
    f"**Ranking rule:** for every image, take the maximum "
    f"value over all spatial positions of `{layer}` filter "
    f"{channel}. The Top-{topk} images have the largest "
    f"maximum responses."
)

if run:
    results = scan_topk(
        model,
        dataset,
        layer,
        int(channel),
        int(scan_n),
        int(topk),
        activation_mode,
    )

    st.success(
        f"Complete — Top-{len(results)} strongest responses displayed."
    )

    st.divider()

    for start in range(0, len(results), 4):
        row = results[start:start + 4]
        columns = st.columns(4)

        for position, item in enumerate(row):
            rank = start + position + 1

            with columns[position]:
                image_tensor = dataset[item["dataset_index"]][0]
                img28 = tensor_to_uint8(image_tensor)

                crop, box, point = crop_rf(
                    img28,
                    item["fy"],
                    item["fx"],
                    layer,
                )

                original = make_original_image(
                    img28,
                    box,
                    point,
                    size=240,
                )

                crop_image = make_crop_image(
                    crop,
                    size=240,
                )

                st.markdown(
                    f"### #{rank} — digit {item['label']}"
                )

                st.write(
                    f"**Activation:** `{item['score']:.4f}`"
                )

                st.image(
                    original,
                    caption="Original + RF box + strongest point",
                    width="stretch",
                )

                st.image(
                    crop_image,
                    caption="Exact theoretical RF crop",
                    width="stretch",
                )

                st.caption(
                    f"Feature-map location: "
                    f"({item['fy']}, {item['fx']})  \n"
                    f"RF = {rf}×{rf} px • jump = {jump}"
                )
else:
    st.info(
        "Select a convolutional layer and filter, then click "
        "**RUN TOP-K VISUALIZATION**."
    )

st.divider()

with st.expander("Interpretation guide"):
    st.markdown(
        """
**POST-RELU:** ranks images using the activation after ReLU.
Negative convolution responses are therefore removed.

**PRE-RELU:** ranks images using the raw convolution output
before ReLU.

**Top-K:** for each image, the selected filter produces a
feature map. We take the maximum spatial value in that map.
Images are then ranked by this maximum.

**Yellow point:** location of the maximum activation in the
selected feature map.

**Red box:** theoretical receptive field of that feature-map
position mapped back to the original 28×28 MNIST image.

**Important:** the receptive field is a theoretical architectural
quantity; it does not mean every pixel inside the box contributes
equally to the activation.
        """
    )
