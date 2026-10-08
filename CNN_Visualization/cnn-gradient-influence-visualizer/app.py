import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.transforms.functional import to_tensor, normalize

st.set_page_config(
    page_title="CNN Input-Pixel Influence",
    layout="centered"
)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    return model, weights


model, weights = load_model()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------
def prepare_image(image, max_dimension=400):
    """
    Resize while preserving the original aspect ratio.

    We intentionally do NOT use the standard 224x224 center crop here.
    The reason is pedagogical: the influence map should have exactly
    the same height/width ratio as the image displayed to the user.
    AlexNet's AdaptiveAvgPool2d allows variable spatial input sizes.
    """
    image = image.convert("RGB")

    w, h = image.size
    scale = min(1.0, max_dimension / max(w, h))

    if scale < 1.0:
        image = image.resize(
            (max(1, int(round(w * scale))),
             max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS
        )

    return image


def make_input_tensor(image):
    """
    Convert the displayed image directly into the model input while
    preserving its aspect ratio.
    """
    tensor = to_tensor(image)
    tensor = normalize(
        tensor,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------
# Forward representations
# ---------------------------------------------------------------------
CONV_INFO = [
    (0, "Conv1", 64),
    (3, "Conv2", 192),
    (6, "Conv3", 384),
    (8, "Conv4", 256),
    (10, "Conv5", 256),
]


def forward_representations(x):
    """
    Return:
      conv_outputs: post-ReLU outputs of Conv1...Conv5
      fc1_pre: FC1 pre-activation (4096)
      fc1_post: FC1 post-ReLU activation (4096)
      fc2_pre: FC2 pre-activation (4096)
      fc2_post: FC2 post-ReLU activation (4096)
      logits: final class logits (1000)
    """
    conv_outputs = {}

    y = x

    # Conv/ReLU/Pool feature extractor.
    for idx, layer in enumerate(model.features):
        y = layer(y)

        if idx == 1:
            conv_outputs["Conv1"] = y
        elif idx == 4:
            conv_outputs["Conv2"] = y
        elif idx == 7:
            conv_outputs["Conv3"] = y
        elif idx == 9:
            conv_outputs["Conv4"] = y
        elif idx == 11:
            conv_outputs["Conv5"] = y

    pooled = model.avgpool(y)
    flattened = torch.flatten(pooled, 1)

    fc1_pre = model.classifier[1](flattened)
    fc1_post = model.classifier[2](fc1_pre)

    fc2_pre = model.classifier[4](fc1_post)
    fc2_post = model.classifier[5](fc2_pre)

    logits = model.classifier[6](fc2_post)

    return (
        conv_outputs,
        fc1_pre,
        fc1_post,
        fc2_pre,
        fc2_post,
        logits,
    )


# ---------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------
TARGET_OPTIONS = [
    "FC1 neuron — recommended ( hidden neuron)",
    "FC2 neuron",
    "Conv1 neuron",
    "Conv2 neuron",
    "Conv3 neuron",
    "Conv4 neuron",
    "Conv5 neuron",
    "Output class logit",
]


def target_description(target_type):
    if target_type.startswith("FC1"):
        return "FC1 pre-activation hⱼ"
    if target_type.startswith("FC2"):
        return "FC2 pre-activation hⱼ"
    if target_type.startswith("Conv"):
        return target_type.split(" ")[0] + " neuron"
    return "Output class logit"


def get_target(
    x,
    target_type,
    index_info
):
    """
    Return the scalar target whose gradient is computed.

    Crucially, this function uses EXACTLY the neuron/class selected
    by the user. There is no hidden 'strongest neuron' override.
    """
    (
        conv_outputs,
        fc1_pre,
        fc1_post,
        fc2_pre,
        fc2_post,
        logits,
    ) = forward_representations(x)

    if target_type.startswith("FC1"):
        return fc1_pre[0, index_info["neuron"]]

    if target_type.startswith("FC2"):
        return fc2_pre[0, index_info["neuron"]]

    if target_type.startswith("Conv"):
        layer_name = target_type.split(" ")[0]
        activation = conv_outputs[layer_name]

        return activation[
            0,
            index_info["channel"],
            index_info["row"],
            index_info["col"],
        ]

    return logits[0, index_info["class_index"]]


# ---------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------
def compute_gradient(image, target_type, index_info):
    x = make_input_tensor(image)
    x.requires_grad_(True)

    target = get_target(
        x,
        target_type,
        index_info
    )

    gradient = torch.autograd.grad(
        outputs=target,
        inputs=x,
        retain_graph=False,
        create_graph=False,
    )[0]

    gradient = gradient.detach()[0].cpu().numpy()

    # C,H,W -> H,W,C
    gradient_rgb = np.transpose(gradient, (1, 2, 0))

    return (
        gradient_rgb,
        float(target.detach().cpu())
    )


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def signed_gradient_image(gradient_rgb):
    """
    Lecture-style signed gradient visualization.

    Zero gradient -> neutral gray.
    Positive/negative channel gradients -> color variation.
    """
    absolute = np.abs(gradient_rgb)

    nonzero = absolute[absolute > 1e-12]

    if nonzero.size == 0:
        return np.full(
            (*gradient_rgb.shape[:2], 3),
            128,
            dtype=np.uint8
        )

    scale = float(np.percentile(nonzero, 99))

    if scale <= 0:
        scale = float(nonzero.max())

    normalized = np.clip(
        gradient_rgb / scale,
        -1.0,
        1.0
    )

    # Map [-1, 1] -> [0, 1], so zero becomes gray.
    display = (normalized * 0.5) + 0.5

    return np.clip(
        display * 255.0,
        0,
        255
    ).astype(np.uint8)


def magnitude_image(gradient_rgb):
    magnitude = np.max(
        np.abs(gradient_rgb),
        axis=2
    )

    nonzero = magnitude[magnitude > 1e-12]

    if nonzero.size == 0:
        normalized = np.zeros_like(magnitude)
    else:
        scale = float(np.percentile(nonzero, 99))

        if scale <= 0:
            scale = float(nonzero.max())

        normalized = np.clip(
            magnitude / scale,
            0.0,
            1.0
        )

    gray = np.uint8(normalized * 255.0)

    return np.stack(
        [gray, gray, gray],
        axis=2
    )


def gradient_stats(gradient_rgb):
    magnitude = np.max(np.abs(gradient_rgb), axis=2)

    return {
        "max": float(magnitude.max()),
        "mean": float(magnitude.mean()),
        "nonzero_pixels": int(np.count_nonzero(magnitude > 1e-12)),
        "total_pixels": int(magnitude.size),
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("CNN Input-Pixel Influence")

st.caption(
    r"Direct backpropagation: "
    r"$\partial h_j / \partial x_i$"
)

st.info(
    "The image is fixed. You select a target neuron, and the app "
    "backpropagates that exact neuron's activation to the input pixels."
)


uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.stop()

image = prepare_image(
    Image.open(uploaded)
)


# ---------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------
st.subheader("1. Select target")

target_type = st.selectbox(
    "Target layer / output",
    TARGET_OPTIONS,
    index=0
)


# Probe the network once for dimensions and activations.
x_probe = make_input_tensor(image)

with torch.no_grad():
    (
        conv_probe,
        fc1_pre_probe,
        fc1_post_probe,
        fc2_pre_probe,
        fc2_post_probe,
        logits_probe,
    ) = forward_representations(x_probe)


index_info = {}


# ---------------------------------------------------------------------
# FC1 / FC2
# ---------------------------------------------------------------------
if target_type.startswith("FC1") or target_type.startswith("FC2"):

    if target_type.startswith("FC1"):
        post_activation = fc1_post_probe[0].cpu().numpy()
        num_neurons = 4096
    else:
        post_activation = fc2_post_probe[0].cpu().numpy()
        num_neurons = 4096

    selection_mode = st.radio(
        "Neuron selection",
        ["Strongest active neuron", "Choose neuron manually"],
        horizontal=True,
        index=0
    )

    if selection_mode == "Strongest active neuron":
        neuron_index = int(
            np.argmax(post_activation)
        )

        st.caption(
            f"Selected h{neuron_index + 1}: "
            f"strongest positive activation in this layer."
        )

    else:
        neuron_index = st.number_input(
            "Neuron hⱼ",
            min_value=1,
            max_value=num_neurons,
            value=1,
            step=1,
            help="Enter the neuron number. h1 corresponds to index 0."
        )

        neuron_index = int(neuron_index) - 1

        st.caption(
            f"Selected h{neuron_index + 1}"
        )

    index_info["neuron"] = neuron_index


# ---------------------------------------------------------------------
# Convolutional layers
# ---------------------------------------------------------------------
elif target_type.startswith("Conv"):

    layer_name = target_type.split(" ")[0]
    activation = conv_probe[layer_name]

    _, channels, height, width = activation.shape

    selection_mode = st.radio(
        "Neuron selection",
        ["Strongest active neuron", "Choose neuron manually"],
        horizontal=True,
        index=0
    )

    if selection_mode == "Strongest active neuron":

        flat_index = int(
            torch.argmax(
                activation[0]
            ).item()
        )

        channel = flat_index // (height * width)
        remainder = flat_index % (height * width)
        row = remainder // width
        col = remainder % width

        st.caption(
            f"Selected {layer_name} neuron: "
            f"channel {channel}, position ({row}, {col})"
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:
            channel = st.number_input(
                "Channel",
                min_value=0,
                max_value=channels - 1,
                value=0,
                step=1,
            )

        with c2:
            row = st.number_input(
                "Row",
                min_value=0,
                max_value=height - 1,
                value=height // 2,
                step=1,
            )

        with c3:
            col = st.number_input(
                "Column",
                min_value=0,
                max_value=width - 1,
                value=width // 2,
                step=1,
            )

        channel = int(channel)
        row = int(row)
        col = int(col)

        st.caption(
            f"Selected {layer_name} neuron: "
            f"channel {channel}, position ({row}, {col})"
        )

    index_info["channel"] = int(channel)
    index_info["row"] = int(row)
    index_info["col"] = int(col)


# ---------------------------------------------------------------------
# Output class
# ---------------------------------------------------------------------
else:

    probabilities = torch.softmax(
        logits_probe[0],
        dim=0
    )

    top_probs, top_indices = torch.topk(
        probabilities,
        5
    )

    class_options = [
        (
            int(index),
            f"{weights.meta['categories'][int(index)]} "
            f"— {float(prob) * 100:.2f}%"
        )
        for prob, index in zip(
            top_probs,
            top_indices
        )
    ]

    selected = st.selectbox(
        "Target output class",
        class_options,
        format_func=lambda item: item[1]
    )

    index_info["class_index"] = selected[0]

    st.caption(
        f"Selected class: "
        f"{weights.meta['categories'][selected[0]]}"
    )


# ---------------------------------------------------------------------
# Compute exact gradient
# ---------------------------------------------------------------------
gradient_rgb, target_value = compute_gradient(
    image,
    target_type,
    index_info
)

stats = gradient_stats(gradient_rgb)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
st.divider()

st.subheader("2. Input-pixel influence")

visualization = st.radio(
    "Visualization",
    ["Signed gradient", "Gradient magnitude"],
    horizontal=True,
    index=0
)

if visualization.startswith("Signed"):
    influence_image = signed_gradient_image(
        gradient_rgb
    )
else:
    influence_image = magnitude_image(
        gradient_rgb
    )

left, right = st.columns(2)

with left:
    st.markdown("**Original image**")
    st.image(
        image,
        width=300
    )

with right:
    st.markdown("**Input-pixel influence**")
    st.image(
        influence_image,
        width=300
    )

st.caption(
    r"Computed for the exact selected target using "
    r"$\partial h_j/\partial x_i$."
)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
with st.expander("Technical details"):

    st.write(
        f"Target: {target_description(target_type)}"
    )

    st.write(
        f"Target value: {target_value:.6e}"
    )

    st.write(
        f"Gradient maximum magnitude: "
        f"{stats['max']:.6e}"
    )

    st.write(
        f"Mean gradient magnitude: "
        f"{stats['mean']:.6e}"
    )

    st.write(
        f"Non-zero pixels: "
        f"{stats['nonzero_pixels']} / "
        f"{stats['total_pixels']}"
    )

    st.write(
        f"Displayed image size: "
        f"{image.width} × {image.height}"
    )


# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------
st.divider()

st.subheader("Interpretation")

st.markdown(
    """
For a selected hidden neuron \(h_j\), the application computes the
gradient of that neuron with respect to the input pixels \(x_i\).

- **Large magnitude:** a small change in that pixel can strongly
  affect the selected neuron.
- **Small magnitude:** the neuron is locally less sensitive to that
  pixel.
- The **signed-gradient** view preserves positive/negative channel
  information.

### Why the target layer matters

A convolutional neuron has a spatial receptive field, so its influence
is localized.

A fully connected neuron after flattening can depend on the entire
feature representation, so its input influence can extend across the
image.

The image itself is never occluded or modified.
"""
)

st.latex(r"\frac{\partial h_j}{\partial x_i}")


st.caption(
    "Pretrained AlexNet • ImageNet-1K • "
    "aspect-ratio-preserving visualization input"
)
