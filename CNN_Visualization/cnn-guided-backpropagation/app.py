import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.transforms.functional import to_tensor, normalize

st.set_page_config(
    page_title="CNN Guided Backpropagation",
    layout="centered",
)

# ============================================================
# Guided ReLU
# ============================================================
class GuidedReLUFunction(torch.autograd.Function):
    """
    Guided backpropagation through ReLU.

    Forward:
        y = max(0, x)

    Backward:
        gradient passes only when:
          1. the forward activation is positive, and
          2. the incoming gradient is positive.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        forward_gate = x > 0
        backward_gate = grad_output > 0

        grad_input = (
            grad_output
            * forward_gate
            * backward_gate
        )

        return grad_input


class GuidedReLU(nn.Module):
    def forward(self, x):
        return GuidedReLUFunction.apply(x)


def replace_relu_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedReLU())
        else:
            replace_relu_modules(child)


# ============================================================
# Models
# ============================================================
@st.cache_resource
def load_models():
    weights = AlexNet_Weights.DEFAULT

    standard = alexnet(weights=weights)
    guided = alexnet(weights=weights)

    standard.eval()
    guided.eval()

    replace_relu_modules(guided)

    return standard, guided, weights


standard_model, guided_model, weights = load_models()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# Image handling
# ============================================================
def prepare_image(image, max_dimension=360):
    image = image.convert("RGB")

    w, h = image.size
    scale = min(1.0, max_dimension / max(w, h))

    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    return image


def make_input_tensor(image):
    x = to_tensor(image)
    x = normalize(x, mean=MEAN, std=STD)
    return x.unsqueeze(0)


# ============================================================
# AlexNet target definitions
# ============================================================
CONV_LAYERS = [
    (1, "Conv1", 64),
    (4, "Conv2", 192),
    (7, "Conv3", 384),
    (9, "Conv4", 256),
    (11, "Conv5", 256),
]

FC_LAYERS = [
    (1, "FC1", 4096),
    (4, "FC2", 4096),
]


# ============================================================
# Generic forward capture
# ============================================================
def capture_module_output(model, module):
    captured = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = module.register_forward_hook(hook)

    return captured, handle


def get_conv_activation(model, x, feature_index):
    captured, handle = capture_module_output(
        model,
        model.features[feature_index],
    )

    model(x)

    handle.remove()

    return captured["output"]


def get_fc_activation(model, x, classifier_index):
    captured, handle = capture_module_output(
        model,
        model.classifier[classifier_index],
    )

    model(x)

    handle.remove()

    return captured["output"]


# ============================================================
# ImageNet output prediction
# ============================================================
def get_output_logits(model, x):
    return model(x)


def class_name(categories, index):
    if 0 <= index < len(categories):
        return categories[index]
    return f"Class {index}"


# ============================================================
# Gradient computation
# ============================================================
def compute_gradient(
    model,
    image,
    target_kind,
    target_index,
    channel=None,
    row=None,
    col=None,
    class_index=None,
):
    """
    Returns:
        input gradient: [3,H,W]
        target scalar
    """

    x = make_input_tensor(image)
    x.requires_grad_(True)

    if target_kind == "Convolutional feature-map neuron":
        activation = get_conv_activation(
            model,
            x,
            target_index,
        )

        target = activation[
            0,
            channel,
            row,
            col,
        ]

    elif target_kind == "Fully connected neuron":
        # target_index is the classifier index:
        # 1 = FC1, 4 = FC2.
        activation = get_fc_activation(
            model,
            x,
            target_index,
        )

        # These are the linear outputs immediately before the
        # following ReLU. Using this scalar avoids the trivial
        # zero-gradient problem for a manually selected inactive
        # post-ReLU FC neuron.
        target = activation[0, channel]

    else:
        logits = get_output_logits(model, x)
        target = logits[0, class_index]

    gradient = torch.autograd.grad(
        outputs=target,
        inputs=x,
        retain_graph=False,
        create_graph=False,
    )[0]

    return (
        gradient.detach()[0].cpu().numpy(),
        float(target.detach().cpu()),
    )


# ============================================================
# Visualization
# ============================================================
def gradient_to_display(gradient_chw, mode):
    """
    Convert [C,H,W] gradient to a displayable RGB image.

    Signed gradient:
        preserves the sign; zero is neutral gray.

    Gradient magnitude:
        displays max absolute channel gradient as grayscale.
    """
    gradient = np.transpose(
        gradient_chw,
        (1, 2, 0),
    )

    if mode == "Signed gradient":
        values = np.abs(gradient)
        nonzero = values[values > 1e-12]

        if nonzero.size == 0:
            normalized = np.zeros_like(gradient)
        else:
            scale = float(np.percentile(nonzero, 99))

            if scale <= 1e-12:
                scale = float(nonzero.max())

            normalized = np.clip(
                gradient / scale,
                -1.0,
                1.0,
            )

        # Negative -> darker, zero -> gray, positive -> lighter.
        display = normalized * 0.5 + 0.5

    else:
        magnitude = np.max(
            np.abs(gradient),
            axis=2,
        )

        nonzero = magnitude[magnitude > 1e-12]

        if nonzero.size == 0:
            normalized = np.zeros_like(magnitude)
        else:
            scale = float(np.percentile(nonzero, 99))

            if scale <= 1e-12:
                scale = float(nonzero.max())

            normalized = np.clip(
                magnitude / scale,
                0.0,
                1.0,
            )

        display = np.stack(
            [normalized, normalized, normalized],
            axis=2,
        )

    return np.clip(
        display * 255.0,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# UI
# ============================================================
st.title("CNN Guided Backpropagation")

st.caption(
    "Compare ordinary backpropagation with guided backpropagation "
    "for a selected CNN neuron or output class."
)

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.stop()

image = prepare_image(Image.open(uploaded))

# Probe tensor used only to determine feature-map dimensions
x_probe = make_input_tensor(image)

st.subheader("1. Select the target")

target_type = st.selectbox(
    "Target",
    [
        "Hidden neuron",
        "Output class",
    ],
)

if target_type == "Hidden neuron":

    # A single layer selector intentionally contains BOTH convolutional
    # and fully-connected hidden layers.
    layer_options = [
        "Conv1 • 64 channels",
        "Conv2 • 192 channels",
        "Conv3 • 384 channels",
        "Conv4 • 256 channels",
        "Conv5 • 256 channels",
        "FC1 • 4096 neurons",
        "FC2 • 4096 neurons",
    ]

    selected_layer = st.selectbox(
        "Layer",
        layer_options,
        index=2,
    )

    if selected_layer.startswith("Conv"):
        layer_number = int(selected_layer[4])

        feature_index, layer_name, channels_expected = next(
            item for item in CONV_LAYERS
            if item[1] == f"Conv{layer_number}"
        )

        with torch.no_grad():
            activation_probe = get_conv_activation(
                standard_model,
                x_probe,
                feature_index,
            )

        _, channels, height, width = activation_probe.shape

        selection_mode = st.radio(
            "Neuron selection",
            [
                "Strongest active neuron",
                "Choose manually",
            ],
            horizontal=True,
        )

        if selection_mode == "Strongest active neuron":
            flat_index = int(
                torch.argmax(activation_probe[0]).item()
            )
            channel = flat_index // (height * width)
            remainder = flat_index % (height * width)
            row = remainder // width
            col = remainder % width

        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                channel = int(st.number_input(
                    "Channel",
                    min_value=0,
                    max_value=channels - 1,
                    value=0,
                    step=1,
                ))

            with c2:
                row = int(st.number_input(
                    "Row",
                    min_value=0,
                    max_value=height - 1,
                    value=height // 2,
                    step=1,
                ))

            with c3:
                col = int(st.number_input(
                    "Column",
                    min_value=0,
                    max_value=width - 1,
                    value=width // 2,
                    step=1,
                ))

        st.caption(
            f"Selected {layer_name}: channel {channel}, "
            f"position ({row}, {col})"
        )

        target_kind = "Convolutional feature-map neuron"
        conv_target_index = feature_index
        fc_target_index = None
        class_index = None

    else:
        # Fully connected hidden neuron
        fc_name = "FC1" if selected_layer.startswith("FC1") else "FC2"

        classifier_index, layer_name, num_neurons = next(
            item for item in FC_LAYERS
            if item[1] == fc_name
        )

        with torch.no_grad():
            fc_probe = get_fc_activation(
                standard_model,
                x_probe,
                classifier_index,
            )

        selection_mode = st.radio(
            "Neuron selection",
            [
                "Strongest active neuron",
                "Choose manually",
            ],
            horizontal=True,
        )

        if selection_mode == "Strongest active neuron":
            positive = fc_probe[0].clone()
            positive[positive <= 0] = -torch.inf

            if torch.isfinite(positive).any():
                neuron = int(torch.argmax(positive).item())
            else:
                neuron = int(torch.argmax(fc_probe[0]).item())
        else:
            neuron = int(st.number_input(
                "Neuron",
                min_value=0,
                max_value=num_neurons - 1,
                value=0,
                step=1,
            ))

        st.caption(
            f"Selected {layer_name}: neuron {neuron}"
        )

        target_kind = "Fully connected neuron"
        conv_target_index = None
        fc_target_index = classifier_index
        channel = neuron
        row = None
        col = None
        class_index = None

else:
    categories = weights.meta["categories"]

    with torch.no_grad():
        logits_probe = get_output_logits(
            standard_model,
            x_probe,
        )[0]

    probabilities = torch.softmax(logits_probe, dim=0)
    top_class = int(torch.argmax(probabilities).item())

    class_labels = [
        f"{i}: {categories[i]}"
        for i in range(len(categories))
    ]

    selected_class_label = st.selectbox(
        "ImageNet class",
        class_labels,
        index=top_class,
    )

    class_index = int(
        selected_class_label.split(":", 1)[0]
    )

    probability = float(probabilities[class_index].item())

    st.caption(
        f"Selected class: {categories[class_index]} "
        f"• current probability: {probability:.2%}"
    )

    target_kind = "Output class"
    conv_target_index = None
    fc_target_index = None
    channel = None
    row = None
    col = None


# ============================================================
# Visualization controls
# ============================================================
st.subheader("2. Gradient visualization")

visualization_mode = st.radio(
    "Visualization",
    [
        "Signed gradient",
        "Gradient magnitude",
    ],
    horizontal=True,
)

# ============================================================
# Compute both gradients
# ============================================================
if target_kind == "Convolutional feature-map neuron":

    standard_gradient, standard_target = compute_gradient(
        standard_model,
        image,
        target_kind,
        conv_target_index,
        channel=channel,
        row=row,
        col=col,
    )

    guided_gradient, guided_target = compute_gradient(
        guided_model,
        image,
        target_kind,
        conv_target_index,
        channel=channel,
        row=row,
        col=col,
    )

    target_description = (
        f"{layer_name}, channel {channel}, "
        f"position ({row}, {col})"
    )

elif target_kind == "Fully connected neuron":

    standard_gradient, standard_target = compute_gradient(
        standard_model,
        image,
        target_kind,
        fc_target_index,
        channel=channel,
    )

    guided_gradient, guided_target = compute_gradient(
        guided_model,
        image,
        target_kind,
        fc_target_index,
        channel=channel,
    )

    target_description = (
        f"{layer_name}, neuron {channel}"
    )

else:

    standard_gradient, standard_target = compute_gradient(
        standard_model,
        image,
        target_kind,
        target_index=None,
        class_index=class_index,
    )

    guided_gradient, guided_target = compute_gradient(
        guided_model,
        image,
        target_kind,
        target_index=None,
        class_index=class_index,
    )

    target_description = (
        f"Output logit, class {class_index}: "
        f"{weights.meta['categories'][class_index]}"
    )


# ============================================================
# Main comparison
# ============================================================
st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Original**")
    st.image(
        image,
        width=210,
    )

with c2:
    st.markdown("**Backpropagation**")
    st.image(
        gradient_to_display(
            standard_gradient,
            visualization_mode,
        ),
        width=210,
    )

with c3:
    st.markdown("**Guided backpropagation**")
    st.image(
        gradient_to_display(
            guided_gradient,
            visualization_mode,
        ),
        width=210,
    )

st.caption(
    "Guided backpropagation suppresses negative incoming gradients "
    "at ReLU units during the backward pass."
)


# ============================================================
# Details
# ============================================================
with st.expander("Technical details"):

    st.write(f"Target: {target_description}")

    st.write(
        f"Standard target value: {standard_target:.6e}"
    )

    st.write(
        f"Guided target value: {guided_target:.6e}"
    )

    st.write(
        f"Input size: {image.width} × {image.height}"
    )


# ============================================================
# Explanation
# ============================================================
st.divider()

st.subheader("How guided backpropagation works")

st.markdown(
    """
### Ordinary backpropagation

The gradient is propagated through the network using the usual
chain rule and the ReLU derivative.

### Guided backpropagation

At a ReLU during the backward pass, a gradient is passed only when
both conditions are satisfied:

- the forward activation is positive
- the incoming gradient is positive

Therefore, negative incoming gradients are suppressed.

The result is often a cleaner input-gradient visualization.

### Interpretation

The selected target is fixed. We compute how the input pixels
influence that target and compare ordinary backpropagation with
guided backpropagation.

The input image itself is never occluded or modified.
"""
)

st.latex(r"\text{Forward activation} > 0")
st.latex(r"\text{Incoming gradient} > 0")
st.latex(
    r"\text{Guided gradient}"
    r"="
    r"\text{incoming gradient}"
    r"\times"
    r"\mathbf{1}(x>0)"
    r"\times"
    r"\mathbf{1}(\text{incoming gradient}>0)"
)

st.caption(
    "Pretrained AlexNet • guided ReLU backward rule • "
    "aspect-ratio-preserving input"
)
