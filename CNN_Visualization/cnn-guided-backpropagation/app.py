import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.transforms.functional import to_tensor, normalize

st.set_page_config(
    page_title="CNN Guided Backpropagation",
    layout="centered"
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    weights = AlexNet_Weights.DEFAULT

    standard = alexnet(weights=weights)
    standard.eval()

    guided = alexnet(weights=weights)
    guided.eval()

    # Replace every ReLU in the guided model with the guided-backward
    # version. Forward behavior remains exactly the same as ReLU.
    for module in guided.modules():
        if isinstance(module, nn.ReLU):
            # We cannot replace modules safely while iterating recursively,
            # so replacement is done below by name.
            pass

    def replace_relu(parent):
        for name, child in parent.named_children():
            if isinstance(child, nn.ReLU):
                setattr(parent, name, GuidedReLU())
            else:
                replace_relu(child)

    replace_relu(guided)

    return standard, guided, weights


class GuidedReLU(nn.Module):
    """
    Forward:
        y = ReLU(x)

    Backward:
        pass gradient only when:
          1. forward activation was positive
          2. incoming gradient is positive

    This is the standard guided-backpropagation rule.
    """

    def forward(self, x):
        return GuidedReLUFunction.apply(x)


class GuidedReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        forward_positive = x > 0
        backward_positive = grad_output > 0

        grad_input = (
            grad_output
            * forward_positive
            * backward_positive
        )

        return grad_input


standard_model, guided_model, weights = load_models()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------
def prepare_image(image, max_dimension=360):
    image = image.convert("RGB")

    w, h = image.size
    scale = min(1.0, max_dimension / max(w, h))

    if scale < 1:
        image = image.resize(
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale)))
            ),
            Image.Resampling.LANCZOS
        )

    return image


def make_input_tensor(image):
    tensor = to_tensor(image)

    tensor = normalize(
        tensor,
        mean=MEAN,
        std=STD
    )

    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------
# Feature-map information
# ---------------------------------------------------------------------
CONV_LAYERS = [
    (1, "Conv1", 64),
    (4, "Conv2", 192),
    (7, "Conv3", 384),
    (9, "Conv4", 256),
    (11, "Conv5", 256),
]


def get_feature_activation(model, x, layer_index):
    captured = {}

    def hook(module, inputs, output):
        captured["activation"] = output

    handle = model.features[layer_index].register_forward_hook(hook)

    model(x)

    handle.remove()

    return captured["activation"]


# ---------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.stop()

image = prepare_image(Image.open(uploaded))

x_probe = make_input_tensor(image)

st.title("CNN Guided Backpropagation")

st.caption(
    "Compare ordinary backpropagation with guided backpropagation "
    "for one selected convolutional neuron."
)

st.info(
    "The selected feature-map neuron is retained as the target. "
    "All other neurons are not used as the backward target."
)

st.subheader("1. Select a feature-map neuron")

layer_labels = [
    f"{name} • {channels} channels"
    for _, name, channels in CONV_LAYERS
]

selected_layer_label = st.selectbox(
    "Layer",
    layer_labels,
    index=2
)

layer_position = layer_labels.index(selected_layer_label)
layer_index, layer_name, num_channels = CONV_LAYERS[layer_position]

with torch.no_grad():
    probe_activation = get_feature_activation(
        standard_model,
        x_probe,
        layer_index
    )

_, channels, height, width = probe_activation.shape

selection_mode = st.radio(
    "Neuron selection",
    ["Strongest active neuron", "Choose manually"],
    horizontal=True
)

if selection_mode == "Strongest active neuron":

    flat_index = int(
        torch.argmax(probe_activation[0]).item()
    )

    channel = flat_index // (height * width)
    remainder = flat_index % (height * width)
    row = remainder // width
    col = remainder % width

    st.caption(
        f"Selected {layer_name}: channel {channel}, "
        f"position ({row}, {col})"
    )

else:

    c1, c2, c3 = st.columns(3)

    with c1:
        channel = st.number_input(
            "Channel",
            min_value=0,
            max_value=channels - 1,
            value=0,
            step=1
        )

    with c2:
        row = st.number_input(
            "Row",
            min_value=0,
            max_value=height - 1,
            value=height // 2,
            step=1
        )

    with c3:
        col = st.number_input(
            "Column",
            min_value=0,
            max_value=width - 1,
            value=width // 2,
            step=1
        )

    channel = int(channel)
    row = int(row)
    col = int(col)

    st.caption(
        f"Selected {layer_name}: channel {channel}, "
        f"position ({row}, {col})"
    )


# ---------------------------------------------------------------------
# Exact target gradient
# ---------------------------------------------------------------------
def compute_target_gradient(model, image, layer_index, channel, row, col):
    x = make_input_tensor(image)
    x.requires_grad_(True)

    activation = get_feature_activation(
        model,
        x,
        layer_index
    )

    target = activation[
        0,
        channel,
        row,
        col
    ]

    gradient = torch.autograd.grad(
        outputs=target,
        inputs=x,
        retain_graph=False,
        create_graph=False
    )[0]

    return (
        gradient.detach()[0].cpu().numpy(),
        float(target.detach().cpu())
    )


standard_gradient, standard_target = compute_target_gradient(
    standard_model,
    image,
    layer_index,
    channel,
    row,
    col
)

guided_gradient, guided_target = compute_target_gradient(
    guided_model,
    image,
    layer_index,
    channel,
    row,
    col
)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def gradient_to_display(gradient_chw, mode):
    # C,H,W -> H,W,C
    gradient = np.transpose(
        gradient_chw,
        (1, 2, 0)
    )

    if mode == "Signed gradient":
        absolute = np.abs(gradient)
        values = absolute[absolute > 1e-12]

        if values.size == 0:
            return np.full(
                (*gradient.shape[:2], 3),
                128,
                dtype=np.uint8
            )

        scale = float(np.percentile(values, 99))

        if scale <= 0:
            scale = float(values.max())

        normalized = np.clip(
            gradient / scale,
            -1,
            1
        )

        # Zero -> mid gray.
        display = normalized * 0.5 + 0.5

    else:
        magnitude = np.max(
            np.abs(gradient),
            axis=2
        )

        values = magnitude[magnitude > 1e-12]

        if values.size == 0:
            normalized = np.zeros_like(magnitude)
        else:
            scale = float(np.percentile(values, 99))

            if scale <= 0:
                scale = float(values.max())

            normalized = np.clip(
                magnitude / scale,
                0,
                1
            )

        display = np.stack(
            [normalized, normalized, normalized],
            axis=2
        )

    return np.clip(
        display * 255,
        0,
        255
    ).astype(np.uint8)


visualization_mode = st.radio(
    "Visualization",
    ["Signed gradient", "Gradient magnitude"],
    horizontal=True
)


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------
st.divider()

left, middle, right = st.columns(3)

with left:
    st.markdown("**Original**")
    st.image(
        image,
        width=210
    )

with middle:
    st.markdown("**Backpropagation**")
    st.image(
        gradient_to_display(
            standard_gradient,
            visualization_mode
        ),
        width=210
    )

with right:
    st.markdown("**Guided backpropagation**")
    st.image(
        gradient_to_display(
            guided_gradient,
            visualization_mode
        ),
        width=210
    )

st.caption(
    "Guided backpropagation suppresses negative gradients during the "
    "backward pass through ReLU units."
)


# ---------------------------------------------------------------------
# Technical details
# ---------------------------------------------------------------------
with st.expander("Technical details"):

    st.write(
        f"Target: {layer_name}, channel {channel}, "
        f"position ({row}, {col})"
    )

    st.write(
        f"Standard target activation: {standard_target:.6e}"
    )

    st.write(
        f"Guided target activation: {guided_target:.6e}"
    )

    st.write(
        f"Input size: {image.width} × {image.height}"
    )


# ---------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------
st.divider()

st.subheader("How guided backpropagation differs")

st.markdown(
    """
### Ordinary backpropagation

The gradient is propagated through the network using the usual chain
rule and the ReLU derivative.

### Guided backpropagation

At a ReLU during the backward pass, the gradient is passed only when
both conditions are satisfied:

- the forward activation is positive
- the incoming gradient is positive

Therefore, negative incoming gradients are suppressed.

The result is often a cleaner input-gradient visualization.

### Interpretation

The selected feature-map neuron is fixed. We compute how the input
pixels influence that neuron and compare ordinary and guided
backpropagation.

The input image itself is never occluded or modified.
"""
)

st.latex(r"\text{Forward activation} > 0")
st.latex(r"\text{Incoming gradient} > 0")
st.latex(r"\text{Guided gradient} = \text{incoming gradient} \times \mathbf{1}(x>0) \times \mathbf{1}(\text{incoming gradient}>0)")


st.caption(
    "Pretrained AlexNet • guided ReLU backward rule • "
    "aspect-ratio-preserving input"
)
