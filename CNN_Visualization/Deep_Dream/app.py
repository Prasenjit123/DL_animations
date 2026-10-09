"""A Streamlit teaching app for object-emphasized, multi-octave DeepDream."""
from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import GoogLeNet_Weights, googlenet

GOOGLE_ARTICLE = "https://research.google/blog/inceptionism-going-deeper-into-neural-networks/"
GOOGLE_CODE = "https://github.com/google/deepdream"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = GoogLeNet_Weights.DEFAULT
IMAGENET_CATEGORIES = WEIGHTS.meta["categories"]

# ImageNet normalization expected by the pretrained weights.
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), device=DEVICE).view(1, 3, 1, 1)

LAYER_NAMES = {
    "Inception 3a — textures and edges": "inception3a",
    "Inception 3b — repeated motifs": "inception3b",
    "Inception 4a — object parts": "inception4a",
    "Inception 4b — object parts": "inception4b",
    "Inception 4c — classic face-like motifs": "inception4c",
    "Inception 4d — animal parts and complex motifs": "inception4d",
    "Inception 4e — high-level object patterns": "inception4e",
    "Inception 5a — high-level patterns": "inception5a",
    "Inception 5b — high-level patterns": "inception5b",
}

# Values must match the category names used by torchvision's pretrained weights.
CLASS_PRESETS = [
    ("Bird — bald eagle", "bald eagle"),
    ("Bird — great grey owl", "great grey owl"),
    ("Bird — kite", "kite"),
    ("Bird — robin", "robin"),
    ("Church tower / bell cote", "bell cote"),
    ("Church building", "church"),
    ("Castle", "castle"),
    ("Monastery", "monastery"),
    ("Palace", "palace"),
    ("Mosque", "mosque"),
    ("Dog — golden retriever", "golden retriever"),
    ("Cat — tabby", "tabby, tabby cat"),
    ("Butterfly — monarch", "monarch, monarch butterfly"),
    ("Goldfish", "goldfish"),
    ("Airplane", "airliner"),
    ("Ship", "container ship"),
    ("Spider", "barn spider"),
]
CLASS_PRESETS = [(label, name) for label, name in CLASS_PRESETS if name in IMAGENET_CATEGORIES]
CLASS_INDEX = {label: IMAGENET_CATEGORIES.index(name) for label, name in CLASS_PRESETS}

st.set_page_config(page_title="CNN DeepDream Lab", page_icon="🌌", layout="wide")
st.title("CNN DeepDream Lab")
st.write(
    "Upload a real image and let a pretrained CNN amplify patterns it detects. "
    "The defaults emphasize detailed, bird-like patterns using Inception 4d and bald-eagle guidance. "
    "Choose the class-free option whenever you want to demonstrate unguided, classic DeepDream."
)


@st.cache_resource
def load_model():
    """Load a frozen ImageNet GoogLeNet feature extractor and classifier."""
    net = googlenet(weights=WEIGHTS).to(DEVICE).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def load_image(upload, max_side: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(upload).convert("RGB")
    scale = min(1.0, float(max_side) / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return image, tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def forward_with_activation(
    model: torch.nn.Module, image_01: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected feature map and final ImageNet logits, with gradients intact."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Layer {layer_name} returned an unexpected output type.")
        captured["value"] = output

    handle = getattr(model, layer_name).register_forward_hook(hook)
    try:
        output = model((image_01 - IMAGENET_MEAN) / IMAGENET_STD)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError(f"No activation was captured from {layer_name}.")
    logits = output.logits if hasattr(output, "logits") else output
    return captured["value"], logits


def get_activation(model: torch.nn.Module, image_01: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Convenience wrapper for reporting an activation map."""
    return forward_with_activation(model, image_01, layer_name)[0]


def choose_channels(activation: torch.Tensor, count: int) -> torch.Tensor:
    """Select channels that respond most strongly to the starting image."""
    scores = activation.detach().abs().mean(dim=(0, 2, 3))
    return torch.topk(scores, k=min(int(count), activation.shape[1])).indices


def feature_objective(
    activation: torch.Tensor,
    focused_channels: torch.Tensor | None,
) -> torch.Tensor:
    """Amplify strongly responding features across the selected layer.

    Squared activation restores the stronger, motif-rich behaviour of the earlier
    version. By default every channel participates, as in classic layer-wide
    DeepDream; focusing on a subset is an optional experiment.
    """
    if focused_channels is not None:
        activation = activation.index_select(1, focused_channels)
    return activation.square().mean()


def smooth_gradient(gradient: torch.Tensor, passes: int) -> torch.Tensor:
    """Low-pass smooth the update so it does not overemphasize pixel-scale noise."""
    result = gradient
    for _ in range(int(passes)):
        result = F.avg_pool2d(result, kernel_size=3, stride=1, padding=1)
    return result


def total_variation(image: torch.Tensor) -> torch.Tensor:
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def build_octaves(image: torch.Tensor, octave_count: int, octave_scale: float = 1.45) -> list[torch.Tensor]:
    """Create progressively smaller image scales while preserving aspect ratio."""
    octaves = [image]
    for _ in range(1, int(octave_count)):
        previous = octaves[-1]
        height = max(64, round(previous.shape[-2] / octave_scale))
        width = max(64, round(previous.shape[-1] / octave_scale))
        if (height, width) == previous.shape[-2:]:
            break
        octaves.append(
            F.interpolate(previous, size=(height, width), mode="bilinear", align_corners=False)
        )
    return octaves


def run_deepdream(
    model: torch.nn.Module,
    original: torch.Tensor,
    layer_name: str,
    octave_count: int,
    iterations_per_octave: int,
    step_size: float,
    jitter: int,
    smoothing_passes: int,
    tv_weight: float,
    blend: float,
    focused_channel_count: int | None,
    target_class_index: int | None,
    class_guidance_strength: float,
    progress_bar,
    status_area,
) -> tuple[torch.Tensor, pd.DataFrame, float, float, float | None, float | None]:
    """Amplify CNN patterns with correctly signed gradient ascent over image scales."""
    octaves = build_octaves(original, octave_count)
    total_steps = len(octaves) * int(iterations_per_octave)
    completed = 0
    history: list[dict] = []

    with torch.no_grad():
        start_activation, start_logits = forward_with_activation(model, original, layer_name)
        focused_channels = (
            choose_channels(start_activation, focused_channel_count)
            if focused_channel_count is not None else None
        )
        initial_feature_objective = float(feature_objective(start_activation, focused_channels).cpu())
        initial_probability = (
            float(torch.softmax(start_logits, dim=1)[0, target_class_index].cpu())
            if target_class_index is not None else None
        )

    # Start small, then upscale the detail residual into the next octave.
    detail = torch.zeros_like(octaves[-1])
    dreamed = None

    for octave_index, base in enumerate(reversed(octaves)):
        if detail.shape[-2:] != base.shape[-2:]:
            detail = F.interpolate(detail, size=base.shape[-2:], mode="bilinear", align_corners=False)
        image = (base + detail).clamp(0.0, 1.0).detach().requires_grad_(True)

        for iteration in range(int(iterations_per_octave)):
            if image.grad is not None:
                image.grad = None
            if jitter > 0:
                shift_y = random.randint(-int(jitter), int(jitter))
                shift_x = random.randint(-int(jitter), int(jitter))
            else:
                shift_y = shift_x = 0
            shifted_image = torch.roll(image, shifts=(shift_y, shift_x), dims=(2, 3))

            activation, logits = forward_with_activation(model, shifted_image.clamp(0.0, 1.0), layer_name)
            feature_score = feature_objective(activation, focused_channels)
            target_probability = None
            target_log_probability = None
            if target_class_index is not None:
                # Optional class guidance is handled as a separate normalized gradient.
                # This makes the slider meaningful even when classifier and layer-loss
                # gradients have very different raw magnitudes.
                target_log_probability = F.log_softmax(logits, dim=1)[0, target_class_index]
                target_probability = float(torch.softmax(logits.detach(), dim=1)[0, target_class_index].cpu())

            bounded_shifted = shifted_image.clamp(0.0, 1.0)
            smoothness = total_variation(bounded_shifted)
            need_class_gradient = target_log_probability is not None
            need_tv_gradient = float(tv_weight) > 0.0

            # Normalize the feature gradient independently so a class gradient or a
            # smoothing term cannot silently dominate it just because of scale.
            feature_gradient = torch.autograd.grad(
                feature_score, image, retain_graph=(need_class_gradient or need_tv_gradient)
            )[0]
            feature_gradient = feature_gradient / feature_gradient.abs().mean().clamp_min(1e-8)
            gradient = feature_gradient

            if target_log_probability is not None:
                class_gradient = torch.autograd.grad(
                    target_log_probability, image, retain_graph=need_tv_gradient
                )[0]
                class_gradient = class_gradient / class_gradient.abs().mean().clamp_min(1e-8)
                gradient = gradient + float(class_guidance_strength) * class_gradient

            if need_tv_gradient:
                tv_gradient = torch.autograd.grad(smoothness, image)[0]
                tv_gradient = tv_gradient / tv_gradient.abs().mean().clamp_min(1e-8)
                gradient = gradient - float(tv_weight) * tv_gradient

            with torch.no_grad():
                gradient = smooth_gradient(gradient, smoothing_passes)
                gradient_scale = gradient.abs().mean().clamp_min(1e-8)
                image.add_(float(step_size) * gradient / gradient_scale)
                image.clamp_(0.0, 1.0)

            completed += 1
            history.append({
                "iteration": completed,
                "octave": len(octaves) - octave_index,
                "feature_objective": float(feature_score.detach().cpu()),
                "target_class_probability": target_probability,
                "total_variation": float(smoothness.detach().cpu()),
            })
            if completed == total_steps or completed % max(1, total_steps // 35) == 0:
                progress_bar.progress(completed / total_steps)
                status_area.caption(
                    f"Scale {octave_index + 1}/{len(octaves)} · "
                    f"iteration {iteration + 1}/{iterations_per_octave} · "
                    f"feature objective {float(feature_score.detach().cpu()):.5g}"
                )

        dreamed = image.detach()
        detail = dreamed - base

    assert dreamed is not None
    dreamed = F.interpolate(dreamed, size=original.shape[-2:], mode="bilinear", align_corners=False)
    result = ((1.0 - float(blend)) * original + float(blend) * dreamed).clamp(0.0, 1.0)

    with torch.no_grad():
        end_activation, end_logits = forward_with_activation(model, result, layer_name)
        final_feature_objective = float(feature_objective(end_activation, focused_channels).cpu())
        final_probability = (
            float(torch.softmax(end_logits, dim=1)[0, target_class_index].cpu())
            if target_class_index is not None else None
        )
    return result, pd.DataFrame(history), initial_feature_objective, final_feature_objective, initial_probability, final_probability


model = load_model()
with st.sidebar:
    st.header("DeepDream controls")
    uploaded_file = st.file_uploader("Starting image", type=["png", "jpg", "jpeg", "webp"])
    max_side = st.select_slider(
        "Maximum image dimension (pixels)", options=[192, 256, 320, 384, 512, 640], value=512
    )
    layer_label = st.selectbox("Feature layer", list(LAYER_NAMES.keys()), index=5)
    focus_mode = st.selectbox(
        "Feature emphasis",
        ["All layer activations (classic DeepDream)", "Focused channels (selective motifs)"],
        index=0,
        help="Classic mode combines all channels in the selected layer. Focused mode uses the most active channels from the input image.",
    )
    focused_channel_count = st.slider("Channels to emphasize", 8, 64, 24, 8)
    object_options = ["No specific object — classic DeepDream"] + [label for label, _ in CLASS_PRESETS]
    # The default matches the recommended sky-image setup. The class-free option
    # remains available at the top of the list for the original unguided experiment.
    default_target_index = next(
        (i for i, label in enumerate(object_options) if label == "Bird — bald eagle"), 0
    )
    target_label = st.selectbox(
        "Object to emphasize (optional)", object_options, index=default_target_index,
        help="Defaults to bald eagle to make bird-like patterns more likely. Select the first option for class-free DeepDream.",
    )
    class_guidance_strength = st.slider(
        "Object-emphasis strength", 0.0, 3.0, 1.5, 0.25,
        help="Used only when an object is selected. Higher values bias the patterns toward that ImageNet class and may distort the original scene.",
    )
    octave_count = st.slider("Image scales (octaves)", 2, 5, 4)
    iterations = st.slider("Iterations per scale", 5, 60, 30, 5)
    step_size = st.select_slider(
        "Gradient-ascent step size", options=[0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02], value=0.01
    )
    jitter = st.slider("Random image jitter", 0, 12, 4)
    smoothing = st.slider("Gradient smoothing passes", 0, 3, 1)
    tv_weight = st.slider(
        "Smoothness regularization", 0.0, 0.05, 0.0, 0.005,
        help="Use 0 for the vivid classic DeepDream appearance. Increase slightly only if the result becomes too noisy.",
    )
    blend = st.slider("Blend with original image", 0.4, 1.0, 1.0, 0.05)
    generate = st.button("Generate DeepDream", type="primary", use_container_width=True)

if uploaded_file is None:
    st.info("Upload a sky or cloud image. Defaults are Inception 4d, all layer activations, bald-eagle guidance (strength 1.5), four scales, 30 iterations per scale, step size 0.01, one smoothing pass, no smoothness regularization, and full-strength output. Select the first object option for class-free DeepDream.")
    st.markdown(
        """
**What makes object patterns stronger?**

1. Inception 4c and 4d often give a more detailed, motif-rich look than the most abstract 5b layer.
2. Classic mode maximizes squared activations across the whole layer; no single neuron is selected.
3. Optional object guidance adds a separately normalized classifier gradient toward a selected ImageNet category.
4. Gradient ascent updates the **image pixels**, not the frozen CNN weights, across multiple image scales.
"""
    )
    st.markdown(f"Background reading: [Google Research article]({GOOGLE_ARTICLE}) · [Google's archived code notebook]({GOOGLE_CODE})")
    st.stop()

original_pil, original_tensor = load_image(uploaded_file, int(max_side))
layer_name = LAYER_NAMES[layer_label]
target_class_index = CLASS_INDEX.get(target_label)
focused_count = focused_channel_count if focus_mode.startswith("Focused") else None
with torch.no_grad():
    initial_features = get_activation(model, original_tensor, layer_name)
    initial_shape = tuple(int(d) for d in initial_features.shape)

st.caption(
    f"Device: {DEVICE} · Model: pretrained torchvision GoogLeNet · Layer: {layer_label} · "
    f"Activation shape: {initial_shape[1]} channels × {initial_shape[2]} × {initial_shape[3]}"
)
left_col, right_col = st.columns(2, gap="large")
with left_col:
    st.subheader("Input image")
    st.image(original_pil, use_container_width=True)
with right_col:
    st.subheader("DeepDream output")
    saved_result = st.session_state.get("deepdream_output")
    if saved_result is not None:
        st.image(saved_result, use_container_width=True)
    else:
        st.image(original_pil, use_container_width=True)
        st.caption("The output will appear here after generation.")

if generate:
    progress_bar = st.progress(0.0)
    status_area = st.empty()
    try:
        with st.spinner("Amplifying high-level CNN patterns…"):
            result, history, start_obj, end_obj, start_prob, end_prob = run_deepdream(
                model=model,
                original=original_tensor,
                layer_name=layer_name,
                octave_count=int(octave_count),
                iterations_per_octave=int(iterations),
                step_size=float(step_size),
                jitter=int(jitter),
                smoothing_passes=int(smoothing),
                tv_weight=float(tv_weight),
                blend=float(blend),
                focused_channel_count=focused_count,
                target_class_index=target_class_index,
                class_guidance_strength=float(class_guidance_strength),
                progress_bar=progress_bar,
                status_area=status_area,
            )
        output_pil = tensor_to_pil(result)
        st.session_state["deepdream_output"] = output_pil
        st.session_state["deepdream_history"] = history
        st.session_state["deepdream_start_obj"] = start_obj
        st.session_state["deepdream_end_obj"] = end_obj
        st.session_state["deepdream_start_prob"] = start_prob
        st.session_state["deepdream_end_prob"] = end_prob
        st.rerun()
    except Exception as error:
        st.error(f"DeepDream generation failed: {error}")
        st.exception(error)

if "deepdream_history" in st.session_state:
    st.subheader("Optimization diagnostics")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Initial feature objective", f"{st.session_state['deepdream_start_obj']:.5g}")
    metric2.metric("Final feature objective", f"{st.session_state['deepdream_end_obj']:.5g}")
    start_obj = float(st.session_state["deepdream_start_obj"])
    end_obj = float(st.session_state["deepdream_end_obj"])
    relative = (end_obj - start_obj) / max(abs(start_obj), 1e-8) * 100
    metric3.metric("Feature-objective change", f"{relative:+.1f}%")
    if st.session_state.get("deepdream_start_prob") is not None:
        class_col1, class_col2 = st.columns(2)
        class_col1.metric("Target class probability before", f"{100 * st.session_state['deepdream_start_prob']:.3f}%")
        class_col2.metric("Target class probability after", f"{100 * st.session_state['deepdream_end_prob']:.3f}%")
    history = st.session_state["deepdream_history"]
    st.line_chart(history.set_index("iteration")[["feature_objective"]])
    output = st.session_state["deepdream_output"]
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download dreamed image", data=png_bytes(output),
            file_name="deepdream_output.png", mime="image/png", use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "Download iteration history (CSV)", data=history.to_csv(index=False).encode("utf-8"),
            file_name="deepdream_history.csv", mime="text/csv", use_container_width=True,
        )

with st.expander("How this relates to Google's DeepDream"):
    st.write(
        "The default now follows classic layer-wide DeepDream more closely: it maximizes squared "
        "intermediate activations by gradient ascent on input pixels, carries detail across image scales, "
        "and applies minimal smoothing. The optional object selector is a separate guided extension. "
        "The model is torchvision's pretrained GoogLeNet rather than Google's exact historical checkpoint, "
        "so results can still differ from the 2015 examples."
    )
    st.markdown(f"[Google Research article]({GOOGLE_ARTICLE}) · [Google's archived notebook]({GOOGLE_CODE})")
