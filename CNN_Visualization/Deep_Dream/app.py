import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision.models import alexnet, AlexNet_Weights

st.set_page_config(page_title="Deep Dream — Multi-scale", page_icon="🌀", layout="wide")
st.title("Deep Dream: Multi-scale Feature Amplification")
st.write(
    "A classic DeepDream-style workflow: optimize at several image scales, use gentle "
    "gradient-ascent steps, and retain the original image structure. AlexNet weights stay fixed."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# AlexNet feature indices after ReLU: non-negative activations.
LAYERS = {
    "Conv1 — edges / colour contrasts": 1,
    "Conv2 — textures": 4,
    "Conv3 — patterns": 7,
    "Conv4 — complex patterns": 9,
    "Conv5 — visual parts": 11,
}

@st.cache_resource
def load_model():
    model = alexnet(weights=AlexNet_Weights.DEFAULT).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def read_image(file, max_side=384):
    im = Image.open(file).convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return im, ten

def to_pil(t):
    arr = t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))

def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def features_at(model, img01, layer_idx):
    x = (img01 - MEAN) / STD
    for idx, layer in enumerate(model.features):
        x = layer(x)
        if idx == layer_idx:
            return x
    raise RuntimeError("Target layer was not found.")

def variation(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

def objective_for(activation, channel, mode, row, col):
    fmap = activation[:, channel]
    if mode == "Single spatial neuron":
        y = min(max(int(row), 0), fmap.shape[1] - 1)
        x = min(max(int(col), 0), fmap.shape[2] - 1)
        v = fmap[0, y, x]
        return v.square()
    if mode == "Strongest spatial neuron":
        return fmap.flatten().max().square()
    # Mean squared activation rewards the selected channel over the spatial map.
    return fmap.square().mean()

def blur_tensor(x, sigma):
    """Apply a mild Gaussian blur to gradients, suppressing pixel-scale noise."""
    if sigma <= 0:
        return x
    # PIL blur is not differentiable, but this is applied to gradients only.
    arr = x.detach().cpu().numpy()[0].transpose(1, 2, 0)
    im = Image.fromarray(np.uint8(np.clip((arr - arr.min()) / (arr.max() - arr.min() + 1e-8), 0, 1) * 255))
    im = im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    out = np.asarray(im).astype(np.float32) / 255.0
    # Preserve the original gradient scale after blur.
    out = (out - out.mean()) / (out.std() + 1e-8) * (x.detach().std().cpu().item() + 1e-8)
    return torch.from_numpy(out.transpose(2, 0, 1)).unsqueeze(0).to(x.device, dtype=x.dtype)

def optimize_octaves(model, original, layer_idx, channel, target_mode, row, col,
                     octaves, steps_per_octave, step_size, tv_weight,
                     preserve_weight, dream_mix, progress, status):
    # Multi-scale optimization: start at reduced resolution, then carry the
    # residual into progressively larger scales, a common DeepDream technique.
    _, _, H, W = original.shape
    scales = np.linspace(0.55, 1.0, int(octaves))
    current = F.interpolate(original, scale_factor=float(scales[0]), mode="bilinear", align_corners=False)
    current = current.detach()
    original_pyramid = [
        F.interpolate(original, size=(max(32, round(H*s)), max(32, round(W*s))),
                      mode="bilinear", align_corners=False)
        for s in scales
    ]
    rows = []
    total_steps = int(octaves) * int(steps_per_octave)
    completed = 0

    for octave_idx, base in enumerate(original_pyramid):
        if octave_idx == 0:
            current = F.interpolate(current, size=base.shape[-2:], mode="bilinear", align_corners=False)
        else:
            # Preserve detail discovered at the previous scale, but softly blend
            # with the original at this scale to prevent unbounded drift.
            current = F.interpolate(current, size=base.shape[-2:], mode="bilinear", align_corners=False)
            current = (0.82 * current + 0.18 * base).detach()

        image = torch.nn.Parameter(current.clone())
        optimizer = torch.optim.Adam([image], lr=float(step_size))
        for inner in range(int(steps_per_octave)):
            optimizer.zero_grad(set_to_none=True)
            bounded = image.clamp(0, 1)
            act = features_at(model, bounded, layer_idx)
            obj = objective_for(act, channel, target_mode, row, col)
            smooth = variation(bounded)
            preserve = (bounded - base).square().mean()
            # Maximize activation while penalizing noise and large drift.
            loss = -obj + float(tv_weight) * smooth + float(preserve_weight) * preserve
            loss.backward()

            with torch.no_grad():
                if image.grad is not None:
                    grad = image.grad
                    # Mild spatial gradient smoothing via average pooling.
                    grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
                    # Normalize conservatively; avoid the very large steps that
                    # can cause rainbow colour explosions.
                    scale = grad.abs().mean().clamp_min(1e-8)
                    image.grad.copy_(grad / scale)

            optimizer.step()
            with torch.no_grad():
                image.clamp_(0, 1)

            completed += 1
            rows.append({
                "step": completed,
                "octave": octave_idx + 1,
                "activation_objective": float(obj.detach().cpu()),
                "total_variation": float(smooth.detach().cpu()),
                "preservation_loss": float(preserve.detach().cpu()),
                "loss": float(loss.detach().cpu()),
            })
            if completed % max(1, total_steps // 40) == 0 or completed == total_steps:
                progress.progress(completed / total_steps)
                status.caption(f"Octave {octave_idx + 1}/{octaves} · step {inner + 1}/{steps_per_octave}")

        current = image.detach().clamp(0, 1)

    dreamed = F.interpolate(current, size=(H, W), mode="bilinear", align_corners=False)
    # Final restrained blend: keep the original scene recognizable.
    result = ((1.0 - float(dream_mix)) * original + float(dream_mix) * dreamed).clamp(0, 1)
    return result, pd.DataFrame(rows)

model = load_model()
with st.sidebar:
    st.header("Experiment settings")
    uploaded = st.file_uploader("Upload a reference image", type=["png", "jpg", "jpeg", "webp"])
    layer_name = st.selectbox("Target layer", list(LAYERS.keys()), index=2)
    target_mode = st.selectbox("Activation target",
        ["Mean channel activation", "Strongest spatial neuron", "Single spatial neuron"])
    octaves = st.slider("Image scales (octaves)", 2, 4, 3)
    steps_per_octave = st.slider("Steps per scale", 5, 50, 15, 5)
    step_size = st.select_slider("Pixel update strength",
        options=[0.0001, 0.0002, 0.0005, 0.001, 0.002], value=0.0005)
    tv_weight = st.slider("Smoothness regularization", 0.0, 0.30, 0.10, 0.01)
    preserve_weight = st.slider("Preserve image structure", 0.05, 2.0, 0.60, 0.05)
    dream_mix = st.slider("Dream effect strength", 0.10, 0.85, 0.45, 0.05)
    run = st.button("Generate Deep Dream", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload a reference image to begin. Clouds, trees, buildings, and animals are useful examples.")
    st.markdown(
        "**Method:** select a CNN feature, compute its gradient with respect to image pixels, "
        "and amplify it across multiple image scales. The network weights remain fixed."
    )
    st.stop()

original_pil, original = read_image(uploaded)
layer_idx = LAYERS[layer_name]
with torch.no_grad():
    act = features_at(model, original, layer_idx)
    n_channels, fh, fw = int(act.shape[1]), int(act.shape[2]), int(act.shape[3])

with st.sidebar:
    channel = st.number_input("Feature channel", min_value=0, max_value=n_channels-1,
                              value=min(10, n_channels-1), step=1)
    if target_mode == "Single spatial neuron":
        row = st.number_input("Feature-map row", min_value=0, max_value=fh-1, value=fh//2, step=1)
        col = st.number_input("Feature-map column", min_value=0, max_value=fw-1, value=fw//2, step=1)
    else:
        row, col = fh//2, fw//2

st.caption(f"Device: {DEVICE} · Layer: {layer_name} · Feature map: {fh} × {fw} · Channels: {n_channels}")
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Original image")
    st.image(original_pil, use_container_width=True)
with right:
    st.subheader("Deep Dream result")
    if "dream_octave_result" in st.session_state:
        st.image(st.session_state["dream_octave_result"], use_container_width=True)
    else:
        st.image(original_pil, use_container_width=True)
        st.caption("Generate a result to compare it with the original.")

if run:
    progress = st.progress(0.0)
    status = st.empty()
    try:
        result, history = optimize_octaves(
            model, original, layer_idx, int(channel), target_mode, int(row), int(col),
            int(octaves), int(steps_per_octave), float(step_size), float(tv_weight),
            float(preserve_weight), float(dream_mix), progress, status
        )
        result_pil = to_pil(result)
        st.session_state["dream_octave_result"] = result_pil
        st.session_state["dream_octave_history"] = history
        st.success("Multi-scale Deep Dream finished.")
        st.rerun()
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        st.exception(exc)

if "dream_octave_result" in st.session_state:
    st.subheader("Optimization history")
    history = st.session_state["dream_octave_history"]
    st.line_chart(history.set_index("step")[["activation_objective", "total_variation", "preservation_loss"]])
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download generated image", png_bytes(st.session_state["dream_octave_result"]),
                           file_name="deep_dream_result.png", mime="image/png")
    with c2:
        st.download_button("Download history (CSV)", history.to_csv(index=False).encode("utf-8"),
                           file_name="deep_dream_history.csv", mime="text/csv")
    with st.expander("Interpretation"):
        st.write(
            "Multi-scale optimization can reveal features at more than one spatial scale. "
            "The original-image blend and regularization are intended to reduce extreme artifacts, "
            "but DeepDream remains an activation-visualization method and may still produce artificial patterns."
        )
