import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights

st.set_page_config(page_title="Deep Dream — CNN Visualization", page_icon="🌀", layout="wide")

st.title("Deep Dream: Enhance Features in an Existing Image")
st.write(
    "Amplify a selected AlexNet feature in a real image. The CNN weights stay fixed; "
    "only image pixels are optimized."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# AlexNet feature sequence: Conv, ReLU, MaxPool, Conv, ReLU, MaxPool, ...
# Use post-ReLU outputs to optimize non-negative feature activations.
LAYER_MAP = {
    "Conv1 — edges and colour contrasts": 1,
    "Conv2 — simple textures": 4,
    "Conv3 — patterns": 7,
    "Conv4 — complex patterns": 9,
    "Conv5 — higher-level visual parts": 11,
}

@st.cache_resource
def load_model():
    model = alexnet(weights=AlexNet_Weights.DEFAULT).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def load_image(file, max_side=320):
    image = Image.open(file).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return image, tensor

def image_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def extract_activation(model, image01, layer_index):
    x = (image01 - MEAN) / STD
    for idx, layer in enumerate(model.features):
        x = layer(x)
        if idx == layer_index:
            return x
    raise ValueError("Could not extract the requested AlexNet layer.")

def total_variation(image):
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()

def target_objective(activation, channel, mode, row, col):
    channel_map = activation[:, channel, :, :]
    if mode == "Single spatial neuron":
        yy = min(max(int(row), 0), channel_map.shape[1] - 1)
        xx = min(max(int(col), 0), channel_map.shape[2] - 1)
        value = channel_map[0, yy, xx]
        return value.square(), value.detach().mean()
    if mode == "Strongest spatial neuron":
        value = channel_map.flatten().max()
        return value.square(), value.detach()
    return channel_map.square().mean(), channel_map.detach().mean()

def optimize_image(model, original, layer_index, channel, target_mode, row, col,
                   steps, learning_rate, tv_weight, preserve_weight,
                   progress, status):
    # Keep pixels in RGB [0,1]. Use conservative, normalized updates to avoid
    # rapid colour explosions and preserve the input's overall structure.
    image = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([image], lr=learning_rate)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        bounded = image.clamp(0.0, 1.0)
        activation = extract_activation(model, bounded, layer_index)
        objective, mean_activation = target_objective(
            activation, channel, target_mode, row, col
        )
        smoothness = total_variation(bounded)
        preservation = (bounded - original).square().mean()
        loss = -objective + tv_weight * smoothness + preserve_weight * preservation
        loss.backward()

        # Smooth the pixel gradient spatially and normalize its scale. This
        # reduces checkerboard/high-frequency colour artifacts.
        with torch.no_grad():
            if image.grad is not None:
                grad = image.grad
                grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
                grad_scale = grad.abs().mean().clamp_min(1e-8)
                image.grad.copy_(grad / grad_scale)

        optimizer.step()
        with torch.no_grad():
            image.clamp_(0.0, 1.0)

        history.append({
            "step": step + 1,
            "objective": float(objective.detach().cpu()),
            "mean_activation": float(mean_activation.detach().cpu()),
            "total_variation": float(smoothness.detach().cpu()),
            "preservation_loss": float(preservation.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        })
        if (step + 1) % max(1, steps // 40) == 0 or step == steps - 1:
            progress.progress((step + 1) / steps)
            status.caption(f"Optimizing image pixels: {step + 1}/{steps}")

    return image.detach().clamp(0.0, 1.0), pd.DataFrame(history)

model = load_model()

with st.sidebar:
    st.header("Experiment settings")
    uploaded = st.file_uploader("Upload a reference image", type=["png", "jpg", "jpeg", "webp"])
    layer_name = st.selectbox("Target layer", list(LAYER_MAP.keys()), index=2)
    target_mode = st.selectbox(
        "Activation target",
        ["Mean channel activation", "Strongest spatial neuron", "Single spatial neuron"],
        help="Start with Mean channel activation for a more stable demonstration.",
    )
    steps = st.slider("Optimization iterations", 20, 400, 100, 20)
    learning_rate = st.select_slider(
        "Pixel update strength", options=[0.0002, 0.0005, 0.001, 0.002, 0.005],
        value=0.001,
    )
    tv_weight = st.slider("Smoothness regularization", 0.0, 0.30, 0.12, 0.01)
    preserve_weight = st.slider("Preserve original image", 0.01, 2.0, 0.50, 0.05)
    run = st.button("Generate Deep Dream", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload an image to begin. Clouds, plants, buildings, and animal images work well.")
    st.markdown(
        "**What happens?** The app selects an internal feature, computes its gradient with "
        "respect to the input pixels, and makes small pixel updates that amplify the feature. "
        "The pretrained model weights are never updated."
    )
    st.stop()

original_pil, original = load_image(uploaded)
layer_index = LAYER_MAP[layer_name]
with torch.no_grad():
    fmap = extract_activation(model, original, layer_index)
    channels, fmap_h, fmap_w = int(fmap.shape[1]), int(fmap.shape[2]), int(fmap.shape[3])

with st.sidebar:
    channel = st.number_input("Feature channel", min_value=0, max_value=channels - 1, value=min(10, channels - 1), step=1)
    if target_mode == "Single spatial neuron":
        row = st.number_input("Feature-map row (y)", min_value=0, max_value=fmap_h - 1, value=fmap_h // 2, step=1)
        col = st.number_input("Feature-map column (x)", min_value=0, max_value=fmap_w - 1, value=fmap_w // 2, step=1)
    else:
        row, col = fmap_h // 2, fmap_w // 2

st.caption(f"Device: {DEVICE} · Layer: {layer_name} · Feature map: {fmap_h} × {fmap_w} · Channels: {channels}")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Original image")
    st.image(original_pil, use_container_width=True)
with right:
    st.subheader("Deep Dream result")
    if "dream_image" in st.session_state:
        st.image(st.session_state["dream_image"], use_container_width=True)
    else:
        st.image(original_pil, use_container_width=True)
        st.caption("Run the experiment to see the optimized result.")

if run:
    progress = st.progress(0.0)
    status = st.empty()
    try:
        result, history = optimize_image(
            model, original, layer_index, int(channel), target_mode, int(row), int(col),
            int(steps), float(learning_rate), float(tv_weight), float(preserve_weight),
            progress, status,
        )
        result_pil = Image.fromarray(
            (result[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        )
        st.session_state["dream_image"] = result_pil
        st.session_state["dream_history"] = history
        st.success("Optimization complete. Compare the result with the original image.")
        st.rerun()
    except Exception as exc:
        st.error(f"Deep Dream failed: {exc}")
        st.exception(exc)

if "dream_image" in st.session_state:
    st.subheader("Optimization history")
    hist = st.session_state["dream_history"]
    st.line_chart(hist.set_index("step")[["objective", "total_variation", "preservation_loss"]])
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download generated image", image_bytes(st.session_state["dream_image"]),
                           file_name="deep_dream_result.png", mime="image/png")
    with c2:
        st.download_button("Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                           file_name="deep_dream_history.csv", mime="text/csv")
    with st.expander("Interpretation and limitations"):
        st.write(
            "A successful run increases the selected activation while regularization helps preserve "
            "image structure. Some feature amplification is expected, but strong psychedelic colours "
            "or repeated texture can indicate an aggressive objective. Deep Dream is a visualization "
            "of learned feature preferences, not a guaranteed realistic enhancement."
        )
