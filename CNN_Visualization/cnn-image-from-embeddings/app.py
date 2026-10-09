import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights

st.set_page_config(page_title="Create Images from CNN Embeddings", page_icon="🧠", layout="centered")

INPUT_SIZE = 224
MEAN_VALUES = (0.485, 0.456, 0.406)
STD_VALUES = (0.229, 0.224, 0.225)

st.title("Create Images from CNN Embeddings")
st.write(
    "Keep a pretrained AlexNet fixed and optimize image pixels so the generated image "
    "matches the reference image's internal representations."
)

@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights

def prepare_image(pil_image):
    """Resize the longest side to 224 without cropping or distorting aspect ratio."""
    pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    scale = INPUT_SIZE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

def normalize(pixels, mean, std):
    return (pixels - mean) / std

LAYER_INFO = {
    "Conv1 — edges and colour contrasts": "First convolutional block, after ReLU.",
    "Conv2 — simple textures": "Second convolutional block, after ReLU.",
    "Conv3 — patterns and parts": "Third convolutional layer, after ReLU.",
    "Conv4 — complex patterns": "Fourth convolutional layer, after ReLU.",
    "Conv5 — higher-level visual parts": "Fifth convolutional layer, after ReLU.",
    "FC7 — semantic features": "Second fully connected layer, after ReLU; more class-semantic than spatial.",
}
DEFAULT_LAYERS = [
    "Conv1 — edges and colour contrasts",
    "Conv3 — patterns and parts",
    "Conv5 — higher-level visual parts",
    "FC7 — semantic features",
]

def get_representations(model, pixels, mean, std):
    """Return named AlexNet activations after each selected layer's ReLU."""
    x = normalize(pixels, mean, std)
    outputs = {}
    conv_indices = {
        "Conv1 — edges and colour contrasts": 1,
        "Conv2 — simple textures": 4,
        "Conv3 — patterns and parts": 7,
        "Conv4 — complex patterns": 9,
        "Conv5 — higher-level visual parts": 11,
    }
    for i, layer in enumerate(model.features):
        x = layer(x)
        for name, index in conv_indices.items():
            if i == index:
                outputs[name] = x
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    for i in range(6):  # classifier[5] is fc7's ReLU output
        x = model.classifier[i](x)
    outputs["FC7 — semantic features"] = x
    return outputs


def embedding_distance(current, target):
    """Scale-normalized RMS distance, comparable across layer sizes."""
    scale = target.detach().std().clamp_min(1e-6)
    return torch.sqrt(F.mse_loss(current / scale, target / scale) + 1e-12)

def total_variation(x):
    vertical = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    horizontal = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return horizontal + vertical

def to_pil(pixels):
    arr = pixels.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))

def top_predictions(model, weights, pixels, mean, std):
    with torch.no_grad():
        logits = model(normalize(pixels, mean, std))
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, 3)
    categories = weights.meta["categories"]
    return [(categories[int(index)], float(value.clamp(0, 1))) for value, index in zip(values, indices)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = torch.tensor(MEAN_VALUES, device=device).view(1, 3, 1, 1)
std = torch.tensor(STD_VALUES, device=device).view(1, 3, 1, 1)

with st.sidebar:
    st.subheader("Experiment settings")
    st.write("Recommended settings are fixed to keep the experiment simple and reproducible.")
    st.markdown("- Optimization steps: **2,000**")
    st.markdown("- Learning rate: **0.02**")
    st.markdown("- Optimization resolution: **40 px** on the longest side")
    st.markdown("- Smoothness: **0.08**")
    st.markdown("- Colour prior: **0.05**")
    st.caption("Embedding layers are selected above the images.")
    st.caption(f"Compute device: {device}")
    st.caption("The CNN weights stay fixed. Only the generated image is optimized.")

uploaded = st.file_uploader("Upload a reference image", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Upload a photograph to begin.")
    st.stop()

try:
    original_pil = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

st.subheader("Choose embedding layers")
st.caption("Select one layer to study it in isolation, or several to match a combination of representations.")
selected_layers = st.multiselect(
    "Layers used as the optimization target",
    options=list(LAYER_INFO.keys()),
    default=DEFAULT_LAYERS,
    help="Earlier convolutional layers emphasize edges/textures; deeper layers emphasize visual parts and semantic features.",
)
if not selected_layers:
    st.warning("Select at least one embedding layer to continue.")
    st.stop()
with st.expander("What does each layer represent?"):
    for layer_name in selected_layers:
        st.markdown(f"- **{layer_name}:** {LAYER_INFO[layer_name]}")

reference = prepare_image(original_pil).to(device)
with st.spinner("Loading pretrained AlexNet (the first run may download its weights)..."):
    model, weights = load_model()
    model = model.to(device)

with torch.no_grad():
    target_representations = get_representations(model, reference, mean, std)
    reference_predictions = top_predictions(model, weights, reference, mean, std)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.subheader("Reference image")
    st.image(to_pil(reference), width=280)
    st.caption("Reference used to extract the target features.")
with col_b:
    st.subheader("Generated image")
    previous = st.session_state.get("embedding_result")
    if previous is None:
        st.image(Image.new("RGB", (280, 280), (238, 238, 238)), width=280)
        st.caption("Click Generate to create an image.")
    else:
        st.image(previous["image"], width=280)
        st.caption("Latest optimized result.")

with st.expander("What is this experiment?"):
    st.markdown(
        "1. Choose one or more AlexNet layers as the embedding target.\n"
        "2. AlexNet processes the reference image and records the selected layer activations.\n"
        "3. The network weights are frozen; only the generated image pixels are optimized.\n"
        "4. Earlier layers emphasize edges and textures, while deeper layers encode more complex parts or semantic information.\n"
        "5. This is **feature inversion**, not exact pixel reconstruction. Different images can have similar features."
    )

if st.button("Generate image", type="primary"):
    STEPS = 2000
    LEARNING_RATE = 0.02
    LOWRES_LONG_SIDE = 40
    SMOOTHNESS_WEIGHT = 0.08
    COLOR_WEIGHT = 0.05

    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)

    ref_h, ref_w = reference.shape[-2:]
    scale = LOWRES_LONG_SIDE / max(ref_h, ref_w)
    low_h = max(12, round(ref_h * scale))
    low_w = max(12, round(ref_w * scale))

    # Start from a neutral, slightly perturbed image; optimize at low resolution
    # to reduce the number of free pixel parameters and high-frequency noise.
    initial = (torch.full((1, 3, low_h, low_w), 0.5, device=device)
               + 0.005 * torch.randn((1, 3, low_h, low_w), device=device)).clamp(0.02, 0.98)
    generated_lowres = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([generated_lowres], lr=LEARNING_RATE)

    target_mean = reference.mean(dim=(2, 3), keepdim=True).detach()
    target_std = reference.std(dim=(2, 3), keepdim=True).detach()

    def upsample(lowres):
        return F.interpolate(lowres, size=(ref_h, ref_w), mode="bilinear", align_corners=False)

    with torch.no_grad():
        initial_image = upsample(generated_lowres).detach()
        initial_representations = get_representations(model, initial_image, mean, std)
        initial_distance = float(torch.stack([
            embedding_distance(initial_representations[name], target_representations[name])
            for name in selected_layers
        ]).mean().item())

    progress = st.progress(0, text="Starting optimization…")
    status = st.empty()
    history = []
    started = time.time()

    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        generated = upsample(generated_lowres)
        current_representations = get_representations(model, generated, mean, std)

        # Each selected layer is scale-normalized before losses are averaged,
        # so a large feature tensor does not automatically dominate the others.
        per_layer_losses = {
            name: F.mse_loss(
                current_representations[name] / target_representations[name].detach().std().clamp_min(1e-6),
                target_representations[name] / target_representations[name].detach().std().clamp_min(1e-6),
            )
            for name in selected_layers
        }
        feature_loss = torch.stack(list(per_layer_losses.values())).mean()

        tv_loss = total_variation(generated)
        current_mean = generated.mean(dim=(2, 3), keepdim=True)
        current_std = generated.std(dim=(2, 3), keepdim=True)
        color_loss = F.mse_loss(current_mean, target_mean) + F.mse_loss(current_std, target_std)
        loss = feature_loss + SMOOTHNESS_WEIGHT * tv_loss + COLOR_WEIGHT * color_loss

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated_lowres.clamp_(0.0, 1.0)

        if step == 0 or (step + 1) % 10 == 0 or step == STEPS - 1:
            history.append({
                "Step": step + 1,
                "Feature loss": float(feature_loss.detach().cpu()),
                **{f"{name} loss": float(value.detach().cpu()) for name, value in per_layer_losses.items()},
                "Smoothness loss": float(tv_loss.detach().cpu()),
                "Total loss": float(loss.detach().cpu()),
            })
        if (step + 1) % 10 == 0 or step == STEPS - 1:
            progress.progress((step + 1) / STEPS, text=f"Optimizing image: {step + 1}/{STEPS}")
            status.caption(f"Feature loss: {float(feature_loss.detach().cpu()):.5f}")

    with torch.no_grad():
        final_image = upsample(generated_lowres.detach()).clamp(0, 1).detach()
        final_representations = get_representations(model, final_image, mean, std)
        final_distance = float(torch.stack([
            embedding_distance(final_representations[name], target_representations[name])
            for name in selected_layers
        ]).mean().item())
        generated_predictions = top_predictions(model, weights, final_image, mean, std)

    result = {
        "image": to_pil(final_image),
        "history": pd.DataFrame(history),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "predictions": generated_predictions,
        "selected_layers": list(selected_layers),
        "elapsed": time.time() - started,
    }
    st.session_state["embedding_result"] = result
    progress.empty()
    status.empty()
    st.rerun()

result = st.session_state.get("embedding_result")
if result is not None:
    st.divider()
    st.subheader("Results")
    st.markdown("**Embedding layers used in this result:** " + ", ".join(result.get("selected_layers", ["FC7 — semantic features"])))
    a, b, c = st.columns(3)
    a.metric("Initial embedding distance", f'{result["initial_distance"]:.3f}')
    b.metric("Final embedding distance", f'{result["final_distance"]:.3f}',
             delta=f'{result["final_distance"] - result["initial_distance"]:.3f}',
             delta_color="inverse")
    c.metric("Optimization time", f'{result["elapsed"]:.1f} s')

    st.write("**Generated-image predictions (secondary diagnostic)**")
    st.caption("Percentages are softmax probabilities and should be between 0% and 100%.")
    for label, probability in result["predictions"]:
        st.write(f"- {label}: {probability * 100:.1f}%")

    history = result["history"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history["Step"], history["Feature loss"], label="Selected-layer embedding loss")
    ax.plot(history["Step"], history["Total loss"], label="Total loss", alpha=0.8)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization progress")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download optimization history (CSV)",
        history.to_csv(index=False).encode("utf-8"),
        "embedding_optimization_history.csv",
        "text/csv",
    )

st.caption("Pretrained AlexNet · fixed network weights · convolutional features + fc7 · aspect ratio preserved")
