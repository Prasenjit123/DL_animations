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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

st.title("Create Images from CNN Embeddings")
st.write(
    "Optimize an image so that its representation inside a fixed, pretrained AlexNet "
    "resembles the representation of a reference image."
)

@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, weights

def prepare_image(pil_image):
    pil_image = pil_image.convert("RGB")
    # Center crop to preserve a square input and keep display geometry consistent.
    w, h = pil_image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    pil_image = pil_image.crop((left, top, left + side, top + side))
    pil_image = pil_image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def normalize(pixels, mean, std):
    return (pixels - mean) / std

def get_features(model, pixels, mean, std):
    """Return fc7 features. The AlexNet weights are fixed, but gradients can flow to pixels."""
    x = normalize(pixels, mean, std)
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    # AlexNet classifier: Dropout -> fc6 -> ReLU -> Dropout -> fc7 -> ReLU.
    for layer_index in range(6):
        x = model.classifier[layer_index](x)
    return x

def total_variation(x):
    dy = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dx = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dx + dy

def image_from_tensor(x):
    arr = x.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))

def top_predictions(model, weights, pixels, mean, std):
    with torch.no_grad():
        logits = model(normalize(pixels, mean, std))
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, 3)
    categories = weights.meta["categories"]
    return [(categories[int(idx)], float(value)) for value, idx in zip(indices, values)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = MEAN.to(device), STD.to(device)

with st.sidebar:
    st.subheader("Optimization settings")
    steps = st.slider("Optimization steps", 100, 1000, 500, step=100)
    learning_rate = st.select_slider(
        "Learning rate", options=[0.01, 0.02, 0.03, 0.05, 0.08], value=0.03
    )
    resolution = st.select_slider(
        "Optimization resolution", options=[32, 48, 56, 72, 96], value=56
    )
    smoothness = st.select_slider(
        "Smoothness regularization", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.03
    )
    color_prior = st.select_slider(
        "Reference colour prior", options=[0.0, 0.01, 0.03, 0.1, 0.2], value=0.03
    )
    st.caption("The CNN weights are fixed. Only the generated image is optimized.")
    st.caption(f"Compute device: {device}")

uploaded = st.file_uploader("Upload a reference image", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Upload a photo to begin. It will be center-cropped and resized to 224 × 224.")
    st.stop()

try:
    reference_pil = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

reference = prepare_image(reference_pil).to(device)
with st.spinner("Loading pretrained AlexNet (first run may download its weights)..."):
    model, weights = load_model()
    model = model.to(device)

with torch.no_grad():
    target_embedding = get_features(model, reference, mean, std).detach()
    ref_top3 = top_predictions(model, weights, reference, mean, std)

col_ref, col_gen = st.columns(2, gap="large")
with col_ref:
    st.subheader("Reference image")
    st.image(image_from_tensor(reference), width=280)
    st.caption("The target embedding is extracted from AlexNet fc7.")
    for label, probability in ref_top3:
        st.write(f"- {label}: {probability * 100:.1f}%")
with col_gen:
    st.subheader("Generated image")
    previous = st.session_state.get("embedding_result")
    if previous is None:
        st.image(Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (128, 128, 128)), width=280)
        st.caption("Run the optimization to generate an image.")
    else:
        st.image(previous["image"], width=280)
        st.caption("Most recent optimized image.")

st.divider()
st.subheader("Generate an image from the embedding")
st.write(
    "The generated image starts from a neutral image at a lower resolution. "
    "It is repeatedly resized to 224 × 224, passed through AlexNet, and updated to reduce "
    "the fc7 embedding difference. Smoothness and reference-colour priors discourage some "
    "unhelpful noisy solutions."
)

if st.button("Generate image from embedding", type="primary"):
    torch.manual_seed(7)
    # Optimize a genuinely separate low-resolution tensor. clone() prevents aliasing with
    # any tensor later used to measure the initial embedding distance.
    initial_lowres = torch.full((1, 3, resolution, resolution), 0.5, device=device)
    initial_lowres = (initial_lowres + 0.01 * torch.randn_like(initial_lowres)).clamp(0.02, 0.98)
    initial_lowres = initial_lowres.detach().clone()

    def upsample(lowres):
        return F.interpolate(lowres, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)

    with torch.no_grad():
        initial_image = upsample(initial_lowres).detach().clone()
        initial_embedding = get_features(model, initial_image, mean, std).detach()
        initial_distance = torch.linalg.vector_norm(initial_embedding - target_embedding).item()
        # A simple colour-statistics prior; this is not pixel-wise reconstruction.
        target_mean = reference.mean(dim=(2, 3), keepdim=True).detach()
        target_std = reference.std(dim=(2, 3), keepdim=True).detach()

    generated_lowres = torch.nn.Parameter(initial_lowres.clone())
    optimizer = torch.optim.Adam([generated_lowres], lr=float(learning_rate))
    history = []
    progress = st.progress(0, text="Starting optimization...")
    status = st.empty()
    started = time.time()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        generated_image = upsample(generated_lowres)
        current_embedding = get_features(model, generated_image, mean, std)

        # Mean-squared error per embedding dimension keeps the scale interpretable.
        embedding_loss = F.mse_loss(current_embedding, target_embedding)
        tv_loss = total_variation(generated_image)
        current_mean = generated_image.mean(dim=(2, 3), keepdim=True)
        current_std = generated_image.std(dim=(2, 3), keepdim=True)
        colour_loss = F.mse_loss(current_mean, target_mean) + F.mse_loss(current_std, target_std)
        loss = embedding_loss + float(smoothness) * tv_loss + float(color_prior) * colour_loss

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            generated_lowres.clamp_(0.0, 1.0)

        if step == 0 or (step + 1) % 5 == 0 or step == steps - 1:
            history.append({
                "Step": step + 1,
                "Embedding loss": float(embedding_loss.detach().cpu()),
                "Total loss": float(loss.detach().cpu()),
                "Smoothness penalty": float(tv_loss.detach().cpu()),
                "Colour prior": float(colour_loss.detach().cpu()),
            })
        if (step + 1) % 5 == 0 or step == steps - 1:
            progress.progress((step + 1) / steps, text=f"Optimizing image: {step + 1}/{steps}")
            status.caption(f"Embedding loss: {float(embedding_loss.detach().cpu()):.6f}")

    with torch.no_grad():
        final_image = upsample(generated_lowres.detach()).clamp(0, 1).detach().clone()
        final_embedding = get_features(model, final_image, mean, std).detach()
        final_distance = torch.linalg.vector_norm(final_embedding - target_embedding).item()
        final_top3 = top_predictions(model, weights, final_image, mean, std)

    result = {
        "image": image_from_tensor(final_image),
        "history": pd.DataFrame(history),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "top3": final_top3,
        "elapsed": time.time() - started,
        "steps": steps,
        "resolution": resolution,
    }
    st.session_state["embedding_result"] = result
    progress.empty()
    status.empty()
    st.rerun()

result = st.session_state.get("embedding_result")
if result is not None:
    st.divider()
    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Initial embedding distance", f'{result["initial_distance"]:.4f}')
    m2.metric(
        "Final embedding distance",
        f'{result["final_distance"]:.4f}',
        delta=f'{result["final_distance"] - result["initial_distance"]:.4f}',
        delta_color="inverse",
    )
    m3.metric("Optimization time", f'{result["elapsed"]:.1f} s')

    st.write("**Generated-image predictions (diagnostic only)**")
    st.caption("The optimization objective is embedding similarity, not class probability.")
    for label, probability in result["top3"]:
        st.write(f"- {label}: {probability * 100:.1f}%")

    hist = result["history"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(hist["Step"], hist["Embedding loss"], label="Embedding loss")
    ax.plot(hist["Step"], hist["Total loss"], label="Total loss")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization progress")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.download_button(
        "Download optimization history (CSV)",
        hist.to_csv(index=False).encode("utf-8"),
        "embedding_optimization_history.csv",
        "text/csv",
    )
    with st.expander("How to interpret the result"):
        st.markdown(
            "- **Embedding distance:** Euclidean distance between the reference and generated fc7 vectors. A lower value means closer fc7 representations.\n"
            "- **Why it may not look identical:** fc7 is a compact representation; it does not uniquely encode every pixel in the original photo.\n"
            "- **Lower-resolution optimization:** reduces the number of free pixel values and discourages some high-frequency noise.\n"
            "- **Smoothness regularization:** penalizes abrupt neighbouring-pixel changes.\n"
            "- **Reference-colour prior:** encourages similar overall channel means and variation, but does not force pixels to match locations.\n"
            "- **Fixed model:** AlexNet parameters are frozen throughout; only the image changes."
        )

st.caption("Pretrained AlexNet · ImageNet-1K · fc7 embedding (4,096 dimensions) · 224 × 224 input")
