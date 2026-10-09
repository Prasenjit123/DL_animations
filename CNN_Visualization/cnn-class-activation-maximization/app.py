import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image

st.set_page_config(page_title="CNN Activation Maximization", layout="centered")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_SIZE_OPTIONS = [160, 192, 224]
OPT_RES_OPTIONS = [32, 40, 48, 56, 64, 72]

@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    net = alexnet(weights=weights).to(DEVICE).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, weights

model, weights = load_model()
categories = weights.meta["categories"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def normalize_image(x):
    return (x - MEAN) / STD

def to_uint8(x):
    arr = x.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.uint8(np.clip(arr, 0, 1) * 255)

def total_variation(x):
    # Mean absolute change between adjacent pixels; discourages noisy speckles.
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy

def smooth_image(x):
    # A light 3x3 blur used occasionally, not on every step.
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)

def optimize_for_class(
    target_class, iterations, learning_rate, output_size,
    optimization_resolution, smoothness, pixel_regularization,
    jitter, snapshot_steps
):
    """Optimize only image pixels; keep all pretrained AlexNet parameters fixed.

    Progressive image sizes and regularization make class-specific structure
    easier to see than unconstrained full-resolution pixel ascent.
    """
    # Start near middle gray with a tiny random perturbation. Tiny noise breaks
    # spatial/channel symmetry; it is not a photograph or a class template.
    image = (0.50 + 0.015 * torch.randn(
        1, 3, optimization_resolution, optimization_resolution, device=DEVICE
    )).clamp(0.05, 0.95).requires_grad_(True)

    optimizer = torch.optim.Adam([image], lr=learning_rate)
    snapshots = {}
    snapshot_steps = sorted(set([0] + [int(s) for s in snapshot_steps if int(s) > 0] + [iterations]))

    for step in range(iterations + 1):
        if step in snapshot_steps:
            with torch.no_grad():
                display = F.interpolate(image, size=(output_size, output_size),
                                        mode="bilinear", align_corners=False)
                snapshots[step] = to_uint8(display)

        if step == iterations:
            break

        optimizer.zero_grad(set_to_none=True)

        # Upsample the optimized low-resolution parameter image to AlexNet input.
        current = F.interpolate(image, size=(output_size, output_size),
                                mode="bilinear", align_corners=False)

        # Small random translations reduce overfitting to a single pixel alignment.
        if jitter > 0:
            shift_y = int(torch.randint(-jitter, jitter + 1, (1,), device=DEVICE).item())
            shift_x = int(torch.randint(-jitter, jitter + 1, (1,), device=DEVICE).item())
            current = torch.roll(current, shifts=(shift_y, shift_x), dims=(2, 3))

        logits = model(normalize_image(current))
        target_score = logits[0, target_class]

        tv = total_variation(image)
        # Keep colours/intensities from drifting to extreme values.
        pixel_penalty = ((image - 0.5) ** 2).mean()

        # Minimize negative target score plus regularizers.
        loss = -target_score + smoothness * tv * 100.0 + pixel_regularization * pixel_penalty * 100.0
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.clamp_(0.0, 1.0)
            if step > 0 and step % 40 == 0:
                image.copy_(smooth_image(image).clamp_(0.0, 1.0))

    with torch.no_grad():
        final = F.interpolate(image, size=(output_size, output_size),
                              mode="bilinear", align_corners=False)
        logits = model(normalize_image(final))
        probs = F.softmax(logits, dim=1)
        score = float(logits[0, target_class].item())
        prob = float(probs[0, target_class].item())
        predicted = int(probs.argmax(dim=1).item())
    return to_uint8(final), snapshots, score, prob, predicted

st.title("CNN Activation Maximization")
st.caption(
    "Optimize image pixels to increase a chosen ImageNet class score while keeping "
    "the pretrained AlexNet weights fixed."
)
st.info(
    "Why this version looks better: it optimizes a lower-resolution image, upsamples it, "
    "and applies mild smoothness and pixel regularization. Results are still synthetic "
    "feature visualizations, not realistic photographs."
)

st.subheader("1. Choose target classes")

# A compact, teaching-oriented default list of recognizable and varied classes.
# Indices come from the pretrained torchvision ImageNet-1K category list.
teaching_class_indices = [
    207,  # golden retriever
    292,  # tiger cat
    386,  # African elephant
    504,  # coffee mug
    779,  # school bus
    546,  # electric guitar
    817,  # sports car
    963,  # pizza
    701,  # parachute
]

advanced_classes = st.checkbox(
    "Advanced mode: show all 1,000 ImageNet classes",
    value=False,
)

if advanced_classes:
    available_indices = list(range(len(categories)))
    st.caption(
        "Advanced mode is intended for exploration. For a clear classroom comparison, "
        "the curated list is usually easier to interpret."
    )
else:
    available_indices = teaching_class_indices

class_options = [f"{i}: {categories[i]}" for i in available_indices]
n_classes = st.slider(
    "Number of target classes to compare",
    min_value=1,
    max_value=min(4, len(available_indices)),
    value=min(3, len(available_indices)),
    step=1,
)

default_indices = [207, 504, 963]  # golden retriever, coffee mug, pizza
selected = []
for i in range(n_classes):
    default_index = default_indices[i] if i < len(default_indices) else available_indices[0]
    if default_index not in available_indices:
        default_index = available_indices[0]
    option_indices = available_indices
    default_position = option_indices.index(default_index)
    chosen = st.selectbox(
        f"Target class {i + 1}",
        class_options,
        index=default_position,
        key=f"target_class_{i}_{'advanced' if advanced_classes else 'curated'}",
    )
    selected.append(int(chosen.split(":", 1)[0]))

st.subheader("2. Optimization settings")
c1, c2, c3 = st.columns(3)
with c1:
    iterations = st.slider("Iterations", 100, 800, 400, 50)
with c2:
    learning_rate = st.slider("Learning rate", 0.005, 0.08, 0.025, 0.005, format="%.3f")
with c3:
    output_size = st.selectbox("Output size", OUTPUT_SIZE_OPTIONS, index=2)

c4, c5, c6 = st.columns(3)
with c4:
    opt_resolution = st.selectbox("Optimization resolution", OPT_RES_OPTIONS, index=4,
                                  help="Lower resolution encourages larger shapes instead of pixel noise.")
with c5:
    smoothness = st.select_slider("Smoothness", options=[0.005, 0.01, 0.02, 0.04, 0.06],
                                  value=0.02, format_func=lambda x: f"{x:.3f}")
with c6:
    pixel_regularization = st.select_slider("Pixel regularization", options=[0.001, 0.003, 0.005, 0.01, 0.02],
                                            value=0.005, format_func=lambda x: f"{x:.3f}")

jitter = st.slider("Translation jitter (pixels)", 0, 8, 4, 1,
                   help="Small random translations reduce brittle pixel-aligned patterns.")
st.caption(
    "Suggested first run: 400 iterations, learning rate 0.025, 56×56 optimization resolution, "
    "smoothness 0.020, pixel regularization 0.005. For clearer shapes, try 40×40 or 48×48."
)

run = st.button("Generate class-maximizing images", type="primary")
if run:
    unique_classes = list(dict.fromkeys(selected))
    if len(unique_classes) != len(selected):
        st.warning("Duplicate classes were selected; each distinct class will be generated once.")

    results = []
    progress = st.progress(0)
    status = st.empty()
    snap_points = [iterations // 4, iterations // 2, (3 * iterations) // 4]
    for pos, class_idx in enumerate(unique_classes):
        status.caption(f"Optimizing {pos + 1}/{len(unique_classes)}: {categories[class_idx]}")
        result = optimize_for_class(
            class_idx, iterations, learning_rate, output_size, opt_resolution,
            smoothness, pixel_regularization, jitter, snap_points
        )
        results.append((class_idx, categories[class_idx], result))
        progress.progress((pos + 1) / len(unique_classes))
    status.empty()
    progress.empty()

    st.divider()
    st.subheader("3. Images obtained by maximizing class scores")
    cols = st.columns(min(2, len(results)))
    for idx, (class_idx, class_name, result) in enumerate(results):
        generated, snapshots, score, probability, predicted = result
        with cols[idx % len(cols)]:
            st.markdown(f"### {class_name}")
            st.image(generated, width=280, caption="Optimized synthetic input")
            m1, m2 = st.columns(2)
            m1.metric("Target class score", f"{score:.2f}")
            m2.metric("Target probability", f"{probability:.2%}")
            st.caption(f"Highest predicted class: **{categories[predicted]}**")
            if predicted != class_idx:
                st.caption("The target score was optimized, but another class has the highest final probability.")
            with st.expander("Optimization progression"):
                prog_cols = st.columns(len(snapshots))
                for c, step in zip(prog_cols, sorted(snapshots)):
                    with c:
                        st.caption(f"Step {step}")
                        st.image(snapshots[step], use_container_width=True)

    st.divider()
    st.subheader("How to interpret the result")
    st.markdown(
        """
- The selected class score is maximized; the network weights are **not** trained or changed.
- The generated image often shows textures, colours, contours, or parts associated with the class.
- A high target probability (even 100%) does **not** mean the image looks realistic; it means AlexNet strongly prefers that class.
- If different classes still look too similar, lower the optimization resolution (40 or 48), increase smoothness slightly, and compare the progression.
"""
    )
    st.latex(r"I_{t+1}=I_t+\eta\,\frac{\partial S_c(I_t)}{\partial I_t}")
    st.caption("This is gradient ascent on the input image. Regularization discourages excessively noisy solutions.")

with st.expander("Technical details"):
    st.markdown(
        f"""
- Network: **AlexNet**, pretrained on ImageNet-1K (1,000 classes)
- Device: **{DEVICE}**
- Optimized variable: image pixels only
- CNN weights: fixed
- Parameter image: **{opt_resolution}×{opt_resolution}**, upsampled to **{output_size}×{output_size}**
- Objective: target-class logit maximization with total-variation smoothness and pixel-intensity regularization
"""
    )
