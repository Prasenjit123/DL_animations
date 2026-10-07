import streamlit as st
import torch
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(
    page_title="CNN Occlusion Visualizer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 CNN Occlusion Experiment")
st.write(
    "Hide one region of an image and observe how much the CNN's "
    "prediction changes."
)

@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]

model, preprocess, categories = load_model()


def prepare_experiment_image(image, max_size=700):
    """Resize for a compact classroom demonstration."""
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def predict(image):
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 5)
    return probs, top_indices.tolist(), top_probs.tolist()


def make_occluded_image(image, x, y, patch_size):
    arr = np.array(image).copy()
    arr[y:y + patch_size, x:x + patch_size, :] = 128
    return Image.fromarray(arr)


uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

original = Image.open(uploaded).convert("RGB")
image = prepare_experiment_image(original)

probs, top_indices, top_probs = predict(image)

target_idx = top_indices[0]
target_label = categories[target_idx]
original_probability = float(probs[target_idx])

st.subheader("Original Prediction")

metric_col, info_col = st.columns([1, 2])

with metric_col:
    st.metric(
        "Top-1 class",
        target_label,
        f"{original_probability * 100:.2f}%"
    )

with info_col:
    st.caption(
        "The model's confidence is shown for this particular image. "
        "It is NOT the model's overall classification accuracy."
    )

# Top-5 table
top5_rows = []
for rank, (idx, p) in enumerate(zip(top_indices, top_probs), start=1):
    top5_rows.append(
        {
            "Rank": rank,
            "Class": categories[idx],
            "Probability": f"{p * 100:.2f}%",
        }
    )

st.dataframe(
    top5_rows,
    hide_index=True,
    use_container_width=False,
)

st.divider()
st.subheader("Single-Patch Occlusion")

st.caption(
    "Move the gray square over different regions. "
    "The probability change is always measured for the ORIGINAL "
    "top-1 class."
)

height, width = image.size[1], image.size[0]

# Allow the patch to range from small local regions up to the
# full size of the shorter image dimension.
max_patch = min(width, height)

patch_size = st.slider(
    "Occlusion patch size",
    min_value=20,
    max_value=max_patch,
    value=min(80, max_patch),
    step=10,
)

max_x = width - patch_size
max_y = height - patch_size

x_position = st.slider(
    "Patch horizontal position",
    min_value=0,
    max_value=max_x,
    value=max_x // 2,
    step=5,
)

y_position = st.slider(
    "Patch vertical position",
    min_value=0,
    max_value=max_y,
    value=max_y // 2,
    step=5,
)

occluded = make_occluded_image(
    image,
    x_position,
    y_position,
    patch_size,
)

occluded_probs, occluded_top_indices, occluded_top_probs = predict(occluded)

# Always measure the original target class.
occluded_target_probability = float(occluded_probs[target_idx])
probability_change = occluded_target_probability - original_probability

display_occluded = occluded.copy()
draw = ImageDraw.Draw(display_occluded)
draw.rectangle(
    [
        x_position,
        y_position,
        x_position + patch_size - 1,
        y_position + patch_size - 1,
    ],
    outline=(220, 40, 40),
    width=max(2, patch_size // 20),
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Original")
    st.image(image, width=500)
    st.metric(
        f"{target_label} probability",
        f"{original_probability * 100:.2f}%"
    )

with col2:
    st.markdown("### Occluded")
    st.image(display_occluded, width=500)
    st.metric(
        f"{target_label} probability",
        f"{occluded_target_probability * 100:.2f}%",
        delta=f"{probability_change * 100:.2f} percentage points",
    )

occluded_top_label = categories[occluded_top_indices[0]]

st.write(
    f"**Original top-1:** {target_label}  \n"
    f"**After occlusion top-1:** {occluded_top_label}"
)

if probability_change < 0:
    st.success(
        f"The target-class probability decreased by "
        f"{abs(probability_change) * 100:.2f} percentage points."
    )
elif probability_change > 0:
    st.warning(
        f"The target-class probability increased by "
        f"{probability_change * 100:.2f} percentage points."
    )
else:
    st.info("The target-class probability did not change.")

st.subheader("Top-5 After Occlusion")

occluded_top5_rows = []
for rank, (idx, p) in enumerate(
    zip(occluded_top_indices, occluded_top_probs), start=1
):
    occluded_top5_rows.append(
        {
            "Rank": rank,
            "Class": categories[idx],
            "Probability": f"{p * 100:.2f}%",
        }
    )

st.dataframe(
    occluded_top5_rows,
    hide_index=True,
    use_container_width=False,
)

st.divider()

with st.expander("What is being measured?"):
    st.latex(
        r"\Delta P = P(\mathrm{target}\mid I_{\mathrm{occluded}})"
        r" - P(\mathrm{target}\mid I)"
    )
    st.write(
        "A large negative change means that hiding that region reduced "
        "the model's confidence in the original predicted class."
    )

with st.expander("About this version"):
    st.write(
        "This is still the single-patch version. The next stage will "
        "automatically move the patch across the whole image and create "
        "an occlusion-sensitivity heatmap."
    )
    st.write(
        "For a stronger classroom example, use an image where the "
        "object is centered and occupies a large part of the image."
    )
    st.write(
        "Model: pretrained AlexNet on ImageNet-1K. "
        "No training dataset is required."
    )
