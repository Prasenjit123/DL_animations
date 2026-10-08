import hashlib

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="CNN Occlusion Visualizer", page_icon="🔍", layout="centered")
st.title("🔍 CNN Occlusion Experiment")
st.caption("Drag the gray square directly over the image. Use the slider only to change patch size.")


@st.cache_resource
def load_model():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    return model, weights.transforms(), weights.meta["categories"]


model, preprocess, categories = load_model()


@torch.inference_mode()
def predict(image):
    x = preprocess(image).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_indices = torch.topk(probs, 5)
    return probs, top_indices.tolist(), top_probs.tolist()


def prepare_image(image, max_size=420):
    image = image.convert("RGB").copy()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def clamp_xy(x, y, width, height, size):
    x = max(0, min(int(round(x)), width - size))
    y = max(0, min(int(round(y)), height - size))
    return x, y


def patch_json(x, y, size):
    # Fabric coordinates are explicitly top-left coordinates.
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "left": float(x),
                "top": float(y),
                "originX": "left",
                "originY": "top",
                "width": float(size),
                "height": float(size),
                "scaleX": 1.0,
                "scaleY": 1.0,
                "angle": 0,
                "fill": "rgba(128,128,128,0.72)",
                "stroke": "rgba(220,40,40,1)",
                "strokeWidth": 2,
                "selectable": True,
                "hasControls": False,
                "lockScalingX": True,
                "lockScalingY": True,
                "lockRotation": True,
            }
        ],
    }


def occlude(image, x, y, size):
    arr = np.asarray(image).copy()
    x, y = int(x), int(y)
    arr[y:y + size, x:x + size, :] = 128
    return Image.fromarray(arr)


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

image = prepare_image(Image.open(uploaded), max_size=420)
width, height = image.width, image.height

# Reset state when a different image is uploaded.
image_id = hashlib.md5(np.asarray(image).tobytes()).hexdigest()
if st.session_state.get("occlusion_image_id") != image_id:
    st.session_state.occlusion_image_id = image_id
    st.session_state.patch_size = min(80, width, height)
    st.session_state.patch_x = max(0, (width - st.session_state.patch_size) // 2)
    st.session_state.patch_y = max(0, (height - st.session_state.patch_size) // 2)


# Original prediction
probs, top_indices, _ = predict(image)
target_idx = top_indices[0]
target_label = categories[target_idx]
original_probability = float(probs[target_idx])

st.markdown(
    f"**Original Top-1:** {target_label} "
    f"({original_probability * 100:.2f}%)"
)

st.divider()
st.subheader("Single-Patch Occlusion")

max_patch = min(width, height)
old_size = int(st.session_state.patch_size)
old_size = max(20, min(old_size, max_patch))

patch_size = st.slider(
    "Occlusion patch size",
    min_value=20,
    max_value=max_patch,
    value=old_size,
    step=10,
)
st.session_state.patch_size = patch_size

# Keep the existing top-left position when the patch size changes.
x, y = clamp_xy(
    st.session_state.patch_x,
    st.session_state.patch_y,
    width,
    height,
    patch_size,
)
st.session_state.patch_x = x
st.session_state.patch_y = y

# The canvas pixel dimensions and the displayed comparison images are identical.
initial = patch_json(x, y, patch_size)

canvas_result = st_canvas(
    fill_color="rgba(128,128,128,0.72)",
    stroke_color="rgba(220,40,40,1)",
    stroke_width=2,
    background_color="white",
    background_image=image,
    background_image_fit="stretch",
    update_streamlit=True,
    height=height,
    width=width,
    drawing_mode="transform",
    initial_drawing=initial,
    display_toolbar=False,
    key="occlusion_canvas",
)

# Read the position directly from the object returned by the canvas.
# IMPORTANT: width/height are not used to infer position; only left/top are used.
if canvas_result is not None and canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        obj = objects[0]
        x = float(obj.get("left", x))
        y = float(obj.get("top", y))
        x, y = clamp_xy(x, y, width, height, patch_size)
        st.session_state.patch_x = x
        st.session_state.patch_y = y

x = int(st.session_state.patch_x)
y = int(st.session_state.patch_y)

st.caption(f"Patch position: ({x}, {y})  •  Patch size: {patch_size} × {patch_size} px")

# Generate the model input from EXACTLY the same x/y/size used by the canvas.
occluded = occlude(image, x, y, patch_size)
occluded_probs, occluded_top_indices, _ = predict(occluded)
occluded_target_probability = float(occluded_probs[target_idx])
change = occluded_target_probability - original_probability
occluded_top_label = categories[occluded_top_indices[0]]

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Original")
    st.image(image, width=width)
    st.metric("Target probability", f"{original_probability * 100:.2f}%")

with col2:
    st.markdown("### Occluded")
    st.image(occluded, width=width)
    st.metric(
        "Target probability",
        f"{occluded_target_probability * 100:.2f}%",
        delta=f"{change * 100:.2f} percentage points",
    )

st.write(f"**Original Top-1:** {target_label}")
st.write(f"**After occlusion Top-1:** {occluded_top_label}")

with st.expander("What is being measured?"):
    st.latex(
        r"\Delta P = P(\mathrm{target}\mid I_{\mathrm{occluded}})"
        r" - P(\mathrm{target}\mid I)"
    )
    st.write(
        "A large negative change means that hiding that region reduced "
        "the probability of the original predicted class."
    )

st.caption("Model: pretrained AlexNet on ImageNet-1K. No training is required.")
