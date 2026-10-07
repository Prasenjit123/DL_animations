from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torchvision.models import alexnet, AlexNet_Weights

st.set_page_config(page_title="CNN Filter Visualizer — RGB", page_icon="🎨", layout="wide")
ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_alexnet():
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()
    return model

def get_conv1_filters(model):
    # AlexNet Conv1: [64, 3, 11, 11] -> [64, 11, 11, 3]
    weights = model.features[0].weight.detach().cpu().numpy()
    return np.transpose(weights, (0, 2, 3, 1))

def normalize_filter_rgb(kernel):
    kmin, kmax = float(kernel.min()), float(kernel.max())
    if kmax - kmin < 1e-12:
        return np.zeros_like(kernel)
    return (kernel - kmin) / (kmax - kmin)

def filter_grid(filters, columns=8):
    n = len(filters)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12.5, 12.5))
    axes = np.asarray(axes).reshape(rows, columns)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(normalize_filter_rgb(filters[i]), interpolation="nearest")
        ax.set_title(f"{i + 1}", fontsize=8, pad=2)

    fig.suptitle("AlexNet Conv1 Learned Filters", fontsize=18, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=.01, right=.99, top=.965, bottom=.01, wspace=.08, hspace=.16)
    return fig

def selected_filter_figure(kernel, number):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(normalize_filter_rgb(kernel), interpolation="nearest")
    ax.set_title(f"Filter {number} — 11 × 11 × 3", fontsize=15, fontweight="bold")
    ax.set_xticks(range(kernel.shape[1]))
    ax.set_yticks(range(kernel.shape[0]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return fig

def channel_figure(kernel, number):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    names = ["Red-channel weights", "Green-channel weights", "Blue-channel weights"]
    limit = max(float(np.max(np.abs(kernel))), 1e-12)
    for c, ax in enumerate(axes):
        im = ax.imshow(kernel[:, :, c], cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(names[c], fontsize=10)
        ax.set_xticks(range(kernel.shape[1]))
        ax.set_yticks(range(kernel.shape[0]))
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(f"Filter {number}: channel-specific learned weights", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig

def main():
    st.title("🎨 Visualizing Filters of a CNN")
    st.markdown("""
    **Objective:** visualize the learned filters of the **first convolutional layer**
    of a CNN as images.

    This version follows the lecture idea more closely: first-layer filters operate
    directly on RGB pixels, so their learned weights can be visualized as small
    RGB image patterns.
    """)

    with st.spinner("Loading pretrained AlexNet..."):
        model = load_alexnet()

    filters = get_conv1_filters(model)

    with st.sidebar:
        st.header("Controls")
        st.write("**Network:** AlexNet")
        st.write("**Layer:** Conv1")
        st.write(f"**Filters:** {filters.shape[0]}")
        st.write(f"**Kernel:** {filters.shape[1]} × {filters.shape[2]}")
        st.write("**Input channels:** RGB (3)")
        st.divider()
        selected = st.slider("Select filter", 1, filters.shape[0], 1)
        st.divider()
        st.info("No MNIST dataset or training is required. The app uses pretrained AlexNet Conv1 weights.")

    st.subheader("All learned Conv1 filters")
    st.caption("Each square is one learned 11 × 11 × 3 RGB filter. The 64 filters are shown in an 8 × 8 grid.")
    st.pyplot(filter_grid(filters), clear_figure=True, use_container_width=True)

    st.divider()
    kernel = filters[selected - 1]
    st.subheader(f"Selected filter — Filter {selected}")
    c1, c2 = st.columns([1, 1.25])

    with c1:
        st.pyplot(selected_filter_figure(kernel, selected), clear_figure=True, use_container_width=True)
    with c2:
        st.markdown("### Filter information")
        st.write("**Weight tensor:** `64 × 3 × 11 × 11`")
        st.write("**Selected filter:**", selected)
        st.write("**Filter shape:** `11 × 11 × 3`")
        st.write(f"**Minimum weight:** `{kernel.min():.6f}`")
        st.write(f"**Maximum weight:** `{kernel.max():.6f}`")
        st.write(f"**Mean weight:** `{kernel.mean():.6f}`")
        st.write(f"**L2 norm:** `{np.linalg.norm(kernel):.6f}`")

    st.subheader("R, G and B weight planes")
    st.caption("The RGB image combines the three channel-specific weight planes shown below.")
    st.pyplot(channel_figure(kernel, selected), clear_figure=True, use_container_width=True)

    st.divider()
    st.subheader("What are we visualizing?")
    st.markdown("""
    The displayed RGB patches are the **actual learned weights of the first
    convolutional layer**.

    ```text
    Conv1 weight tensor
    [64, 3, 11, 11]

    64 filters
    × 3 input channels (R, G, B)
    × 11 × 11 spatial weights
    ```

    Because this is the first convolutional layer, the filters operate directly
    on image pixels. Their learned spatial and color patterns can therefore be
    visualized as images.

    **This app does not perform activation maximization.**
    It directly plots the learned first-layer weights, following the visualization
    idea in the lecture slide.
    """)

if __name__ == "__main__":
    main()
