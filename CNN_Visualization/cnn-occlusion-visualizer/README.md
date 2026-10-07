# CNN Occlusion Visualizer — Updated Patch Range

This version keeps the existing V4 single-patch occlusion experiment but expands the occlusion patch-size control.

## Patch-size range

The patch can now range from:

- Minimum: 20 px
- Maximum: the shorter dimension of the displayed image
- Default: 80 px (or the maximum if the image is smaller)
- Step: 10 px

This allows both fine-grained and large-region occlusion experiments.

## Run

Use the existing environment:

```bash
conda activate torch-env
streamlit run app.py
```

No reinstallation is needed if the required packages are already installed.
