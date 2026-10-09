# Create Images from CNN Embeddings — Version 6

## Run locally
```bash
conda activate torch-env
pip install -r requirements.txt
streamlit run app.py
```

The pretrained AlexNet weights are downloaded on the first run if they are not cached.

## Selectable embedding layers
Use **Choose embedding layers** to select one layer or a combination of layers:
- **Conv1** — edges and colour contrasts
- **Conv2** — simple textures
- **Conv3** — patterns and parts
- **Conv4** — complex patterns
- **Conv5** — higher-level visual parts
- **FC7** — higher-level semantic features

When multiple layers are selected, their scale-normalized feature losses are averaged. The selected layer names are shown in the results, and the CSV contains a separate loss column for each selected layer.

## Fixed optimization defaults
- 2,000 optimization steps
- Adam learning rate 0.02
- 40 px optimization grid on the longest side
- Total-variation smoothness weight 0.08
- Reference colour-statistics weight 0.05

The CNN is frozen; only image pixels are optimized. This is feature inversion, not guaranteed photorealistic reconstruction: an embedding does not uniquely specify the original pixels.


## Version 6 update
The optimization loop now runs for 2,000 steps instead of 500. All other optimization settings and layer-selection behavior are unchanged. The longer run may take substantially more time, especially on CPU.
