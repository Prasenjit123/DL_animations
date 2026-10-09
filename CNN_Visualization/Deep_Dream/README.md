# Deep Dream — multi-scale Streamlit app

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads pretrained AlexNet weights.

## Improvements over the previous version
- Multi-scale/octave optimization rather than a single-resolution run.
- Conservative pixel-update default and spatially smoothed gradients.
- Smoothness and image-preservation penalties.
- Final blend with the original image to retain scene structure.
- Adjustable layer, channel, activation target, number of octaves, and steps per octave.
- Downloads for the generated image and optimization history.

## Recommended starting settings
- Layer: Conv3
- Target: Mean channel activation
- Image scales: 3
- Steps per scale: 10–15
- Pixel update strength: 0.0005
- Smoothness: 0.10
- Preserve image structure: 0.60
- Dream effect strength: 0.45

This should reduce excessive colour/swirling artifacts compared with single-scale aggressive optimization, but cannot guarantee the exact appearance of the original Google DeepDream examples.
