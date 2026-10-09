# CNN Image from Embeddings — Version 2

## Run locally

```bash
conda activate torch-env
pip install -r requirements.txt
streamlit run app.py
```

The app uses pretrained AlexNet and may download its weights the first time it runs.

## What changed in v2

- Fixed initial-distance measurement by cloning the starting image before optimization.
- Optimizes a lower-resolution image and upsamples it to 224×224, reducing free pixel degrees of freedom.
- Adds configurable total-variation smoothness and a light reference colour-statistics prior.
- Reports the initial and final Euclidean distance between fc7 feature vectors.
- Keeps AlexNet frozen; only image pixels are optimized.

This is feature inversion, not guaranteed photographic reconstruction. A feature embedding does not uniquely determine the original pixels.
