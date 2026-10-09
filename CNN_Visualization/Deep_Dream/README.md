# Deep Dream — corrected Streamlit app

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads pretrained AlexNet weights.

## What was improved
- Much more conservative default pixel update rate.
- Stronger image-preservation and smoothness regularization.
- Spatially smoothed, normalized image gradients to reduce high-frequency colour artifacts.
- Lower default iteration count for a safer first run.
- Generated image and history persist after Streamlit reruns.
- Post-ReLU AlexNet feature maps are used for activation maximization.

## Recommended classroom settings
Start with Conv3, Mean channel activation, 60–100 iterations, pixel update strength 0.001,
smoothness 0.12, and preserve-original 0.50. Increase iterations gradually only after inspecting
the result. Deep Dream intentionally amplifies CNN features; it does not guarantee photorealism.
