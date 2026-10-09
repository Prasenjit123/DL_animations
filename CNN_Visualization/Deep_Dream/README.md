# CNN DeepDream Lab (PyTorch + Streamlit)

A runnable educational implementation of multi-octave DeepDream with an optional **object-emphasis** mode. It uses pretrained torchvision GoogLeNet, freezes the network weights, and updates image pixels with gradient ascent.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads pretrained GoogLeNet weights, so an internet connection is required once. CUDA is used automatically when available; CPU also works but is slower.

## Suggested settings for a sky/cloud image

- Feature layer: **Inception 4e** or **Inception 5a**
- Feature emphasis: **Focused channels (stronger motifs)**
- Object to emphasize: **Bird — bald eagle** (or choose a bird, church/tower, castle, etc.)
- Object-emphasis strength: **0.10–0.20**
- Image dimension: **512**
- Image scales: **3**
- Iterations per scale: **25–35**
- Gradient-ascent step size: **0.002–0.004**
- Gradient smoothing: **2**
- Smoothness regularization: **0.0025**
- Blend with original image: **0.9–1.0**

For an unconditioned version, select **No specific object — classic DeepDream**. Object-guidance is an optional extension that encourages a specific ImageNet class; it is not part of the class-free original objective.

## Algorithm outline

1. Build a multi-scale image pyramid and start at the smallest scale.
2. Maximize selected Inception-layer activations by **gradient ascent on the input pixels**. The network's parameters stay frozen.
3. Optionally focus on the channels that respond most strongly to the uploaded image.
4. Optionally add a classifier log-probability term for a selected ImageNet category to make a particular kind of object more prominent.
5. Upscale the learned detail residual across octaves, then blend and save the result.

The current torchvision GoogLeNet model is not Google's exact historical internal checkpoint, so outputs will differ from the 2015 images.

References:
- https://research.google/blog/inceptionism-going-deeper-into-neural-networks/
- https://github.com/google/deepdream
