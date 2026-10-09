# CNN DeepDream Lab — detailed-motif edition

A Streamlit application for exploring class-free DeepDream and optional class-guided dreaming on a real input image. It uses pretrained torchvision GoogLeNet/ImageNet weights, keeps model parameters frozen, and applies gradient ascent to the input pixels.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The pretrained GoogLeNet weights download the first time the app starts. A CUDA-enabled PyTorch installation is recommended for faster experiments; CPU works but can be slow.

## Settings for detailed, object-like patterns

- Start at **Inception 4d**; also compare Inception 4c and 4e. The 5a/5b layers can look coarser or more abstract.
- Start with 4 image scales, 30 iterations per scale, step size 0.01, one gradient-smoothing pass, and smoothness regularization 0.
- The app now defaults to **Bird — bald eagle** with guidance strength **1.5** to bias patterns toward bird-like forms. Select **No specific object — classic DeepDream** to switch back to class-free amplification; guidance is an optional extension, not part of the original class-free method.
- If the output becomes too noisy, enable one more smoothing pass or use a small smoothness regularization value.

## Important limitation

DeepDream amplifies features the model responds to; it does not guarantee a fully recognizable object from every image. The optional class guidance can bias feature changes toward a category but is not a text-to-image generator. The model is torchvision's pretrained GoogLeNet, not Google's original historical checkpoint, so outputs will differ from the 2015 examples.
