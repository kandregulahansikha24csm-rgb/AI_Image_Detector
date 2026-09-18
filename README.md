# AI vs Real Image Detector

A desktop prototype that classifies an uploaded image as **real** or **AI-generated** using interpretable image-statistical features and a supervised machine-learning classifier.

## Project structure

- `app.py` - Tkinter desktop interface.
- `feature_extraction.py` - shared 28-feature image pipeline.
- `train_model.py` - leakage-aware model training and model-card creation.
- `evaluate_model.py` - evaluation of the saved model using an image-level holdout set.
- `predict.py` - command-line inference.
- `dataset/real` and `dataset/ai` - labelled source images.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

1. Put real photographs in `dataset/real` and AI-generated images in `dataset/ai`.
2. Train the model: `python train_model.py`
3. Evaluate the saved model: `python evaluate_model.py`
4. Start the prototype: `python app.py`

## Important limitation

This is an educational prototype, not a forensic verification tool. Its prediction reflects patterns in the training data and can be unreliable for new generators, edited images, compressed images, or domains absent from the dataset. Keep the classes reasonably balanced and reserve unseen source images for testing.
