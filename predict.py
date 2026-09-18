"""Predict one image: python predict.py path/to/image.jpg"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from feature_extraction import FEATURE_NAMES, extract_features


def main():
    parser = argparse.ArgumentParser(description="Classify an image as real or AI-generated.")
    parser.add_argument("image", help="Path to a JPG, JPEG, PNG, BMP, or WEBP image")
    args = parser.parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"File not found: {image_path}")
    with open("model.pkl", "rb") as handle:
        model = pickle.load(handle)
    features = extract_features(str(image_path))
    if features is None:
        parser.error("The file could not be decoded as an image.")
    probabilities = model.predict_proba(pd.DataFrame([features], columns=FEATURE_NAMES))[0]
    label = "AI GENERATED IMAGE" if model.classes_[probabilities.argmax()] == 1 else "REAL IMAGE"
    print(f"Prediction: {label}")
    print(f"Model confidence: {probabilities.max() * 100:.2f}%")


if __name__ == "__main__":
    main()
