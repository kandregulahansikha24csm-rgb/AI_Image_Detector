"""Evaluate a saved detector with a source-image-level holdout split."""

import json
import pickle
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_NAMES, extract_features_from_array

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_source_images(data_dir="dataset"):
    rows, labels = [], []
    for folder in sorted(Path(data_dir).iterdir()):
        if not folder.is_dir():
            continue
        lower_name = folder.name.lower()
        label = 0 if "real" in lower_name else 1 if ("ai" in lower_name or "fake" in lower_name) else None
        if label is None:
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path))
            if image is not None:
                rows.append(extract_features_from_array(image))
                labels.append(label)
    return pd.DataFrame(rows, columns=FEATURE_NAMES), labels


if __name__ == "__main__":
    X, y = collect_source_images()
    if len(set(y)) < 2 or min(y.count(0), y.count(1)) < 2:
        raise RuntimeError("Evaluation needs at least two readable images in each class.")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    with open("model.pkl", "rb") as handle:
        model = pickle.load(handle)
    predictions = model.predict(X_test)
    metrics = {
        "test_images": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, predictions, target_names=["real", "ai-generated"], zero_division=0))
