"""Train the AI-vs-real image detector without augmentation leakage."""

import json
import pickle
from pathlib import Path

import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_NAMES, extract_features_from_array

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42


def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))


def augment(image):
    """Augment training images only; never augment before splitting."""
    return [
        image,
        cv2.flip(image, 1),
        rotate_image(image, 12),
        cv2.convertScaleAbs(image, alpha=1.05, beta=5),
    ]


def read_sources(data_dir="dataset"):
    sources, labels = [], []
    data_path = Path(data_dir)
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        label = 0 if "real" in name else 1 if ("ai" in name or "fake" in name) else None
        if label is None:
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image = cv2.imread(str(image_path))
                if image is not None:
                    sources.append(image)
                    labels.append(label)
    return sources, labels


def features_for_images(images, use_augmentation):
    rows = []
    for image in images:
        variants = augment(image) if use_augmentation else [image]
        rows.extend(extract_features_from_array(variant) for variant in variants)
    return rows


def main():
    sources, labels = read_sources()
    class_counts = {"real": labels.count(0), "ai_generated": labels.count(1)}
    if min(class_counts.values()) < 10:
        raise RuntimeError(f"At least 10 source images per class are needed; found {class_counts}.")

    train_images, test_images, train_labels, test_labels = train_test_split(
        sources, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    # Augmentation happens only after the source images are separated.
    X_train = pd.DataFrame(features_for_images(train_images, True), columns=FEATURE_NAMES)
    y_train = [label for label in train_labels for _ in range(4)]
    X_test = pd.DataFrame(features_for_images(test_images, False), columns=FEATURE_NAMES)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(C=2, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    }
    best_name, best_model, best_score = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        score = balanced_accuracy_score(test_labels, model.predict(X_test))
        if score > best_score:
            best_name, best_model, best_score = name, model, score

    predictions = best_model.predict(X_test)
    report = {
        "model": best_name,
        "dataset_source_images": class_counts,
        "train_source_images": len(train_labels),
        "test_source_images": len(test_labels),
        "training_rows_after_augmentation": len(X_train),
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_labels, predictions)), 4),
        "confusion_matrix": confusion_matrix(test_labels, predictions).tolist(),
        "note": "Holdout images are source-level separated before augmentation.",
    }
    with open("model.pkl", "wb") as handle:
        pickle.dump(best_model, handle)
    Path("model_card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(classification_report(test_labels, predictions, target_names=["real", "ai-generated"], zero_division=0))


if __name__ == "__main__":
    main()
