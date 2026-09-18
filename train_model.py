"""Train the AI-vs-real image detector without augmentation leakage."""

import json
import pickle
from pathlib import Path

import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_NAMES, extract_features_from_array


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42


def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        angle,
        1.0
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )


def augment(image):
    """Augment training images only."""
    return [
        image,
        cv2.flip(image, 1),
        rotate_image(image, 12),
        cv2.convertScaleAbs(
            image,
            alpha=1.05,
            beta=5
        ),
    ]


def read_sources(data_dir="dataset"):
    """
    Read only source-image folders.

    real -> label 0
    ai   -> label 1
    fake -> label 1

    Processed folders are intentionally ignored.
    """

    sources = []
    labels = []

    data_path = Path(data_dir)

    source_folders = [
        ("real", 0),
        ("ai", 1),
        ("fake", 1),
    ]

    for folder_name, label in source_folders:

        folder = data_path / folder_name

        if not folder.exists():
            print(f"Warning: folder not found: {folder}")
            continue

        for image_path in sorted(folder.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is not None:
                sources.append(image)
                labels.append(label)

    return sources, labels


def features_for_images(images, use_augmentation):

    rows = []

    for image in images:

        variants = (
            augment(image)
            if use_augmentation
            else [image]
        )

        for variant in variants:
            rows.append(
                extract_features_from_array(variant)
            )

    return rows


def main():

    print("\nLoading source images...")

    sources, labels = read_sources()

    class_counts = {
        "real": labels.count(0),
        "ai_generated": labels.count(1),
    }

    print("\nSource dataset:")
    print(json.dumps(class_counts, indent=2))

    if min(class_counts.values()) < 10:
        raise RuntimeError(
            f"At least 10 source images per class are needed; "
            f"found {class_counts}."
        )

    print("\nSplitting source images into training and holdout sets...")

    train_images, test_images, train_labels, test_labels = train_test_split(
        sources,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("Training source images:", len(train_images))
    print("Holdout source images :", len(test_images))

    # IMPORTANT:
    # Augmentation happens ONLY after source-level splitting.

    print("\nExtracting training features with augmentation...")

    X_train = pd.DataFrame(
        features_for_images(
            train_images,
            True
        ),
        columns=FEATURE_NAMES,
    )

    y_train = [
        label
        for label in train_labels
        for _ in range(4)
    ]

    print("Training rows after augmentation:", len(X_train))

    print("\nExtracting holdout features without augmentation...")

    X_test = pd.DataFrame(
        features_for_images(
            test_images,
            False
        ),
        columns=FEATURE_NAMES,
    )

    print("Holdout rows:", len(X_test))

    candidates = {

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                C=2,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    }

    best_name = None
    best_model = None
    best_score = -1.0

    print("\nTraining candidate models...")

    for name, model in candidates.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        score = balanced_accuracy_score(
            test_labels,
            predictions
        )

        print(
            f"{name} balanced accuracy: "
            f"{score:.4f}"
        )

        if score > best_score:
            best_name = name
            best_model = model
            best_score = score

    predictions = best_model.predict(X_test)

    report = {

        "model": best_name,

        "dataset_source_images": class_counts,

        "train_source_images": len(train_labels),

        "test_source_images": len(test_labels),

        "training_rows_after_augmentation": len(X_train),

        "accuracy": round(
            float(
                accuracy_score(
                    test_labels,
                    predictions
                )
            ),
            4,
        ),

        "balanced_accuracy": round(
            float(
                balanced_accuracy_score(
                    test_labels,
                    predictions
                )
            ),
            4,
        ),

        "confusion_matrix": (
            confusion_matrix(
                test_labels,
                predictions
            ).tolist()
        ),

        "note": (
            "Only source images from dataset/real, dataset/ai "
            "and dataset/fake were used. Holdout images were "
            "separated before augmentation."
        ),
    }

    with open(
        "model.pkl",
        "wb"
    ) as handle:

        pickle.dump(
            best_model,
            handle
        )

    Path(
        "model_card.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("FINAL MODEL RESULTS")
    print("=" * 50)

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=[
                "real",
                "ai-generated"
            ],
            zero_division=0,
        )
    )

    print("=" * 50)
    print("Training completed successfully.")
    print("Model saved as model.pkl")
    print("Model card saved as model_card.json")
    print("=" * 50)


if __name__ == "__main__":
    main()