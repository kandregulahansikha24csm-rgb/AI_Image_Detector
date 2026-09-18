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

from feature_extraction import FEATURE_NAMES, extract_features_from_array


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 500


def load_images():

    real_files = []
    ai_files = []

    # Real images
    real_folder = Path("dataset/real")

    for file in sorted(real_folder.iterdir()):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            real_files.append(file)

    # AI images = ai + fake
    for folder_name in ["ai", "fake"]:

        folder = Path("dataset") / folder_name

        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                ai_files.append(file)

    # Use a fixed sample
    real_files = real_files[:SAMPLES_PER_CLASS]
    ai_files = ai_files[:SAMPLES_PER_CLASS]

    print("Real images selected:", len(real_files))
    print("AI images selected  :", len(ai_files))

    images = []
    labels = []

    for file in real_files:

        image = cv2.imread(str(file))

        if image is not None:
            images.append(image)
            labels.append(0)

    for file in ai_files:

        image = cv2.imread(str(file))

        if image is not None:
            images.append(image)
            labels.append(1)

    return images, labels


def extract_features(images):

    rows = []

    for image in images:
        rows.append(
            extract_features_from_array(image)
        )

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES
    )


print("\nLoading sample dataset...")

images, labels = load_images()

print("\nTotal readable images:", len(images))

X = extract_features(images)
y = labels

print("Feature matrix:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Test images:", len(X_test))

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 50)
print("FAST MODEL RESULTS")
print("=" * 50)

print("Accuracy:", round(accuracy, 4))
print(
    "Balanced Accuracy:",
    round(balanced_accuracy, 4)
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "real",
            "ai-generated"
        ],
        zero_division=0
    )
)

# Save this fast experiment separately
with open(
    "fast_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

report = {
    "experiment": "Fast validation experiment",
    "samples_per_class": SAMPLES_PER_CLASS,
    "total_images": len(images),
    "train_images": len(X_train),
    "test_images": len(X_test),
    "features": len(FEATURE_NAMES),
    "accuracy": round(
        float(accuracy),
        4
    ),
    "balanced_accuracy": round(
        float(balanced_accuracy),
        4
    ),
    "confusion_matrix": cm.tolist()
}

Path(
    "fast_results.json"
).write_text(
    json.dumps(
        report,
        indent=2
    ),
    encoding="utf-8"
)

print("\nFast model saved as: fast_model.pkl")
print("Results saved as: fast_results.json")
print("=" * 50)