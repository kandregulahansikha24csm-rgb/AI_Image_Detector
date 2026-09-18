from pathlib import Path

import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_NAMES, extract_features_from_array


MAX_SAMPLES_PER_CLASS = 500
RANDOM_STATE = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_source_features(data_path=Path("dataset")):
    rows = []
    labels = []
    skipped_files = 0

    # Match train_model.py: processed folders are deliberately excluded.
    source_folders = [("real", 0), ("ai", 1), ("fake", 1)]
    for folder_name, label in source_folders:
        folder = data_path / folder_name
        if not folder.is_dir():
            continue

        files = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in files[:MAX_SAMPLES_PER_CLASS]:
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_files += 1
                continue
            features = extract_features_from_array(image)
            if features is None:
                skipped_files += 1
                continue
            rows.append(features)
            labels.append(label)

    return pd.DataFrame(rows, columns=FEATURE_NAMES), labels, skipped_files


def main():
    X, y, skipped_files = collect_source_features()
    if len(set(y)) < 2:
        raise SystemExit("Need readable source images from both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    candidates = {
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 2],
            },
        ),
        "SVM": (
            make_pipeline(
                StandardScaler(),
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            {
                "svc__C": [0.5, 2, 8],
                "svc__gamma": ["scale", "auto"],
            },
        ),
    }

    best_name = None
    best_model = None
    best_score = -1.0
    for name, (estimator, parameter_grid) in candidates.items():
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring="balanced_accuracy",
            cv=cross_validator,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        print(f"{name} 5-fold CV balanced accuracy: {search.best_score_:.4f}")
        print(f"{name} best parameters: {search.best_params_}")
        if search.best_score_ > best_score:
            best_name = name
            best_model = search.best_estimator_
            best_score = search.best_score_

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, list(best_model.classes_).index(1)]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    print(f"Source-only samples: {len(y)} ({y.count(0)} real, {y.count(1)} AI-generated)")
    print(f"Skipped unreadable files: {skipped_files}")
    print(f"Selected model: {best_name}")
    print(f"Holdout size: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Sensitivity / recall (AI-generated): {true_positive / (true_positive + false_negative):.4f}")
    print(f"Specificity (real): {true_negative / (true_negative + false_positive):.4f}")
    print(f"Matthews correlation coefficient: {matthews_corrcoef(y_test, predictions):.4f}")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["real", "ai-generated"],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
import pickle
from pathlib import Path
import cv2
import random
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from feature_extraction import FEATURE_NAMES, extract_features_from_array

MAX_SAMPLES = 500

rows = []
labels = []

data_path = Path('dataset')
for folder in sorted(data_path.iterdir()):
    if not folder.is_dir():
        continue
    name = folder.name.lower()
    if 'real' in name:
        label = 0
    elif 'ai' in name or 'fake' in name:
        label = 1
    else:
        continue
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
    random.shuffle(files)
    for f in files[:MAX_SAMPLES]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        feat = extract_features_from_array(img)
        if feat is None:
            continue
        rows.append(feat)
        labels.append(label)
        if sum(item == label for item in labels) >= MAX_SAMPLES:
            break

if not rows:
    raise SystemExit('No samples found')

X = pd.DataFrame(rows, columns=FEATURE_NAMES)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

with open('model.pkl','rb') as f:
    model = pickle.load(f)

pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
matrix = confusion_matrix(y_test, pred, labels=[0, 1])
true_negative, false_positive, false_negative, true_positive = matrix.ravel()
acc = accuracy_score(y_test, pred)
print(f'Sampled dataset size: {len(rows)} ({sum(label == 0 for label in labels)} real, {sum(label == 1 for label in labels)} AI-generated)')
print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
print(f'Accuracy: {acc*100:.2f}%')
print(f'Balanced accuracy: {balanced_accuracy_score(y_test, pred)*100:.2f}%')
print(f'ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}')
print(f'Sensitivity / recall (AI-generated): {true_positive / (true_positive + false_negative):.4f}')
print(f'Specificity (real): {true_negative / (true_negative + false_positive):.4f}')
print(f'Matthews correlation coefficient: {matthews_corrcoef(y_test, pred):.4f}')
print('Confusion matrix:')
print(matrix)
print('Classification report:')
print(classification_report(y_test, pred, labels=[0, 1], target_names=['real', 'ai-generated'], digits=4, zero_division=0))
