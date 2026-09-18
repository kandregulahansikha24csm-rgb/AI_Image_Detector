import cv2
import numpy as np

FEATURE_NAMES = [
    "Brightness",
    "Noise",
    "Sharpness",
    "Mean_R",
    "Mean_G",
    "Mean_B",
    "Std_R",
    "Std_G",
    "Std_B",
    "Saturation",
    "Contrast",
    "Edge_Density",
    "Hist_0",
    "Hist_1",
    "Hist_2",
    "Hist_3",
    "Hist_4",
    "Hist_5",
    "Hist_6",
    "Hist_7",
    "Hist_8",
    "Hist_9",
    "Hist_10",
    "Hist_11",
    "Hist_12",
    "Hist_13",
    "Hist_14",
    "Hist_15",
]


def extract_features_from_array(image):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    noise = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mean_rgb = np.mean(image, axis=(0, 1))
    std_rgb = np.std(image, axis=(0, 1))

    saturation = float(np.mean(hsv[:, :, 1]))
    contrast = float(np.std(gray))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist_features = [float(value) for value in hist]

    return [
        brightness,
        noise,
        sharpness,
        float(mean_rgb[2]),
        float(mean_rgb[1]),
        float(mean_rgb[0]),
        float(std_rgb[2]),
        float(std_rgb[1]),
        float(std_rgb[0]),
        saturation,
        contrast,
        edge_density,
        *hist_features,
    ]


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return None

    return extract_features_from_array(image)


if __name__ == "__main__":
    image_path = "dataset/real/r.jpg"
    features = extract_features(image_path)

    if features is not None:
        print("\nImage Features")
        print("----------------------------")
        for name, value in zip(FEATURE_NAMES, features):
            print(f"{name}: {value:.4f}")