import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pickle
from pathlib import Path
import pandas as pd

from feature_extraction import FEATURE_NAMES, extract_features

MODEL_PATH = Path(__file__).with_name("model.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError("model.pkl is missing. Run train_model.py first.")
with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)
selected_image = ""


def upload_image():
    global selected_image

    filename = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if filename:
        selected_image = filename
        image = Image.open(filename)
        image = image.resize((250, 250))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


def predict_image():
    if selected_image == "":
        result_label.config(text="Please upload an image")
        return

    features = extract_features(selected_image)
    if features is None:
        result_label.config(text="The selected file could not be decoded as an image.")
        return
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = model.predict(X)
    confidence = model.predict_proba(X)
    confidence = max(confidence[0]) * 100

    if prediction[0] == 0:
        result = "REAL IMAGE"
    else:
        result = "AI GENERATED IMAGE"

    result_label.config(text=f"{result}\nConfidence : {confidence:.2f}%")


window = tk.Tk()
window.title("AI Image Detector")
window.geometry("500x650")
window.resizable(False, False)

title = tk.Label(window, text="AI IMAGE DETECTOR", font=("Arial", 18, "bold"))
title.pack(pady=10)

upload_button = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 12))
upload_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

predict_button = tk.Button(window, text="Predict", command=predict_image, font=("Arial", 12))
predict_button.pack(pady=20)

result_label = tk.Label(window, text="Prediction will appear here", font=("Arial", 14))
result_label.pack()

window.mainloop()
