import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG_SIZE = (225, 225)
DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered",
)



st.markdown(
    """
    <style>
    .main > div {
        max-width: 980px;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names(class_names_path: str):
    path = Path(class_names_path)

    if not path.exists():
        return DEFAULT_CLASS_NAMES

    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return class_names


def find_model_files():
    model_files = sorted(Path(".").glob("*.keras"))
    model_files += sorted(Path(".").glob("*.h5"))
    return model_files


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize(IMG_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array


def predict_image(model, image_array, class_names):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_class, confidence, probabilities


st.title("Brain Tumor MRI Classification")
st.caption("Custom CNN vs Transfer Learning - Final Deployment Demo")

st.warning(
    "Educational demo only. This app is not a medical diagnosis tool.",
    icon="⚠️",
)

model_files = find_model_files()
model_names = [str(path) for path in model_files]

preferred_model = "best_mobilenetv2.keras"
default_model_index = 0

if preferred_model in model_names:
    default_model_index = model_names.index(preferred_model)

with st.sidebar:
    st.header("Model Settings")

    if model_names:
        selected_model_path = st.selectbox(
            "Model file",
            model_names,
            index=default_model_index,
        )
    else:
        selected_model_path = preferred_model
        st.error("No `.keras` or `.h5` model file found in this folder.")

    class_names_path = st.text_input("Class names file", "class_names.json")
    st.markdown(
        "<p class='small-note'>Put the model and class_names.json in the same folder as app.py.</p>",
        unsafe_allow_html=True,
    )

class_names = load_class_names(class_names_path)

if not Path(selected_model_path).exists():
    st.info(
        "Add `best_mobilenetv2.keras` beside `app.py`, then run the app again."
    )
    st.stop()

try:
    model = load_trained_model(selected_model_path)
except Exception as exc:
    st.error("Could not load the model file.")
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.markdown("Upload an MRI image to get a prediction.")
    st.stop()

original_image, image_array = prepare_image(uploaded_file)
predicted_class, confidence, probabilities = predict_image(
    model,
    image_array,
    class_names,
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.image(original_image, caption="Uploaded MRI", use_container_width=True)

with right_col:
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>Prediction</h3>
            <h2>{predicted_class}</h2>
            <p><b>Confidence:</b> {confidence * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

probability_df = pd.DataFrame(
    {
        "Class": class_names,
        "Probability": probabilities,
    }
).sort_values("Probability", ascending=False)

st.subheader("Class Probabilities")
st.bar_chart(probability_df.set_index("Class"))

with st.expander("Prediction Details"):
    details_df = probability_df.copy()
    details_df["Probability"] = details_df["Probability"].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(details_df, use_container_width=True, hide_index=True)
