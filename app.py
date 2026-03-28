import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# إعدادات الصفحة
# -----------------------------
st.set_page_config(
    page_title="Shape Classifier",
    page_icon="🔷",
    layout="centered"
)

# -----------------------------
# أسماء الفئات (نفس المشروع)
# -----------------------------
CLASS_NAMES = [
    "triangle",
    "trapezoid",
    "square",
    "rhombus",
    "rectangle",
    "parallelogram",
    "kite",
    "circle"
]

IMG_SIZE = (128, 128)
MODEL_PATH = "shapes_classification.h5"   

# -----------------------------
# تحميل الموديل
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# تجهيز الصورة
# -----------------------------
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    resized = image.resize(IMG_SIZE)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return image, arr

# -----------------------------
# واجهة التطبيق
# -----------------------------
st.title("🔷 Geometric Shape Classifier")
st.write("Upload an image and the model will predict the shape.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    try:
        model = load_model()

        original_image, processed_image = preprocess_image(uploaded_file)

        st.image(original_image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            prediction = model.predict(processed_image)[0]

        predicted_idx = int(np.argmax(prediction))
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(prediction[predicted_idx])

        st.subheader("Result")
        st.success(f"Prediction: {predicted_label}")
        st.write(f"Confidence: {confidence:.2%}")

        st.subheader("All Class Probabilities")
        for label, prob in zip(CLASS_NAMES, prediction):
            st.write(f"{label}: {prob:.2%}")
            st.progress(float(prob))

    except FileNotFoundError:
        st.error("❌ Model file 'cnn_model.h5' not found. Make sure it is in the same folder as app.py")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload an image to start.")