# app/streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import json
from pathlib import Path

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────

METADATA_PATH = "models/metadata.json"
MODEL_PATH    = "models/saved/final_model"
CONFIDENCE_THRESHOLD = 0.60

# Fallback class list if metadata.json is not present
FALLBACK_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

TREATMENTS = {
    "Bacterial_spot":         "Apply copper-based bactericide. Remove and destroy infected leaves. Avoid overhead watering.",
    "Early_blight":           "Apply fungicide containing chlorothalonil or mancozeb. Improve air circulation. Remove lower infected leaves.",
    "Late_blight":            "Apply mancozeb or metalaxyl fungicide immediately. Destroy infected plants. Avoid leaf wetness.",
    "Leaf_Mold":              "Reduce humidity and improve ventilation. Apply fungicide (chlorothalonil) if severe.",
    "Septoria_leaf_spot":     "Remove infected leaves. Apply fungicide (chlorothalonil or copper). Avoid overhead irrigation.",
    "Spider_mites":           "Apply miticide or neem oil spray. Increase ambient humidity.",
    "Target_Spot":            "Apply fungicide (azoxystrobin or chlorothalonil). Avoid overhead irrigation. Rotate crops.",
    "mosaic_virus":           "No chemical cure. Remove and destroy infected plants. Control aphid vectors.",
    "YellowLeaf__Curl_Virus": "No chemical cure. Remove infected plants immediately. Control whitefly vectors.",
    "healthy":                "No treatment needed — plant appears healthy!",
    "Healthy":                "No treatment needed — plant appears healthy!",
}


# ── Model loader (cached — loaded only once per session) ──────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_meta():
    """
    Tries to load the TF SavedModel and metadata.json.
    Returns (model, class_names, img_size, demo_mode).
    If the model file is not found, returns (None, fallback_classes, 224, True)
    so the app still loads in demo mode on Streamlit Cloud.
    """
    meta_path  = Path(METADATA_PATH)
    model_path = Path(MODEL_PATH)

    # Load class names from metadata if available, else use fallback
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        class_names = meta["class_names"]
        img_size    = meta["img_size"]
    else:
        class_names = FALLBACK_CLASSES
        img_size    = 224

    if not model_path.exists():
        return None, class_names, img_size, True  # demo mode

    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))
    return model, class_names, img_size, False


def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """
    Applies EfficientNetV2S preprocess_input — matches data_preprocessing.py.
    This is the correct preprocessing for the trained model.
    """
    import tensorflow as tf
    img = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr


def get_treatment(class_name: str) -> str:
    for key, treatment in TREATMENTS.items():
        if key.lower() in class_name.lower():
            return treatment
    return "Consult a local agronomist for specific treatment advice."


# ── Load model ────────────────────────────────────────────────────────────────

model, class_names, img_size, demo_mode = load_model_and_meta()

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌿 Plant Disease Detection")
st.markdown(
    "Upload a leaf image to detect plant diseases using "
    "**EfficientNetV2S** transfer learning."
)

if demo_mode:
    st.warning(
        "⚠️ **Demo mode** — trained model not deployed on this server. "
        "Predictions shown are illustrative only. "
        "To run with the real model, clone the repo and run locally after training."
    )
else:
    st.info(
        f"🔬 **Model:** EfficientNetV2S  |  "
        f"**Classes:** {len(class_names)}  |  "
        f"**Input size:** {img_size}×{img_size}"
    )

uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, close-up photo of a single plant leaf"
)

if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large. Please upload an image smaller than 10 MB.")
        st.stop()

    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.markdown("### Analysis")

        if demo_mode:
            # Demo mode — show random plausible prediction
            with st.spinner("Analysing leaf..."):
                import time
                time.sleep(1.0)

            predicted_idx   = np.random.randint(0, len(class_names))
            confidence      = np.random.uniform(0.78, 0.97)
            predicted_class = class_names[predicted_idx]

            top3_idx = np.random.choice(len(class_names), 3, replace=False)
            top3_idx[0] = predicted_idx
            probs    = np.random.dirichlet(np.ones(3)) * 0.3
            probs[0] = confidence
            top3     = [(class_names[top3_idx[i]], float(probs[i])) for i in range(3)]

        else:
            # Real model inference
            with st.spinner("Analysing leaf..."):
                arr   = preprocess_image(image, img_size)
                preds = model.predict(arr, verbose=0)[0]

            predicted_idx   = int(np.argmax(preds))
            confidence      = float(preds[predicted_idx])
            predicted_class = class_names[predicted_idx]
            top3_idx        = np.argsort(preds)[::-1][:3]
            top3            = [(class_names[i], float(preds[i])) for i in top3_idx]

        # Display results
        if confidence < CONFIDENCE_THRESHOLD and not demo_mode:
            st.warning(
                f"⚠️ Low confidence ({confidence:.1%}). "
                "The image may not be a recognisable plant leaf."
            )
        else:
            display_name = predicted_class.replace("_", " ").replace("  ", " ")
            st.success(f"**Detected:** {display_name}")
            st.metric("Confidence", f"{confidence:.1%}")

        st.markdown("### Top 3 predictions")
        for cls, prob in top3:
            display = cls.replace("_", " ").replace("  ", " ")
            st.progress(float(min(prob, 1.0)), text=f"{display}: {prob:.1%}")

        st.markdown("### 💊 Recommended treatment")
        treatment = get_treatment(predicted_class)
        if "healthy" in predicted_class.lower():
            st.success(treatment)
        else:
            st.info(treatment)

st.divider()
st.markdown(
    "Built by [Komal Kumari](https://github.com/Komal036) · "
    "[GitHub Repo](https://github.com/Komal036/Plant-disease-detection)"
)
