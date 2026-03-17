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

METADATA_PATH   = "models/metadata.json"
MODEL_PATH      = "models/saved/final_model"
CONFIDENCE_THRESHOLD = 0.60

TREATMENTS = {
    "Bacterial_spot":       "Apply copper-based bactericide. Remove and destroy infected leaves. Avoid overhead watering.",
    "Early_blight":         "Apply fungicide containing chlorothalonil or mancozeb. Improve air circulation. Remove lower infected leaves.",
    "Late_blight":          "Apply mancozeb or metalaxyl fungicide immediately. Destroy infected plants. Avoid leaf wetness.",
    "Leaf_Mold":            "Reduce humidity and improve ventilation. Apply fungicide (chlorothalonil) if severe.",
    "Septoria_leaf_spot":   "Remove infected leaves. Apply fungicide (chlorothalonil or copper). Avoid overhead irrigation.",
    "Spider_mites":         "Apply miticide or neem oil spray. Increase ambient humidity. Introduce predatory mites if available.",
    "Target_Spot":          "Apply fungicide (azoxystrobin or chlorothalonil). Avoid overhead irrigation. Rotate crops.",
    "mosaic_virus":         "No chemical cure. Remove and destroy infected plants. Control aphid vectors with insecticide.",
    "YellowLeaf__Curl_Virus": "No chemical cure. Remove infected plants immediately. Control whitefly vectors.",
    "healthy":              "No treatment needed — plant appears healthy!",
    "Healthy":              "No treatment needed — plant appears healthy!",
}


# ── Model loader (cached — loaded only once per session) ──────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_meta():
    """
    Loads the TF SavedModel and metadata.json.
    Cached by Streamlit so the model is only loaded once per session.
    Returns (model, class_names, img_size) or raises an error with a
    helpful message if the model file is not found.
    """
    import tensorflow as tf

    meta_path  = Path(METADATA_PATH)
    model_path = Path(MODEL_PATH)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at '{METADATA_PATH}'. "
            "Run training first: python main.py --mode train"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Run training first: python main.py --mode train"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    model = tf.keras.models.load_model(str(model_path))
    return model, meta["class_names"], meta["img_size"]


def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """
    Applies the same preprocessing as data_preprocessing.py:
    EfficientNetV2S preprocess_input (NOT simple /255 normalisation).
    FIX: The original app used arr / 255.0 which does not match how the
    model was trained. This function uses the correct preprocessing.
    """
    import tensorflow as tf
    img = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr


def get_treatment(class_name: str) -> str:
    """
    Returns the treatment recommendation for a predicted class name.
    Matches by substring so minor class name variations still resolve.
    """
    for key, treatment in TREATMENTS.items():
        if key.lower() in class_name.lower():
            return treatment
    return "Consult a local agronomist for specific treatment advice."


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌿 Plant Disease Detection")
st.markdown(
    "Upload a leaf image to detect plant diseases using "
    "**EfficientNetV2S** transfer learning."
)

# Try to load model — show a clear error if not available
model_loaded = False
model, class_names, img_size = None, [], 224

try:
    model, class_names, img_size = load_model_and_meta()
    model_loaded = True
    st.info(
        f"🔬 **Model:** EfficientNetV2S  |  "
        f"**Classes:** {len(class_names)}  |  "
        f"**Input size:** {img_size}×{img_size}"
    )
except FileNotFoundError as e:
    st.error(f"⚠️ Model not loaded: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, close-up photo of a single plant leaf"
)

if uploaded_file is not None:
    # File size guard
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large. Please upload an image smaller than 10 MB.")
        st.stop()

    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.markdown("### Analysis")
        with st.spinner("Analysing leaf..."):
            arr   = preprocess_image(image, img_size)
            preds = model.predict(arr, verbose=0)[0]

        predicted_idx  = int(np.argmax(preds))
        confidence     = float(preds[predicted_idx])
        predicted_class = class_names[predicted_idx]

        # Top-3 predictions
        top3_idx = np.argsort(preds)[::-1][:3]
        top3     = [(class_names[i], float(preds[i])) for i in top3_idx]

        # Confidence threshold check
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence ({confidence:.1%}). "
                "The image may not be a recognisable leaf, "
                "or the disease may not be in the training set."
            )
        else:
            # Clean display name (replace underscores)
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
