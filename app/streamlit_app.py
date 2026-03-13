# app/streamlit_app.py
import logging
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR    = 'models/saved/final_model'
META_PATH    = 'models/metadata.json'
CONF_THRESH  = 0.60   # below this → "uncertain" response


# ── Model loading (cached — runs only once per session) ───────────────────────

@st.cache_resource
def load_model_and_metadata():
    """
    Loads the SavedModel and class metadata once and caches them.
    Using @st.cache_resource prevents reloading on every user interaction.
    """
    try:
        model = tf.keras.models.load_model(MODEL_DIR)
        with open(META_PATH) as f:
            meta = json.load(f)
        logger.info(
            f"Model loaded. Classes: {meta['num_classes']}, "
            f"Input size: {meta['img_size']}"
        )
        return model, meta['class_names'], meta['img_size']
    except FileNotFoundError as e:
        logger.error(f'Model or metadata not found: {e}')
        st.error(
            '⚠️ Model files not found.\n\n'
            'Run `python main.py --mode train` first, '
            'then restart the app.'
        )
        st.stop()
    except Exception as e:
        logger.error(f'Unexpected error loading model: {e}')
        st.error(f'Failed to load model: {e}')
        st.stop()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, img_size: int) -> np.ndarray:
    """
    Converts a PIL image to a normalised numpy array ready for inference.
    Shape: (1, img_size, img_size, 3), dtype float32, range [0, 1].
    """
    image = image.convert('RGB')
    image = image.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, img_array: np.ndarray, class_names: list):
    """
    Runs inference and returns (label, confidence).
    If confidence < CONF_THRESH, returns a low-confidence message.
    """
    preds      = model.predict(img_array, verbose=0)[0]
    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])

    if confidence < CONF_THRESH:
        return 'Uncertain — please upload a clearer leaf image', confidence

    return class_names[idx], confidence


def get_top3(preds: np.ndarray, class_names: list):
    """Returns top-3 (class_name, probability) tuples."""
    top3_idx = np.argsort(preds)[::-1][:3]
    return [(class_names[i], float(preds[i])) for i in top3_idx]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title='Plant Disease Detector',
        page_icon='🌿',
        layout='centered'
    )

    # Header
    st.title('🌿 Plant Disease Detector')
    st.markdown(
        'Upload a **clear, close-up photo of a single plant leaf** to '
        'identify diseases using deep learning.'
    )
    st.divider()

    # Load model
    model, class_names, img_size = load_model_and_metadata()

    # Sidebar info
    with st.sidebar:
        st.header('ℹ️ About')
        st.write(f'**Model:** EfficientNetV2S')
        st.write(f'**Classes:** {len(class_names)}')
        st.write(f'**Input size:** {img_size}×{img_size}')
        st.divider()
        st.subheader('Supported Plants')
        for name in class_names:
            st.write(f'• {name}')

    # File uploader
    uploaded_file = st.file_uploader(
        'Choose a leaf image',
        type=['jpg', 'jpeg', 'png'],
        help='Upload a JPG or PNG image of a plant leaf'
    )

    if uploaded_file is not None:
        # Validate file size (max 10 MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error('File too large. Please upload an image smaller than 10 MB.')
            return

        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f'Could not open image: {e}')
            return

        # Validate minimum resolution
        w, h = image.size
        if w < 50 or h < 50:
            st.warning('Image resolution is very low. Results may be inaccurate.')

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption=f'Uploaded: {uploaded_file.name}',
                     use_column_width=True)

        with col2:
            with st.spinner('Analysing leaf...'):
                img_array = preprocess_image(image, img_size)
                preds     = model.predict(img_array, verbose=0)[0]
                label, confidence = predict(model, img_array, class_names)

            if confidence >= CONF_THRESH:
                st.success(f'**Prediction:** {label}')
                st.metric('Confidence', f'{confidence:.1%}')

                st.subheader('Top 3 Predictions')
                for cls, prob in get_top3(preds, class_names):
                    st.progress(prob, text=f'{cls}: {prob:.1%}')
            else:
                st.warning(f'⚠️ {label}')
                st.metric('Confidence', f'{confidence:.1%}')
                st.info(
                    'Tips for better results:\n'
                    '- Use a close-up photo of a single leaf\n'
                    '- Ensure good lighting\n'
                    '- Avoid blurry or dark images'
                )

    st.divider()
    st.caption('Built with TensorFlow & Streamlit · PlantVillage Dataset')


if __name__ == '__main__':
    main()
