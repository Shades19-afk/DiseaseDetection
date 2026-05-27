import os

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import load_model, predict_image


@st.cache_resource(show_spinner="Loading chest X-ray demo model...")
def load_demo_model(model_path: str):
    return load_model(model_path)


def main():
    st.set_page_config(
        page_title="Chest X-ray ML Demo",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Chest X-ray ML Demo")
    st.caption("Upload a chest X-ray image for a lightweight pneumonia screening demo with Grad-CAM explainability.")

    model_path = os.getenv('MODEL_PATH', 'model.best4.keras')
    model = load_demo_model(model_path)

    uploaded_file = st.file_uploader(
        'Upload a chest X-ray image',
        type=['png', 'jpg', 'jpeg'],
        help='Supported formats: PNG, JPG, JPEG',
    )

    if uploaded_file is not None:
        result = predict_image(uploaded_file, model)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader('Original image')
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption='Uploaded X-ray', use_container_width=True)

        with col_right:
            st.subheader('Explainability view')
            st.image(result.overlay, caption=f'Grad-CAM overlay — {result.label}', use_container_width=True)

        st.subheader('Prediction summary')
        st.metric('Predicted class', result.label)
        st.metric('Confidence', f"{result.confidence:.1%}")

        st.markdown('### Class probabilities')
        st.write({f'Class {idx}': float(prob) for idx, prob in enumerate(np.asarray(result.raw_probabilities))})

        st.info(
            'The heatmap highlights regions that most influenced the prediction. In a real screening workflow, this should be reviewed alongside clinical context.'
        )
    else:
        st.info('Upload an image to start the demo. The model is preloaded and optimized for lightweight inference.')
        st.markdown(
            """
            ### What this demo shows
            - Upload a chest X-ray image.
            - The model returns a predicted class and confidence score.
            - Grad-CAM overlays highlight the image regions contributing most to the prediction.
            """
        )


if __name__ == '__main__':
    main()
