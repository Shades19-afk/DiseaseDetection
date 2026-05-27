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
        col_metric_1, col_metric_2 = st.columns(2)
        with col_metric_1:
            st.metric('Predicted class', result.label)
        with col_metric_2:
            st.metric('Confidence', f"{result.confidence:.1%}")

        st.markdown('### Class probabilities')
        st.write({f'Class {idx}': float(prob) for idx, prob in enumerate(np.asarray(result.raw_probabilities))})

        st.divider()

        st.markdown('### How the AI Interprets the X-ray')
        st.markdown(
            """
            **Grad-CAM Visualization:**
            The heatmap overlay shows which regions of the X-ray image the AI model focused on when making its prediction.
            
            **Color interpretation:**
            - 🔴 **Red/Yellow regions** indicate high attention — the model found these areas most important for the prediction
            - 🔵 **Blue regions** indicate low attention — the model paid less attention to these areas
            
            **What it means:**
            For a pneumonia prediction, the model highlights lung regions where it detected patterns similar to pneumonia cases from its training data. 
            These patterns typically include areas of opacity or consolidation in the lungs.
            """
        )

        st.info(
            '⚠️ **Important:** This visualization is an interpretability tool for understanding the model\'s reasoning, not a medical diagnosis. '
            'Real clinical decisions should always involve qualified medical professionals who review the full patient context.'
        )

        st.markdown(
            """
            **How the model works:**
            The AI model predicts pneumonia when it detects visual patterns in the X-ray that are similar to pneumonia cases it learned during training. 
            The Grad-CAM heatmap helps visualize which parts of the image triggered this prediction.
            """
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
