# Lightweight chest X-ray demo deployment guide

This demo is optimized for inference-only deployment. It keeps the model in memory, uses a compact Streamlit UI, and shows both prediction confidence and Grad-CAM overlays.

## Local demo

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo locally:

```bash
streamlit run streamlit_app.py
```

4. Upload a chest X-ray image and review:
   - predicted class
   - confidence score
   - Grad-CAM visualization

## Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select **Streamlit** as the Space SDK.
3. Upload or connect this repository.
4. The Space will use `app.py` as the entry point.
5. Install dependencies from `requirements.txt` automatically in the Space environment.

### Recommended Space settings

- Runtime: CPU
- Python version: 3.10 or 3.11
- Space type: Streamlit

### Environment notes

- The app loads `model.best4.keras` from the repository root.
- If you replace the model, keep the same filename or set `MODEL_PATH` in the Space environment.
- The demo is lightweight because it avoids extra APIs, databases, and training workflows.

## Troubleshooting

- If the model fails to load, confirm the file exists at the repository root.
- If Grad-CAM fails, make sure the model includes nested convolutional layers and the app uses the updated inference pipeline.
- For slower cold starts, keep the model file small and avoid unnecessary dependencies.
