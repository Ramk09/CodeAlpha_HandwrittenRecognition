
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"

st.set_page_config(
    page_title="AI Digit Lab | Handwriting Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #0b1020;
    --panel: #131b30;
    --border: #293550;
    --purple: #a78bfa;
    --cyan: #67e8f9;
    --muted: #9caac5;
}

.stApp {
    background: radial-gradient(ellipse at top right, #202047 0%, #0b1020 48%);
    color: #f3f5ff;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0e1629;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f5f3ff !important;
}

.hero {
    padding: 32px;
    border: 1px solid #34365f;
    border-radius: 24px;
    background: linear-gradient(125deg, #1c2543, #30235a 65%, #172b46);
    margin: 10px 0 24px;
}

.eyebrow {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 48px);
    margin: 12px 0;
}

.hero p {
    color: #c6cde2;
    font-size: 16px;
    max-width: 700px;
}

.metric-card {
    background: rgba(19, 27, 48, 0.9);
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 16px;
    min-height: 110px;
}

.metric-label {
    color: var(--muted);
    font-size: 13px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}

.panel {
    background: rgba(19, 27, 48, 0.72);
    border: 1px solid var(--border);
    padding: 22px;
    border-radius: 18px;
    margin: 12px 0;
}

.prediction {
    background: linear-gradient(135deg, #22264c, #182d43);
    border: 1px solid #5955a2;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
}

.big-digit {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 82px;
    line-height: 1.2;
    font-weight: 700;
    color: var(--cyan);
}

.small-muted {
    color: var(--muted);
    font-size: 13px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #8b5cf6, #6366f1);
    border: 0;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    min-height: 48px;
}

div.stButton > button[kind="primary"]:hover {
    border: 1px solid #c4b5fd;
    color: white;
}

[data-testid="stFileUploader"] {
    background: rgba(19, 27, 48, 0.7);
    border: 1px dashed #6263a0;
    border-radius: 16px;
    padding: 12px;
}

hr {
    border-color: var(--border);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


def prepare_digit(crop):
    """Prepare an image as 784 normalized MNIST-style pixel values."""
    image = Image.fromarray(crop.astype(np.uint8)).convert("L")

    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, 784)


def predict_crop(crop):
    features = prepare_digit(crop)
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))

    return (
        int(model.classes_[best_index]),
        float(probabilities[best_index]),
        probabilities,
    )


def split_grid(image):
    """Split a regular image into two rows and five columns."""
    height, width = image.shape[:2]
    cells = []

    for row in range(2):
        for col in range(5):
            x1 = round(col * width / 5)
            x2 = round((col + 1) * width / 5)
            y1 = round(row * height / 2)
            y2 = round((row + 1) * height / 2)

            pad_x = max(1, round((x2 - x1) * 0.04))
            pad_y = max(1, round((y2 - y1) * 0.04))

            crop = image[
                y1 + pad_y:y2 - pad_y,
                x1 + pad_x:x2 - pad_x,
            ]

            if crop.size:
                cells.append(crop)

    return cells


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## 🔢 AI DIGIT LAB")
    st.caption("HANDWRITING INTELLIGENCE")
    st.divider()

    mode = st.radio(
        "RECOGNITION MODE",
        ["Single Digit", "10-Digit Grid"],
        help="Grid mode expects exactly 2 rows and 5 columns.",
    )

    st.divider()
    st.markdown("### About this project")
    st.write(
        "A handwritten-digit classifier built with Python, "
        "Scikit-learn and the MNIST dataset."
    )
    st.caption("Model: MLPClassifier")
    st.caption("Input: 28 × 28 grayscale pixels")
    st.caption("Classes: digits 0–9")


st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ Machine Learning Playground</div>
    <h1>Turn handwriting into intelligence.</h1>
    <p>Upload a handwritten digit and let the trained neural network
    analyze the image, predict the number, and reveal its confidence.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("DIGIT CLASSES", "10")
with col2:
    metric_card("MODEL INPUT", "28 × 28")
with col3:
    metric_card("PREDICTION", "Instant")

st.markdown("## Workspace")
st.write("Upload an image to begin your recognition session.")

uploaded_file = st.file_uploader(
    "Choose a handwritten image",
    type=["png", "jpg", "jpeg"],
    help="Use a clear, high-contrast image for the best results.",
)

if mode == "Single Digit":
    st.caption("Single-digit mode works best with one large digit centered in the image.")
else:
    st.info(
        "Grid mode divides the image into equal cells. Upload a regular "
        "grid containing exactly 2 rows and 5 columns, one digit per cell."
    )

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.asarray(pil_image)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Your image")
            st.image(pil_image, use_container_width=True)
            st.caption(f"Image dimensions: {pil_image.width} × {pil_image.height}")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("### 🧠 Recognition engine")

            if mode == "Single Digit":
                st.write("Analyze one handwritten digit.")
                run_prediction = st.button(
                    "✦ Analyze digit",
                    type="primary",
                    use_container_width=True,
                )

                if run_prediction:
                    with st.spinner("Analyzing handwriting..."):
                        digit, confidence, probabilities = predict_crop(image_array)

                    st.markdown(
                        f'<div class="prediction">'
                        f'<div class="small-muted">PREDICTED DIGIT</div>'
                        f'<div class="big-digit">{digit}</div>'
                        f'<div>Estimated probability: {confidence:.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Probability by digit")
                    probability_data = pd.DataFrame({
                        "Digit": [str(int(label)) for label in model.classes_],
                        "Probability": probabilities,
                    }).set_index("Digit")

                    st.bar_chart(probability_data)
                    st.caption(
                        "Probabilities are model estimates, not guarantees "
                        "of correctness."
                    )

            else:
                st.write("Analyze all ten cells in reading order.")
                run_prediction = st.button(
                    "✦ Analyze all 10 digits",
                    type="primary",
                    use_container_width=True,
                )

                if run_prediction:
                    with st.spinner("Analyzing all grid cells..."):
                        cells = split_grid(image_array)

                        if len(cells) != 10:
                            st.error(f"Expected 10 cells; found {len(cells)}.")
                        else:
                            results = []

                            for index, cell in enumerate(cells):
                                digit, confidence, _ = predict_crop(cell)
                                results.append({
                                    "Cell": index + 1,
                                    "Position": (
                                        f"Row {index // 5 + 1}, "
                                        f"Column {index % 5 + 1}"
                                    ),
                                    "Predicted digit": digit,
                                    "Confidence": f"{confidence:.1%}",
                                })

                            sequence = "".join(
                                str(item["Predicted digit"]) for item in results
                            )

                            st.markdown(
                                f'<div class="prediction">'
                                f'<div class="small-muted">PREDICTED SEQUENCE</div>'
                                f'<div class="big-digit" style="font-size:48px">'
                                f'{sequence}</div>'
                                f'<div>10 cells analyzed</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.markdown("#### Individual grid cells")
                            for start in range(0, len(cells), 5):
                                crop_columns = st.columns(5)
                                for offset, cell in enumerate(
                                    cells[start:start + 5]
                                ):
                                    with crop_columns[offset]:
                                        st.image(
                                            cell,
                                            caption=f"Cell {start + offset + 1}",
                                            use_container_width=True,
                                        )

                            st.caption(
                                "The grid is split into equal-sized cells; "
                                "this is not automatic character detection."
                            )

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Could not process this image: {error}")

else:
    st.markdown("""
    <div class="panel">
        <h3>✨ Ready when you are</h3>
        <p>Upload a PNG or JPG image above. For best results, use a
        clear digit with a plain background and minimal extra marks.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p style="text-align:center;color:#9caac5;font-size:12px;">'
    'AI DIGIT LAB · MNIST · SCIKIT-LEARN · BUILT WITH STREAMLIT'
    '</p>',
    unsafe_allow_html=True,
)
