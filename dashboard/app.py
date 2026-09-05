"""
BrainLens AI
Brain MRI Classification & Explainable AI
Research Prototype — Not for Clinical Diagnosis

Single-file Streamlit application.
"""

import io
import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
import tensorflow as tf

# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = os.path.join(
    "models", "final_densenet121", "FINAL_DENSENET121_BRISC2025.keras"
)

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary",
}

INPUT_SIZE = 224
GRAD_CAM_LAYER_NAME = "conv5_block16_2_conv"

TEST_METRICS = {
    "Accuracy": 97.20,
    "Balanced Accuracy": 97.44,
    "Macro Precision": 97.53,
    "Macro Recall": 97.44,
    "Macro F1": 97.47,
    "Weighted F1": 97.19,
    "Macro AUC": 99.79,
}

PER_CLASS_METRICS = pd.DataFrame(
    [
        ["Glioma", 97.94, 93.70, 95.77],
        ["Meningioma", 94.87, 96.73, 95.79],
        ["No Tumor", 99.29, 100.00, 99.64],
        ["Pituitary", 98.03, 99.33, 98.68],
    ],
    columns=["Class", "Precision", "Recall", "F1"],
)

CONFUSION_MATRIX = pd.DataFrame(
    [
        [238, 15, 0, 1],
        [4, 296, 1, 5],
        [0, 0, 140, 0],
        [1, 1, 0, 298],
    ],
    index=["True_glioma", "True_meningioma", "True_no_tumor", "True_pituitary"],
    columns=["Pred_glioma", "Pred_meningioma", "Pred_no_tumor", "Pred_pituitary"],
)

MODEL_INFO_ROWS = [
    ("Model", "DenseNet121"),
    ("Dataset", "BRISC2025"),
    ("Input", "224 × 224 × 3 RGB"),
    ("Classes", "4"),
    ("Parameters", "7,305,028"),
    ("Transfer learning", "ImageNet"),
    ("Fine-tuning", "Yes"),
    ("Explainability", "Grad-CAM"),
    ("Inference", "Single-image"),
]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BrainLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS INJECTION
# ============================================================

CUSTOM_CSS = """
<style>
:root {
    --bl-page: #07111F;
    --bl-sidebar: #0B1728;
    --bl-card: #101E31;
    --bl-card-elevated: #14263D;
    --bl-border: #223854;
    --bl-blue: #3BA7FF;
    --bl-teal: #20D6C7;
    --bl-text: #F5F9FF;
    --bl-text-secondary: #9FB2C8;
    --bl-text-muted: #6F849C;
    --bl-warning: #F4B942;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bl-page);
    color: var(--bl-text);
}

[data-testid="stHeader"] {
    background-color: transparent;
}

[data-testid="stSidebar"] {
    background-color: var(--bl-sidebar);
    border-right: 1px solid var(--bl-border);
}

[data-testid="stSidebar"] * {
    color: var(--bl-text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Hero */
.bl-hero {
    background: linear-gradient(135deg, #0C1E36 0%, #0A1729 60%, #081120 100%);
    border: 1px solid var(--bl-border);
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 2rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.bl-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--bl-teal);
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.bl-hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--bl-text);
    margin: 0 0 0.3rem 0;
}
.bl-hero-sub {
    font-size: 1rem;
    color: var(--bl-text-secondary);
    margin-bottom: 1rem;
    max-width: 640px;
    line-height: 1.5;
}
.bl-pill {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--bl-blue);
    background: rgba(59,167,255,0.10);
    border: 1px solid rgba(59,167,255,0.35);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
}

/* Section label */
.bl-section-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--bl-text-muted);
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.bl-section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--bl-text);
    margin: 0 0 0.9rem 0;
}

/* Workflow steps */
.bl-workflow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.bl-step {
    flex: 1;
    min-width: 110px;
    text-align: center;
    padding: 0.7rem 0.4rem;
    border-radius: 10px;
    border: 1px solid var(--bl-border);
    background: var(--bl-card);
}
.bl-step-active {
    border-color: var(--bl-blue);
    background: rgba(59,167,255,0.08);
}
.bl-step-done {
    border-color: var(--bl-teal);
    background: rgba(32,214,199,0.07);
}
.bl-step-num {
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    color: var(--bl-text-muted);
    text-transform: uppercase;
    font-weight: 700;
}
.bl-step-label {
    font-size: 0.82rem;
    color: var(--bl-text-secondary);
    font-weight: 600;
    margin-top: 0.15rem;
}
.bl-step-arrow {
    color: var(--bl-text-muted);
    font-size: 1rem;
    padding: 0 0.2rem;
}

/* Cards */
.bl-card {
    background: var(--bl-card);
    border: 1px solid var(--bl-border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
}
.bl-card-elevated {
    background: var(--bl-card-elevated);
    border: 1px solid var(--bl-border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
}

/* Empty state */
.bl-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--bl-text-secondary);
}
.bl-empty-icon {
    font-size: 2.4rem;
    margin-bottom: 0.6rem;
}
.bl-empty-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--bl-text);
    margin-bottom: 0.3rem;
}

/* Prediction card */
.bl-pred-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--bl-teal);
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.bl-pred-class {
    font-size: 2rem;
    font-weight: 800;
    color: var(--bl-text);
    margin-bottom: 0.4rem;
}
.bl-pred-conf-label {
    font-size: 0.75rem;
    color: var(--bl-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.bl-pred-conf-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--bl-blue);
}
.bl-pred-note {
    color: var(--bl-text-secondary);
    font-size: 0.88rem;
    margin-top: 0.8rem;
    line-height: 1.5;
}

/* Probability bars */
.bl-prob-row {
    margin-bottom: 0.75rem;
}
.bl-prob-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--bl-text-secondary);
    margin-bottom: 0.25rem;
}
.bl-prob-top-active {
    color: var(--bl-text);
    font-weight: 700;
}
.bl-prob-track {
    background: #0B1728;
    border: 1px solid var(--bl-border);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.bl-prob-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--bl-blue), var(--bl-teal));
}

/* Disclaimer */
.bl-disclaimer {
    border: 1px solid rgba(244,185,66,0.35);
    background: rgba(244,185,66,0.07);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin: 2rem 0 1.2rem 0;
}
.bl-disclaimer-title {
    color: var(--bl-warning);
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 0.3rem;
}
.bl-disclaimer-text {
    color: var(--bl-text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Footer */
.bl-footer {
    text-align: center;
    color: var(--bl-text-muted);
    font-size: 0.78rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--bl-border);
    line-height: 1.6;
}

/* Sidebar internals */
.bl-side-brand-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--bl-text);
    margin-bottom: 0.1rem;
}
.bl-side-brand-sub {
    font-size: 0.75rem;
    color: var(--bl-text-secondary);
    line-height: 1.4;
    margin-bottom: 0.5rem;
}
.bl-side-status-dot {
    color: var(--bl-teal);
}
.bl-side-heading {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--bl-text-muted);
    font-weight: 700;
    margin: 1.1rem 0 0.5rem 0;
}
.bl-side-divider {
    border-top: 1px solid var(--bl-border);
    margin: 0.9rem 0;
}
.bl-side-nav-item {
    font-size: 0.86rem;
    color: var(--bl-text-secondary);
    padding: 0.25rem 0;
}
.bl-side-nav-active {
    color: var(--bl-text);
    font-weight: 700;
}
.bl-side-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.2rem 0;
}
.bl-side-metric-value {
    color: var(--bl-teal);
    font-weight: 700;
    font-size: 0.95rem;
}
.bl-side-metric-label {
    color: var(--bl-text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.bl-side-system-row {
    font-size: 0.8rem;
    color: var(--bl-text-secondary);
    padding: 0.15rem 0;
}
.bl-side-footer-note {
    font-size: 0.72rem;
    color: var(--bl-text-muted);
    line-height: 1.5;
    margin-top: 0.6rem;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, var(--bl-blue), #2E8FE0);
    color: #061323;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
}
div.stButton > button:hover {
    filter: brightness(1.08);
    color: #061323;
}

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bl-card);
    border: 1.5px dashed var(--bl-border);
    border-radius: 14px;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bl-card);
    border: 1px solid var(--bl-border);
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--bl-border);
    border-radius: 10px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "uploaded_file_bytes": None,
        "uploaded_file_name": None,
        "analysis_done": False,
        "predicted_class": None,
        "confidence": None,
        "probabilities": None,
        "display_image": None,
        "gradcam_heatmap": None,
        "gradcam_overlay": None,
        "show_overlay": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


@st.cache_resource(show_spinner=False)
def find_backbone_layer(_model):
    """Locate the nested DenseNet121 backbone layer inside the outer model."""
    for layer in _model.layers:
        if hasattr(layer, "get_layer"):
            try:
                layer.get_layer(GRAD_CAM_LAYER_NAME)
                return layer
            except ValueError:
                continue
    return None


model = load_model()
backbone_layer = find_backbone_layer(model) if model is not None else None
gpu_available = len(tf.config.list_physical_devices("GPU")) > 0

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(pil_image: Image.Image, target_size: int = INPUT_SIZE):
    """
    Reproduces the original training-time preprocessing exactly:
    RGB conversion -> aspect-ratio preserving resize (LANCZOS) ->
    symmetric black padding to target_size -> float32 [0,1] normalization.
    """
    img = pil_image.convert("RGB")
    width, height = img.size
    scale = target_size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))

    array = np.array(padded).astype(np.float32) / 255.0
    return padded, array


# ============================================================
# INFERENCE
# ============================================================

def run_inference(input_array: np.ndarray):
    batch = np.expand_dims(input_array, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    return class_idx, preds


# ============================================================
# GRAD-CAM (nested-backbone, logit-based)
# ============================================================

def generate_gradcam(input_array: np.ndarray, class_idx: int):
    """
    CPU-safe, logit-based Grad-CAM.

    Normal model inference remains on the GPU. Grad-CAM is executed on
    CPU because TensorFlow 2.10 + cuDNN on the RTX 3050 can fail during
    Conv2D backward operations used by GradientTape.
    """
    if backbone_layer is None:
        return None

    try:
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(input_array, axis=0),
            dtype=tf.float32
        )

        # Run the complete Grad-CAM graph on CPU to avoid the cuDNN
        # Conv2DBackpropFilter failure encountered on the GPU.
        with tf.device("/CPU:0"):
            conv_layer = backbone_layer.get_layer(GRAD_CAM_LAYER_NAME)

            feature_extractor = tf.keras.Model(
                inputs=backbone_layer.input,
                outputs=[conv_layer.output, backbone_layer.output],
            )

            outer_layers = [layer for layer in model.layers if layer is not backbone_layer]
            last_layer = outer_layers[-1]
            intermediate_layers = outer_layers[:-1]

            with tf.GradientTape() as tape:
                conv_output, backbone_output = feature_extractor(
                    input_tensor,
                    training=False
                )

                tape.watch(conv_output)

                x = backbone_output

                for layer in intermediate_layers:
                    x = layer(x, training=False)

                # The final classifier is Dense(4, softmax).
                # Reconstruct its pre-softmax logits so Grad-CAM is based
                # on the target logit rather than the saturated probability.
                if isinstance(x, tf.Tensor) and x.shape.rank and x.shape.rank > 2:
                    x = tf.keras.layers.Flatten()(x)

                logits = tf.matmul(x, last_layer.kernel) + last_layer.bias
                target_logit = logits[:, class_idx]

            grads = tape.gradient(target_logit, conv_output)

            if grads is None:
                return None

            pooled_grads = tf.reduce_mean(
                grads,
                axis=(0, 1, 2)
            )

            conv_output = conv_output[0]

            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0)
            max_value = tf.reduce_max(heatmap)

            heatmap = heatmap / (max_value + 1e-8)

            return heatmap.numpy()

    except Exception as exc:
        print(f"Grad-CAM generation failed: {exc}")
        return None

def colorize_heatmap(heatmap: np.ndarray, target_size: int = INPUT_SIZE):
    heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (target_size, target_size), Image.BILINEAR
    )
    heatmap_arr = np.array(heatmap_resized) / 255.0
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap_arr)[:, :, :3]
    colored_img = Image.fromarray(np.uint8(colored * 255))
    return colored_img


def overlay_heatmap(base_image: Image.Image, colored_heatmap: Image.Image, alpha: float = 0.42):
    base_rgba = base_image.convert("RGBA")
    heat_rgba = colored_heatmap.convert("RGBA")
    blended = Image.blend(base_rgba, heat_rgba, alpha)
    return blended.convert("RGB")


# ============================================================
# REPORT GENERATION
# ============================================================

def build_report_text(filename, predicted_class, confidence, probabilities):
    lines = []
    lines.append("BrainLens AI")
    lines.append("Brain MRI Classification & Explainable AI")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Filename: {filename}")
    lines.append("")
    lines.append(f"Predicted class: {CLASS_DISPLAY[predicted_class]}")
    lines.append(f"Prediction confidence: {confidence * 100:.2f}%")
    lines.append("")
    lines.append("Class probabilities:")
    for cls, prob in zip(CLASS_NAMES, probabilities):
        lines.append(f"  {CLASS_DISPLAY[cls]}: {prob * 100:.2f}%")
    lines.append("")
    lines.append("Model: DenseNet121")
    lines.append("Input: 224 x 224 x 3")
    lines.append("Preprocessing:")
    lines.append("  RGB conversion")
    lines.append("  Aspect-ratio preserving resize (LANCZOS)")
    lines.append("  Symmetric black padding")
    lines.append("  Normalization to [0,1]")
    lines.append(f"Grad-CAM target layer: {GRAD_CAM_LAYER_NAME}")
    lines.append("")
    lines.append("-" * 50)
    lines.append("Research Prototype — Not for Clinical Diagnosis")
    lines.append(
        "BrainLens AI is developed for research and educational purposes. "
        "Model predictions and Grad-CAM visualizations should not be used "
        "for medical diagnosis, treatment decisions, or patient management."
    )
    return "\n".join(lines)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="bl-side-brand-title">🧠 BrainLens AI</div>
        <div class="bl-side-brand-sub">Brain MRI Classification<br/>&amp; Explainable AI</div>
        <div class="bl-side-nav-item"><span class="bl-side-status-dot">●</span> Research Prototype</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bl-side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-side-heading">Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bl-side-nav-item bl-side-nav-active">◉ Analyze MRI</div>
        <div class="bl-side-nav-item">○ Explainability</div>
        <div class="bl-side-nav-item">○ Model Overview</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bl-side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-side-heading">Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bl-side-nav-item">DenseNet121</div>
        <div class="bl-side-nav-item">BRISC2025</div>
        <div class="bl-side-nav-item">224 × 224 × 3</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bl-side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-side-heading">Test Performance</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="bl-side-metric-row">
            <span class="bl-side-metric-value">{TEST_METRICS['Accuracy']:.2f}%</span>
            <span class="bl-side-metric-label">Accuracy</span>
        </div>
        <div class="bl-side-metric-row">
            <span class="bl-side-metric-value">{TEST_METRICS['Macro F1']:.2f}%</span>
            <span class="bl-side-metric-label">Macro F1</span>
        </div>
        <div class="bl-side-metric-row">
            <span class="bl-side-metric-value">{TEST_METRICS['Macro AUC']:.2f}%</span>
            <span class="bl-side-metric-label">Macro AUC</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bl-side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-side-heading">System</div>', unsafe_allow_html=True)
    model_status = "Model loaded" if model is not None else "Model not found"
    st.markdown(
        f"""
        <div class="bl-side-system-row"><span class="bl-side-status-dot">●</span> {model_status}</div>
        <div class="bl-side-system-row"><span class="bl-side-status-dot">●</span> {"GPU available" if gpu_available else "CPU mode"}</div>
        <div class="bl-side-system-row"><span class="bl-side-status-dot">●</span> Research mode</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bl-side-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="bl-side-footer-note">
        Research Prototype<br/>Not for Clinical Diagnosis
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="bl-hero">
        <div class="bl-eyebrow">Explainable Deep Learning for Brain MRI</div>
        <div class="bl-hero-title">🧠 BrainLens AI</div>
        <div class="bl-hero-sub">
            Analyze a single brain MRI using a fine-tuned DenseNet121 model and
            visualize the regions influencing its prediction.
        </div>
        <span class="bl-pill">Research Prototype • BRISC2025</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# WORKFLOW STRIP
# ============================================================

steps = ["Upload MRI", "Preview & Analyze", "AI Classification", "Explainable AI", "Model Information"]
if st.session_state.analysis_done:
    active_idx = 4
elif st.session_state.uploaded_file_bytes is not None:
    active_idx = 1
else:
    active_idx = 0

step_html = '<div class="bl-workflow">'
for i, step_name in enumerate(steps):
    css_class = "bl-step"
    if i < active_idx:
        css_class += " bl-step-done"
    elif i == active_idx:
        css_class += " bl-step-active"
    step_html += (
        f'<div class="{css_class}">'
        f'<div class="bl-step-num">Step {i+1:02d}</div>'
        f'<div class="bl-step-label">{step_name}</div>'
        f"</div>"
    )
    if i < len(steps) - 1:
        step_html += '<div class="bl-step-arrow">→</div>'
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Place the trained "
        f".keras file at this path relative to the app's working directory."
    )
    st.stop()

# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown('<div class="bl-section-label">Step 01</div>', unsafe_allow_html=True)
st.markdown('<div class="bl-section-title">Upload your brain MRI</div>', unsafe_allow_html=True)
st.caption(
    "Upload a JPG, JPEG or PNG brain MRI image. The image will be standardized "
    "to 224 × 224 × 3 before model inference. Allowed formats: JPG, JPEG, PNG. "
    "Maximum upload size: 10 MB."
)

uploaded_file = st.file_uploader(
    "Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is not None:
    new_bytes = uploaded_file.getvalue()
    if new_bytes != st.session_state.uploaded_file_bytes:
        st.session_state.uploaded_file_bytes = new_bytes
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.analysis_done = False
        st.session_state.predicted_class = None
        st.session_state.confidence = None
        st.session_state.probabilities = None
        st.session_state.gradcam_heatmap = None
        st.session_state.gradcam_overlay = None

# ============================================================
# EMPTY STATE / PREVIEW
# ============================================================

if st.session_state.uploaded_file_bytes is None:
    st.markdown(
        """
        <div class="bl-card">
            <div class="bl-empty">
                <div class="bl-empty-icon">🧠</div>
                <div class="bl-empty-title">Ready to analyze</div>
                <div>Upload a brain MRI scan to begin the BrainLens AI analysis workflow.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    pil_image = Image.open(io.BytesIO(st.session_state.uploaded_file_bytes))
    original_format = pil_image.format or "Unknown"
    original_dims = f"{pil_image.size[0]} × {pil_image.size[1]}"

    st.markdown('<div class="bl-section-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-section-title">Preview &amp; Analyze</div>', unsafe_allow_html=True)

    col_preview, col_meta = st.columns([1, 1.2])
    with col_preview:
        st.image(pil_image, caption="Original MRI", use_container_width=True)
    with col_meta:
        st.markdown(
            f"""
            <div class="bl-card">
                <div class="bl-side-metric-row"><span>Filename</span><span>{st.session_state.uploaded_file_name}</span></div>
                <div class="bl-side-metric-row"><span>Original dimensions</span><span>{original_dims}</span></div>
                <div class="bl-side-metric-row"><span>Image format</span><span>{original_format}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button("Analyze MRI", type="primary", use_container_width=True)

        if analyze_clicked:
            status_box = st.status("Running BrainLens AI analysis…", expanded=True)
            status_box.write("✓ Image loaded")

            padded_image, input_array = preprocess_image(pil_image, INPUT_SIZE)
            status_box.write("✓ Preprocessing complete")

            status_box.write("✓ DenseNet121 inference")
            class_idx, probs = run_inference(input_array)
            status_box.write("✓ Prediction generated")

            heatmap = generate_gradcam(input_array, class_idx)
            gradcam_overlay_img = None
            gradcam_colored_img = None
            if heatmap is not None:
                gradcam_colored_img = colorize_heatmap(heatmap, INPUT_SIZE)
                gradcam_overlay_img = overlay_heatmap(padded_image, gradcam_colored_img)
            status_box.write("✓ Explainability generated")
            status_box.update(label="Analysis complete", state="complete", expanded=False)

            st.session_state.display_image = padded_image
            st.session_state.predicted_class = CLASS_NAMES[class_idx]
            st.session_state.confidence = float(probs[class_idx])
            st.session_state.probabilities = probs
            st.session_state.gradcam_heatmap = gradcam_colored_img
            st.session_state.gradcam_overlay = gradcam_overlay_img
            st.session_state.analysis_done = True
            st.rerun()

# ============================================================
# PREDICTION RESULT
# ============================================================

if st.session_state.analysis_done and st.session_state.predicted_class is not None:
    st.markdown('<div class="bl-section-label">Step 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-section-title">AI Classification</div>', unsafe_allow_html=True)

    predicted_display = CLASS_DISPLAY[st.session_state.predicted_class]
    confidence_pct = st.session_state.confidence * 100

    col_pred, col_probs = st.columns([1, 1.3])
    with col_pred:
        st.markdown(
            f"""
            <div class="bl-card-elevated">
                <div class="bl-pred-label">AI Prediction</div>
                <div class="bl-pred-class">{predicted_display}</div>
                <div class="bl-pred-conf-label">Confidence</div>
                <div class="bl-pred-conf-value">{confidence_pct:.2f}%</div>
                <div class="bl-pred-note">
                    Highest model probability: {predicted_display}. This reflects the
                    model's output distribution only and is not a clinical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_probs:
        st.markdown('<div class="bl-card">', unsafe_allow_html=True)
        for cls, prob in zip(CLASS_NAMES, st.session_state.probabilities):
            pct = prob * 100
            active = " bl-prob-top-active" if cls == st.session_state.predicted_class else ""
            st.markdown(
                f"""
                <div class="bl-prob-row">
                    <div class="bl-prob-top{active}">
                        <span>{CLASS_DISPLAY[cls]}</span><span>{pct:.2f}%</span>
                    </div>
                    <div class="bl-prob-track">
                        <div class="bl-prob-fill" style="width:{pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------
    st.markdown('<div class="bl-section-label">Step 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="bl-section-title">Explainable AI</div>', unsafe_allow_html=True)
    st.caption("Visualizing regions contributing to the model prediction")

    if st.session_state.gradcam_heatmap is None:
        st.info(
            "Grad-CAM could not be generated for this model — the expected "
            f"backbone or target layer (`{GRAD_CAM_LAYER_NAME}`) was not found."
        )
    else:
        show_overlay = st.checkbox(
            "Show attention overlay", value=st.session_state.show_overlay
        )
        st.session_state.show_overlay = show_overlay

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(st.session_state.display_image, caption="Original MRI", use_container_width=True)
        with col_b:
            st.image(st.session_state.gradcam_heatmap, caption="Grad-CAM Attention", use_container_width=True)
        with col_c:
            if show_overlay:
                st.image(st.session_state.gradcam_overlay, caption="Attention Overlay", use_container_width=True)
            else:
                st.image(st.session_state.display_image, caption="Original MRI (overlay hidden)", use_container_width=True)

        st.markdown(
            """
            <div class="bl-card">
            Grad-CAM highlights image regions that contributed to the model's
            prediction. It is an attribution visualization and should not be
            interpreted as a tumor segmentation or anatomical boundary.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    report_text = build_report_text(
        st.session_state.uploaded_file_name,
        st.session_state.predicted_class,
        st.session_state.confidence,
        st.session_state.probabilities,
    )
    st.download_button(
        "Download Prediction Report",
        data=report_text,
        file_name=f"brainlens_report_{os.path.splitext(st.session_state.uploaded_file_name)[0]}.txt",
        mime="text/plain",
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown('<div class="bl-section-label">Step 05</div>', unsafe_allow_html=True)
st.markdown('<div class="bl-section-title">Model Information</div>', unsafe_allow_html=True)

with st.expander("Model Information", expanded=False):
    for label, value in MODEL_INFO_ROWS:
        st.markdown(
            f"""
            <div class="bl-side-metric-row">
                <span>{label}</span><span style="color:var(--bl-text);font-weight:600;">{value}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Research Performance", expanded=False):
    for label, value in TEST_METRICS.items():
        st.markdown(
            f"""
            <div class="bl-side-metric-row">
                <span>{label}</span><span style="color:var(--bl-teal);font-weight:700;">{value:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br/>", unsafe_allow_html=True)
    st.dataframe(PER_CLASS_METRICS, use_container_width=True, hide_index=True)

    st.markdown('<div class="bl-side-heading">Confusion Matrix</div>', unsafe_allow_html=True)
    st.dataframe(CONFUSION_MATRIX, use_container_width=True)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="bl-disclaimer">
        <div class="bl-disclaimer-title">Research Prototype — Not for Clinical Diagnosis</div>
        <div class="bl-disclaimer-text">
            BrainLens AI is developed for research and educational purposes.
            Model predictions and Grad-CAM visualizations should not be used
            for medical diagnosis, treatment decisions, or patient management.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="bl-footer">
        BrainLens AI · DenseNet121 · BRISC2025 · Explainable Deep Learning<br/>
        Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)
