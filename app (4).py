import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from skfuzzy import control as ctrl
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="🧠 Fuzzy Post-Processing for Image Classification", layout="wide")
st.title("🧠 Fuzzy Post-Processing for Image Classification")
st.write("Upload an image and see how CNN + Fuzzy Logic interpret its certainty.")

# ---------------------------
# Helper: Image Quality Check
# ---------------------------
def calculate_sharpness(image_file):
    """Calculate image sharpness using Laplacian variance."""
    img = Image.open(image_file).convert('L')
    img_cv = np.array(img)
    lap_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
    return lap_var

def calculate_brightness(image_file):
    """Calculate image brightness (0–255)."""
    img = Image.open(image_file).convert('L')
    img_cv = np.array(img)
    return np.mean(img_cv)

# ---------------------------
# Helper: Fuzzy Logic Setup
# ---------------------------
confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
gap = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'gap')
entropy = ctrl.Antecedent(np.arange(0, 3.01, 0.01), 'entropy')
certainty = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'certainty')

# Membership functions
confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.4])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.2, 0.5, 0.8])
confidence['high'] = fuzz.trimf(confidence.universe, [0.6, 1, 1])

gap['small'] = fuzz.trimf(gap.universe, [0, 0, 0.2])
gap['medium'] = fuzz.trimf(gap.universe, [0.1, 0.3, 0.5])
gap['large'] = fuzz.trimf(gap.universe, [0.4, 1, 1])

entropy['low'] = fuzz.trimf(entropy.universe, [0, 0, 1])
entropy['medium'] = fuzz.trimf(entropy.universe, [0.5, 1.5, 2.2])
entropy['high'] = fuzz.trimf(entropy.universe, [1.5, 3, 3])

certainty['low'] = fuzz.trimf(certainty.universe, [0, 0, 0.4])
certainty['medium'] = fuzz.trimf(certainty.universe, [0.3, 0.5, 0.7])
certainty['high'] = fuzz.trimf(certainty.universe, [0.6, 1, 1])

rule1 = ctrl.Rule(confidence['high'] & gap['large'] & entropy['low'], certainty['high'])
rule2 = ctrl.Rule(confidence['medium'] & gap['medium'] & entropy['medium'], certainty['medium'])
rule3 = ctrl.Rule(confidence['low'] | entropy['high'], certainty['low'])
rule4 = ctrl.Rule(gap['small'] & entropy['high'], certainty['low'])

certainty_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
certainty_sim = ctrl.ControlSystemSimulation(certainty_ctrl)

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="🖼 Uploaded Image", use_column_width=True)
    st.markdown("---")

    # ---------------------------
    # Image Quality Analysis
    # ---------------------------
    sharpness_value = calculate_sharpness(uploaded_file)
    brightness_value = calculate_brightness(uploaded_file)

    sharpness_score = np.clip((sharpness_value / 500.0) * 100, 0, 100)
    brightness_score = np.clip((brightness_value / 255.0) * 100, 0, 100)

    if sharpness_score > 60:
        sharpness_msg = "🟢 Image is clear — good for classification."
        sharp_color = "green"
    elif sharpness_score > 25:
        sharpness_msg = "🟠 Slightly blurry — results may be ambiguous."
        sharp_color = "orange"
    else:
        sharpness_msg = "🔴 Image is too blurry — model may be uncertain."
        sharp_color = "red"

    if brightness_score < 25:
        bright_msg = "🔴 Image is too dark — may affect prediction."
        bright_color = "red"
    elif brightness_score > 85:
        bright_msg = "🔴 Image is too bright — may cause loss of detail."
        bright_color = "red"
    else:
        bright_msg = "🟢 Brightness is good."
        bright_color = "green"

    st.subheader("📷 Image Quality Check")
    st.markdown(f"**Sharpness Score:** `{sharpness_score:.2f}` / 100")
    st.markdown(f"<h4 style='color:{sharp_color};'>{sharpness_msg}</h4>", unsafe_allow_html=True)
    st.markdown(f"**Brightness Score:** `{brightness_score:.2f}` / 100")
    st.markdown(f"<h4 style='color:{bright_color};'>{bright_msg}</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # ---------------------------
    # CNN Prediction
    # ---------------------------
    img = Image.open(uploaded_file).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = mobilenet_v2.preprocess_input(x)

    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    st.subheader("📈 CNN Predictions (Top 3)")
    for label, name, prob in decoded:
        st.write(f"**{name}** — {prob*100:.2f}%")

    # ---------------------------
    # Fuzzy Logic Post-Processing
    # ---------------------------
    probs = [p[2] for p in decoded]
    top_conf = probs[0]
    gap_val = probs[0] - probs[1]
    entropy_val = -np.sum(np.array(probs) * np.log(np.array(probs) + 1e-9))

    certainty_sim.input['confidence'] = top_conf
    certainty_sim.input['gap'] = gap_val
    certainty_sim.input['entropy'] = entropy_val
    certainty_sim.compute()
    fuzzy_score = certainty_sim.output['certainty']

    if fuzzy_score > 0.7:
        interp = "Certain ✅"
    elif fuzzy_score > 0.4:
        interp = "Ambiguous 😐"
    else:
        interp = "Uncertain 🤔"

    st.subheader("🌀 Fuzzy Logic Interpretation")
    st.write(f"**Top Confidence:** {top_conf:.2f}")
    st.write(f"**Confidence Gap:** {gap_val:.2f}")
    st.write(f"**Entropy:** {entropy_val:.2f}")
    st.write(f"**Fuzzy Certainty Score:** {fuzzy_score:.2f}")
    st.write(f"**Interpretation:** {interp}")

    # ---------------------------
    # Explanation
    # ---------------------------
    st.markdown("---")
    st.subheader("💬 Explanation")
    st.markdown("""
    This enhanced system combines **CNN (MobileNetV2)** + **Fuzzy Logic** to evaluate how confident
    the model really is in its prediction.

    **How it works:**
    - **Top-1 Confidence** → Strength of main prediction  
    - **Confidence Gap** → How far the top prediction is from next one  
    - **Entropy** → How spread/confused all probabilities are  

    **Interpretation Rules:**
    - High confidence + large gap + low entropy → ✅ *Certain*
    - Medium values → 😐 *Ambiguous*
    - Low confidence or high entropy → 🤔 *Uncertain*

    Also, the app now checks:
    - 🔍 **Sharpness (blur detection)**  
    - 💡 **Brightness (lighting quality)**  
    Before classifying, so you know if the image is suitable.
    """)

