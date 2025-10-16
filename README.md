# DLSC_Fuzzy-Post-Processing-for-Image-Classification
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
