import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
import time

# Import your model class
from model import HybridTumorDetector  # Assuming model.py contains your HybridTumorDetector class

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and return detector instance"""
    try:
        detector = HybridTumorDetector()
        
        # Load your pre-trained models
        detector.load_models('models/')  # Load ResNet classifier
        detector.load_gan_models(
            'models/generator.keras',
            'models/discriminator.keras'
        )
        
        # Load threshold
        threshold_path = Path('models/threshold.npy')
        if threshold_path.exists():
            detector.threshold = float(np.load(threshold_path))
        
        return detector
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image"""
    try:
        # Read image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        
        # Convert to numpy array and resize
        image_np = np.array(image)
        image_np = cv2.resize(image_np, (256, 256))
        
        # Normalize
        image_np = image_np.astype(np.float32) / 255.0
        
        return image_np
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def main():
    # Header
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("""
    This application uses a hybrid deep learning model combining GAN-based anomaly detection
    and ResNet classification to analyze brain MRI images for tumor detection and classification.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        detector = load_model()
        if detector is None:
            st.error("Failed to load model. Please try again later.")
            return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Show uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process image button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                processed_image = preprocess_image(uploaded_file)
                if processed_image is None:
                    return
                
                # Make prediction
                prediction, confidence_scores, anomaly_score = detector.predict_hybrid(processed_image)
                
                with col2:
                    st.markdown("### Analysis Results")
                    
                    # Display results in a formatted box
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    # Show primary prediction with confidence
                    st.markdown(f"**Detected Condition:** {prediction.title()}")
                    st.markdown(f"**Confidence:** {confidence_scores[np.argmax(confidence_scores)]:.2%}")
                    
                    # Show anomaly score if available
                    if anomaly_score is not None:
                        st.markdown(f"**Anomaly Score:** {anomaly_score:.4f}")
                        if detector.threshold is not None:
                            st.markdown(f"**Threshold:** {detector.threshold:.4f}")
                            if anomaly_score > detector.threshold:
                                st.warning("⚠️ Anomaly detected: Image shows unusual patterns")
                            else:
                                st.success("✅ No unusual patterns detected")
                    
                    # Show confidence scores for all classes
                    st.markdown("### Confidence Scores by Class")
                    classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
                    for cls, conf in zip(classes, confidence_scores):
                        st.progress(float(conf))
                        st.markdown(f"{cls}: {conf:.2%}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add information about the model
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        This system uses a hybrid approach combining:
        1. **GAN-based Anomaly Detection:** Identifies unusual patterns in brain MRI images
        2. **ResNet Classifier:** Categorizes tumors into specific types
        3. **Combined Analysis:** Provides comprehensive results using both approaches
        
        The model can detect and classify:
        - No Tumor (Normal brain MRI)
        - Glioma
        - Meningioma
        - Pituitary Tumor
        """)

if __name__ == "__main__":
    main()