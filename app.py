import streamlit as st
import os
from model_utils import preprocess_image, predict_with_quantized_model
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# Define CSS styles
st.markdown("""
<style>
    .result-benign {
        color: #0c8a26;
        font-size: 32px;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        background-color: #e6f7e8;
        text-align: center;
        margin: 20px 0;
    }
    .result-malignant {
        color: #d62728;
        font-size: 32px;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        background-color: #fce8e8;
        text-align: center;
        margin: 20px 0;
    }
    .confidence {
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # App header
    st.markdown("<h1 class='header'>Breast Cancer Detection</h1>", unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.markdown("This app uses a Gabor-enhanced ConvNeXtBase CNN model to detect breast cancer from mammogram images.")
        st.markdown("The model was trained on the CBIS-DDSM dataset and classifies images as BENIGN or MALIGNANT.")
        st.markdown("Upload an image to get the prediction.")
    
    # Use absolute path for model
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Define model path - specifically for TFLite model
    model_path = os.path.join(base_dir, "model", "ConvNeXt_quant_model.keras")
    
    # Check if model file exists
    if os.path.exists(model_path):
        st.success(f"Model loaded successfully")
    else:
        st.warning(f"Model not found at: {model_path}")
        st.info("Please ensure your model file is placed in the 'model' directory with the correct name.")
    
    # Load model
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            # Use predict_with_quantized_model directly for TFLite inference
            if os.path.exists(model_path):
                st.session_state['model_path'] = model_path
                st.session_state['model_type'] = "TFLite"
                model_type = "TFLite"
                model = None
            else:
                # If model not found, don't use any fallback
                st.error(f"Model not found at {model_path}. Please add the model file and restart the application.")
                st.stop()
            
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
    else:
        model = st.session_state.get('model')
        model_type = "TFLite"  # Always use TFLite
    
    # Main content area
    st.markdown("### Upload a Mammogram Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image and make prediction
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image_bytes)
            
            # Show processed image
            with col2:
                # Convert from (1, H, W, 3) to (H, W, 3) for display
                display_img = np.squeeze(processed_image, axis=0)
                st.image(display_img, caption="Processed Image", use_container_width=True)
            
            # Make prediction
            if 'model_path' in st.session_state:
                # Use direct TFLite prediction with the original image
                prediction, confidence = predict_with_quantized_model(image, st.session_state['model_path'])
            else:
                st.error("Model path not found in session state. Please restart the application.")
                st.stop()
            
            # Display result
            if prediction:
                st.markdown("## Prediction Result:")
                if prediction == "BENIGN":
                    st.markdown(f"<div class='result-benign'>{prediction}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-malignant'>{prediction}</div>", unsafe_allow_html=True)
                
                # Display confidence
                confidence_percentage = round(confidence * 100, 2)
                st.markdown(f"<div class='confidence'>Confidence: {confidence_percentage}%</div>", 
                            unsafe_allow_html=True)
                
                # Display progress bar for confidence
                st.progress(confidence)
            else:
                st.error("Error occurred during prediction. Please try again.")
    else:
        # Show information when no image is uploaded
        st.info("Please upload a mammogram image to get started.")
        
        # Show example images if available
        if os.path.exists("examples"):
            st.markdown("### Example Images")
            example_files = [f for f in os.listdir("examples") if f.endswith(('.png', '.jpg', '.jpeg'))]
            if example_files:
                st.write("Click on any example to use it:")
                cols = st.columns(min(3, len(example_files)))
                for i, file in enumerate(example_files[:3]):
                    with cols[i % 3]:
                        st.image(f"examples/{file}", caption=file, use_container_width=True)

if __name__ == "__main__":
    main() 