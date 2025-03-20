# Breast Cancer Detection App

A Streamlit application for breast cancer detection using a Gabor-enhanced ConvNeXtBase CNN model to classify mammography images as BENIGN or MALIGNANT.

## Model Information

The app uses a Gabor-enhanced ConvNeXtBase model that:
- Incorporates Gabor filters to enhance feature extraction from mammography images
- Was trained on the CBIS-DDSM breast cancer dataset
- Achieves improved detection of subtle tissue abnormalities
- Expects input images of size 224x224 pixels
- Outputs a binary classification (BENIGN vs MALIGNANT)

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your trained model in the `model` directory:
   - Default path: `model/breast_cancer_model.keras`
   - The newer .keras format is recommended over the legacy .h5 format
   - The model should be the Gabor-enhanced ConvNeXtBase model from your training
   - You can specify a different path in the app's sidebar

3. (Optional) Add example mammography images to the `examples` directory for demonstration purposes.

## Running the App

To start the Streamlit app, run:
```
streamlit run app.py
```

The app will launch in your default web browser.

## Using the App

1. Upload a mammography image using the file uploader
2. The app will display both the original and preprocessed images
3. The model will analyze the image and display the result as BENIGN or MALIGNANT with a confidence score

## Customization

You can customize the app by:
- Modifying the preprocessing steps in `model_utils.py` to match your model's requirements
- Adjusting the UI styling in `app.py`
- Adding additional features like batch processing or result history 