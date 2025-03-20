import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
import os

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess the uploaded image
    """
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if grayscale and convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to target size
    img_array = cv2.resize(img_array, target_size)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_quantized_model(image, model_path):
    """
    Predict the class of a mammography image using a quantized TFLite model.
    
    Args:
        image (PIL.Image): The input image.
        model_path (str): Path to the TFLite model file.
        
    Returns:
        tuple: (str, float) - Prediction label ("BENIGN" or "MALIGNANT") and confidence score
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        return None, None
        
    print(f"Using TFLite model from {model_path}")
    
    # Preprocess the image: convert to RGB and resize to (224, 224)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    
    # Convert image to numpy array
    input_data = np.array(image, dtype=np.float32)
    
    # Normalize pixel values to [0, 1]
    input_data = input_data / 255.0
    
    # Load and run the quantized model with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details from the interpreter
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Model input shape: {input_details['shape']}, output shape: {output_details['shape']}")
    
    # If the model is quantized, adjust the input data.
    input_scale, input_zero_point = input_details["quantization"]
    if input_scale > 0:
        # Quantize the float32 image to integer representation
        input_data = input_data / input_scale + input_zero_point
    
    # Expand dimensions to match expected shape: (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)
    # Convert to the expected data type (e.g., uint8 or int8)
    input_data = input_data.astype(input_details["dtype"])
    
    # Set the tensor to the interpreter
    interpreter.set_tensor(input_details["index"], input_data)
    # Run inference
    interpreter.invoke()
    
    # Get output predictions
    output_data = interpreter.get_tensor(output_details["index"])
    
    # If the output is quantized, dequantize it.
    output_scale, output_zero_point = output_details["quantization"]
    if output_scale > 0:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"TFLite PREDICTION: {output_data}")
    
    # Get the predicted class (assuming a softmax output over 2 classes)
    if output_data.shape[1] == 1:
        # Single output neuron (sigmoid)
        result = "MALIGNANT" if output_data[0][0] > 0.5 else "BENIGN"
        confidence = float(output_data[0][0]) if output_data[0][0] > 0.5 else float(1 - output_data[0][0])
    else:
        # Two output neurons (softmax)
        predicted_class = np.argmax(output_data[0])
        result = "MALIGNANT" if predicted_class == 1 else "BENIGN"
        confidence = float(output_data[0][predicted_class])
    
    return result, confidence 