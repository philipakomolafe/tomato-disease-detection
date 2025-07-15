"""
Tomato Disease Detection Feature Module

This module provides core functionality for tomato plant disease detection using deep learning.
It includes image preprocessing, model loading, prediction interpretation, and plant validation
using MobileNetV2 for ensuring uploaded images contain plant/vegetation content.

Author: Phil. A.O
Version: 1.0.0
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
from loguru import logger as log 
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import io

root = Path(__file__).parent
sys.path.append(str(root))
os.makedirs('log', exist_ok=True)

# Configure Logger.
log.add(os.path.join(root, 'log', 'app.log'), rotation="1 MB", retention="7 days")


# Define the model path
path = os.path.join(root, 'model', 'custom_model.keras')
image_dir = os.path.join(root, 'images')

# Class mapping from your training data
CLASS_NAMES = {
    0: 'Bacterial Spot and Speck of Tomato',
    1: 'Early blight',
    2: 'Grey leaf spot (fungi)',
    3: 'Healthy',
    4: 'Late Blight'
}

PLANT_CLASSES = [
    'strawberry', 'orange', 'lemon', 'banana', 'apple', 'broccoli', 'carrot', 'hot_pepper',
    'bell_pepper', 'cauliflower', 'mushroom', 'artichoke', 'corn', 'cucumber', 'zucchini',
    'acorn_squash', 'butternut_squash', 'cucumber', 'head_cabbage', 'broccoli', 'cauliflower',
    'ear', 'rapeseed', 'daisy', 'yellow_lady_slipper', 'cliff', 'valley', 'alp',
    'tree', 'plant', 'leaf', 'flower', 'bud', 'bloom', 'botanical', 'foliage', 'vegetation'
]

# Load MobileNetV2 for plant validation 
PLANT_VALIDATOR = None


img_ext = ['*.png', "*.jpg", "*.bmp", "*.jpeg"]

img_paths = []
for ext in img_ext:
    img_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

def load_model(model_path: str = path):
    """
    Load a pre-trained Keras model for tomato disease detection.
    
    Args:
        model_path (str): Path to the saved Keras model file (.keras format)
        
    Returns:
        tf.keras.Model: Loaded TensorFlow/Keras model ready for prediction
        
    Raises:
        FileNotFoundError: If model file doesn't exist at specified path
        Exception: If model loading fails due to compatibility or corruption issues
    """
    return tf_load_model(model_path)


def preprocess_image(img_data, target_size=(224, 224)):
    """
    Preprocess image data for model prediction.
    
    Handles both file paths and byte data, converts to RGB format,
    resizes to target dimensions, normalizes pixel values to [0,1],
    and adds batch dimension for model compatibility.
    
    Args:
        img_data (bytes or str): Image data as bytes or file path
        target_size (tuple): Target image dimensions (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, height, width, 3)
        
    Raises:
        PIL.UnidentifiedImageError: If image data is corrupted or invalid format
        ValueError: If image cannot be processed or resized
    """
    # Handle both file path and byte data
    if isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
    else:
        img = image.load_img(img_data, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1]
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def interpret_prediction(prediction_array, top_k=3):
    """
    Convert model prediction probabilities to human-readable format.
    
    Takes raw model output and converts it to class names with confidence scores,
    providing both the top prediction and a ranked list of alternatives.
    
    Args:
        prediction_array (numpy.ndarray): Raw prediction probabilities from model
        top_k (int): Number of top predictions to include in results
        
    Returns:
        dict: Dictionary containing:
            - predicted_class (str): Name of the most likely disease class
            - confidence_percentage (str): Confidence as formatted percentage
            - top_predictions (list): List of top predictions with classes and confidences
            
    Raises:
        IndexError: If prediction array doesn't match expected class count
        ValueError: If prediction array contains invalid values
    """
    try:
        prediction = np.squeeze(prediction_array)
        top_indices = np.argsort(prediction)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "class": CLASS_NAMES[idx],
                "confidence": float(prediction[idx])
            })
        
        return {
            "predicted_class": CLASS_NAMES[np.argmax(prediction)],
            "confidence_percentage": f'{float(np.max(prediction)) * 100:.2f}%',
            "top_predictions": results
               }

    except Exception as e:
        log.error(f'Error in prediction interpreting: {e}')
        return {
            "predicted_class": "Unknown",
            "confidence_percentage": '0.00%',
            "top_predictions": [],
            "error": str(e),
        }

def load_plant_validator():
    """
    Load and initialize MobileNetV2 model for plant/vegetation validation.
    
    Uses ImageNet pre-trained weights to classify images and determine if they
    contain plant or vegetation content. Model is loaded once and cached globally
    to avoid repeated loading overhead.
    
    Returns:
        tf.keras.Model or None: Loaded MobileNetV2 model, or None if loading fails
        
    Raises:
        ImportError: If TensorFlow or model dependencies are missing
        MemoryError: If insufficient memory to load the model
    """
    global PLANT_VALIDATOR 
    
    if PLANT_VALIDATOR is None:
        try:
            log.info("Loading MobileNetV2 for plant validation...")
            # mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
            PLANT_VALIDATOR = MobileNetV2(weights='imagenet', include_top=True)
            log.info("MobileNetV2 loaded successfully...")          

        except Exception as e:
            log.error(f"Error loading MobileNetV2...")
            PLANT_VALIDATOR = None
        
    return PLANT_VALIDATOR


def is_plant_image(img_array, confidence_threshold=0.1):
    """
    Validate if an image contains plant or vegetation content using MobileNetV2.
    
    Uses pre-trained MobileNetV2 to classify the image and checks if any of the
    top predictions correspond to plant-related ImageNet classes. This helps
    filter out non-plant images before disease detection.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array with shape (1, 224, 224, 3)
        confidence_threshold (float): Minimum confidence score (0.0-1.0) required 
                                    for positive plant detection
        
    Returns:
        dict: Dictionary containing:
            - is_plant (bool): True if plant/vegetation detected above threshold
            - plant_confidence (float): Highest confidence score for plant classes
            - top_predictions (list): Top 5 ImageNet class predictions
            - reason (str): Human-readable explanation of the result
            
    Raises:
        ValueError: If img_array has incorrect shape or invalid confidence_threshold
        RuntimeError: If MobileNetV2 prediction fails
    """

    try:
        validator = load_plant_validator()
        if validator is None:
            return {"is_plant": True, "confidence": 0.5, "reason": "Validator not available"}

        # Preprocess for MobileNetV2 
        # MobileNetV2 accepts image resolutions between [0.0, 255.0]
        img_mobilenet = preprocess_input(img_array * 255.0)

        # MobileNetV2 prediction.
        predictions = validator.predict(img_mobilenet, verbose=0)
        decoded = decode_predictions(predictions, top=10)[0] 


        # Check if top predictions are plant related..
        plant_confidence = 0.0
        detected_classes = []

        for class_id, class_name, confidence in decoded:
            detected_classes.append({'class': class_name, 'confidence': float(confidence)})

            # checking if class name contains plant related keywords.
            class_lower = class_name.lower()
            for plant_keyword in PLANT_CLASSES:
                if plant_keyword in class_lower or any(keyword in class_lower for keyword in ["leaf", 'plant', 'flower', 'tree', 'vegetable', 'fruit', 'herb', 'crop']):
                    plant_confidence = max(plant_confidence, confidence)
                    break

        is_plant = plant_confidence >= confidence_threshold

        return {
                'is_plant': is_plant,
                'plant_confidence': float(plant_confidence),
                "top_predictions": detected_classes[:5],
                'reason': "Plant detected" if is_plant else "No plant/vegetation detected"
            }

    except Exception as e:
        log.error(f"Error in plant validation: {e}")
        return {"is_plant": True, "confidence": 0.5, "reason": f"Validation error: {str(e)}"}
