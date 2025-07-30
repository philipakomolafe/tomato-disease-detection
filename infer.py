"""
Tomato Disease Detection API Server

FastAPI-based web service for detecting diseases in tomato plants using deep learning.
Provides REST endpoints for image upload, plant validation, disease classification,
and treatment recommendations.

Features:
- Plant/vegetation validation using MobileNetV2
- Disease classification using custom CNN model
- Treatment recommendations for detected diseases
- Comprehensive error handling and validation

Author: Phil. A.O
Version: 1.0.0
"""

import os
from fastapi import FastAPI, UploadFile, File
from feature import preprocess_image, load_model, interpret_prediction, is_plant_image



# Init app instance
app = FastAPI(
    title="Tomato Disease Detection API",
    description="API for detecting diseases in tomato plants using deep learning",
    version="1.0.0"
)

# Load model once at startup
MODEL = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup.
    
    Loads the tomato disease detection model into memory for faster
    prediction responses. Handles loading errors gracefully by setting
    MODEL to None, which triggers lazy loading on first prediction.
    
    Global Variables:
        MODEL: Stores the loaded TensorFlow/Keras model instance
        
    Raises:
        Exception: Catches and logs any model loading errors
    """
    global MODEL
    try:
        # Try to load model if it exists
        MODEL = load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        MODEL = None

@app.get("/")
async def home():
    """
    Root endpoint providing API information and status.
    
    Returns basic information about the API including supported disease classes,
    version information, and current operational status.
    
    Returns:
        dict: JSON response containing API metadata and supported classes
    """
    return {
        "Message": "Welcome to the Tomato Inference API",
        "version": "1.0.0",
        "description": "This API provides endpoints for tomato plant disease detection and classification using custom-trained model",
        "author": "Phil. A.O",
        "status": "online",
        "supported_class": [
            "Bcterial Spot and Speck of Tomato",
            "Early blight",
            "Grey leaf spot (fungi)",
            "Healthy",
            "Late Blight"
        ]
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring API status.
    
    Returns the operational status of the API and whether the
    ML model is successfully loaded and ready for predictions.
    
    Returns:
        dict: Health status and model availability information
    """
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/classes")
async def get_classes():
    """
    Get all supported tomato disease classes.
    
    Returns a complete list of disease classifications that the model
    can detect, along with the total count of supported classes.
    
    Returns:
        dict: Dictionary containing list of disease classes and total count
    """
    return {
        "classes": [
            "Bacterial Spot and Speck of Tomato",
            "Early blight", 
            "Grey leaf spot (fungi)",
            "Healthy",
            "Late Blight"
        ],
        "total_classes": 5
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint for tomato disease detection.
    
    Accepts image uploads, validates that they contain plant/vegetation content,
    and returns disease classification with confidence scores and treatment
    recommendations. Implements multi-stage validation to ensure accuracy.
    
    Args:
        file (UploadFile): Image file containing tomato plant leaf
        
    Returns:
        dict: Prediction results containing:
            - status: "success", "rejected", or "error"
            - predicted_class: Disease classification name
            - confidence_percentage: Prediction confidence as percentage
            - top_predictions: Ranked list of alternative predictions
            - recommendation: Treatment advice for detected condition
            - validation_details: Plant validation information (if rejected)
            
    Raises:
        HTTPException: If file upload or processing fails
        ValueError: If image format is unsupported
    """
    global MODEL
    
    # Load the model if not already loaded
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            return {"error": f"Model could not be loaded: {str(e)}"}

    try:
        # Read the image file
        image = await file.read()
        
        # Preprocess the image
        img = preprocess_image(image)

        # Validate image type..
        plant_validation = is_plant_image(img, confidence_threshold=0.05)

        if not plant_validation['is_plant']:
            return {
                "status": "rejected",
                "reason": "Image does not appear to contain plant or vegetation",
                "message": "Please upload an image of a tomato plant leaf",
                "validation_details": {
                    "plant_confidence": plant_validation.get("plant_confidence", 0.0),
                    "detected_objects": plant_validation.get("top_predictions", []),
                    "validation_reason": plant_validation.get("reason", "Unknown")
                },
                "filename": file.filename
            }

        # Make prediction
        prediction = MODEL.predict(img, verbose=0)

        # Interpret the prediction result..
        result = interpret_prediction(prediction)
        recommendation, description = get_treatment_recommendation(result['predicted_class'])
        # Return the prediction result
        return {
            "filename": file.filename,
            "status": "success",
            "predicted_class": result["predicted_class"],
            "confidence_percentage": result['confidence_percentage'],
            "top_predictions": result['top_predictions'],
            'recommendation': recommendation,
            "description": description,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Prediction failed: {str(e)}",
            "message": "An error occurred while processing your image",
            "filename": file.filename
        }

def get_treatment_recommendation(disease_class: str) -> str:
    """
    Get treatment recommendations based on predicted disease class.
    
    Provides specific, actionable treatment advice for each type of
    tomato plant disease that can be detected by the model.
    
    Args:
        disease_class (str): Name of the detected disease class
        
    Returns:
        str: Detailed description and treatment recommendation text
        
    Note:
        Recommendations are for informational purposes and should not
        replace professional agricultural consultation for severe cases.
    """
    recommendations = {
        "Healthy": "Your tomato plant appears healthy! Continue with regular care and monitoring.",
        "Early blight": "Apply fungicide containing chlorothalonil or copper. Improve air circulation and avoid overhead watering.",
        "Late Blight": "Remove affected plants immediately. Apply fungicide with copper or mancozeb. Ensure good drainage.",
        "Bacterial Spot and Speck of Tomato": "Use copper-based bactericides. Avoid overhead irrigation and improve air circulation.",
        "Grey leaf spot (fungi)": "Apply fungicide containing azoxystrobin or propiconazole. Remove affected leaves and improve air circulation."
    }

    descriptions = {
        "Healthy": "Healthy leaf. Color is right, no spots or holes. Your crop is doing just fine.",
        "Early blight": "Dark brown to black spots with concentric rings on lower leaves, often with yellowing. Caused by Alternaria solani, it thrives in warm, wet conditions.",
        "Late Blight": "Large, irregular gray-green to dark brown spots on leaves, often with white mold in humid conditions. Caused by Phytophthora infestans, spreads rapidly in cool, wet weather.",
        "Bacterial Spot and Speck of Tomato": " Small, water-soaked spots on leaves turn dark brown to black with yellow halos; spots may merge, causing leaf drop. Caused by Xanthomonas bacteria, spread by water splashes in warm, wet conditions.",
        "Grey leaf spot (fungi)": "Yellow spots on upper leaf surfaces with grayish-white to olive-green mold on undersides. Caused by Fulvia fulva, favored by high humidity.",
    }
    
    return recommendations.get(disease_class, "Consult with a plant pathologist for proper treatment."), descriptions.get(disease_class, "Consult with a plant pathologist for proper evaluation."), 
