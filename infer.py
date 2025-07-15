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
    global MODEL
    try:
        # Try to load model if it exists
        MODEL = load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        MODEL = None

@app.get("/")
async def home():
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
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/classes")
async def get_classes():
    """Get all supported disease classes"""
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
    Endpoint to predict the class of a tomato plant disease from an image.
    Returns class name, confidence score and the top predictions.
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

        # Return the prediction result
        return {
            "filename": file.filename,
            "status": "success",
            "predicted_class": result["predicted_class"],
            "confidence_percentage": result['confidence_percentage'],
            "top_predictions": result['top_predictions'],
            'recommendation': get_treatment_recommendation(result['predicted_class'])
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Prediction failed: {str(e)}",
            "message": "An error occurred while processing your image",
            "filename": file.filename
        }

def get_treatment_recommendation(disease_class: str) -> str:
    """Get treatment recommendations based on predicted disease"""
    recommendations = {
        "Healthy": "Your tomato plant appears healthy! Continue with regular care and monitoring.",
        "Early blight": "Apply fungicide containing chlorothalonil or copper. Improve air circulation and avoid overhead watering.",
        "Late Blight": "Remove affected plants immediately. Apply fungicide with copper or mancozeb. Ensure good drainage.",
        "Bacterial Spot and Speck of Tomato": "Use copper-based bactericides. Avoid overhead irrigation and improve air circulation.",
        "Grey leaf spot (fungi)": "Apply fungicide containing azoxystrobin or propiconazole. Remove affected leaves and improve air circulation."
    }
    return recommendations.get(disease_class, "Consult with a plant pathologist for proper treatment.")