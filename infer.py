import os
from fastapi import FastAPI, UploadFile, File
from feature import preprocess_image, load_model, interpret_prediction



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

        # Make prediction
        prediction = MODEL.predict(img)

        # Interpret the prediction result..
        result = interpret_prediction(prediction)

        # Return the prediction result
        return {
            "filename": file.filename,
            "status": "success",
            "predicted_class": result["predicted_class"],
            "confidence_percentage": result['confidence_percentage'],
            "top_predictions": result['top_predictions'],
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "status": "failed"}

