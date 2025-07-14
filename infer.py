import os
from fastapi import FastAPI, UploadFile, File
from feature import preprocess_image, load_model




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
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of a tomato plant disease from an image.
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

        # Return the prediction result
        return {
            "prediction": prediction.tolist(),
            "filename": file.filename,
            "status": "success"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "status": "failed"}

