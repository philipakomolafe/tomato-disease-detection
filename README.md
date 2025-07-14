# Tomato Disease Detection API

A FastAPI-based web service for detecting diseases in tomato plants using deep learning.

## Features

- **FastAPI Integration**: Fast and modern web API framework
- **Image Classification**: Detects tomato plant diseases from uploaded images
- **TensorFlow/Keras**: Uses custom-trained neural network model
- **Real-time Inference**: Quick image preprocessing and prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/philipakomolafe/tomato-disease-detection.git
cd tomato-disease-detection
```

2. Install dependencies:
```bash
pip install fastapi uvicorn tensorflow pillow numpy
```

3. Place your trained model file (`custom_model.keras`) in the `model/` directory

## Usage

1. Start the API server:
```bash
uvicorn infer:app --reload
```

2. Access the API at `http://localhost:8000`

3. Use the `/predict` endpoint to upload images for disease detection

## API Endpoints

- `GET /` - Welcome message and API information
- `POST /predict` - Upload an image file to get disease prediction

## Project Structure

```
├── infer.py          # Main FastAPI application
├── feature.py        # Model loading and image preprocessing
├── model/           # Directory for trained model files
├── images/          # Directory for test images
└── README.md        # Project documentation
```

## Author

Phil. A.O