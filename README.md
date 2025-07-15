# 🍅 Tomato Disease Detection API

A sophisticated FastAPI-based web service for detecting diseases in tomato plants using deep learning and computer vision. This API combines custom CNN models with MobileNetV2 for accurate plant validation and disease classification.

## 🌟 Features

- **🔍 Disease Detection**: Identifies 5 different tomato plant diseases with high accuracy
- **🌱 Plant Validation**: Uses MobileNetV2 to ensure uploaded images contain plant/vegetation
- **💊 Treatment Recommendations**: Provides specific treatment advice for detected diseases
- **📊 Confidence Scoring**: Returns prediction confidence levels and alternative classifications
- **🚀 Fast API**: Built with FastAPI for high performance and automatic API documentation
- **🔒 Input Validation**: Comprehensive error handling and image format validation
- **📱 Easy Integration**: RESTful API design for seamless integration with web and mobile apps

## 🏥 Supported Disease Classes

| Disease | Description | Severity |
|---------|-------------|----------|
| **Healthy** | No disease detected | ✅ None |
| **Bacterial Spot and Speck** | Bacterial infection causing dark spots | 🟡 Moderate |
| **Early Blight** | Fungal disease with concentric ring patterns | 🟠 Moderate-High |
| **Grey Leaf Spot** | Fungal infection with grey lesions | 🟠 Moderate-High |
| **Late Blight** | Severe fungal disease, can destroy crops | 🔴 Critical |

## 🛠 Technical Architecture

### Core Components

#### `feature.py` - Core Feature Module
Contains the main functionality for image processing and model operations:

- **`load_model(model_path)`**: Loads pre-trained Keras model for disease detection
- **`preprocess_image(img_data, target_size)`**: Handles image preprocessing including:
  - Format conversion (bytes/file paths to RGB arrays)
  - Resizing to model input dimensions (224x224)
  - Pixel normalization (0-1 range)
  - Batch dimension addition
- **`interpret_prediction(prediction_array, top_k)`**: Converts raw model outputs to human-readable results
- **`load_plant_validator()`**: Initializes MobileNetV2 for plant content validation
- **`is_plant_image(img_array, confidence_threshold)`**: Validates if images contain plant/vegetation using ImageNet classifications

#### `infer.py` - FastAPI Server
Main API server with comprehensive endpoints:

- **`startup_event()`**: Initializes models on server startup for optimal performance
- **`home()`**: Root endpoint with API information and supported classes
- **`health_check()`**: Health monitoring endpoint for deployment monitoring
- **`get_classes()`**: Returns all supported disease classifications
- **`predict(file)`**: Main prediction endpoint with multi-stage validation
- **`get_treatment_recommendation(disease_class)`**: Provides treatment advice based on detection results

### Data Flow Architecture

```
Image Upload → Format Validation → Plant Validation (MobileNetV2) → Disease Detection (Custom CNN) → Treatment Recommendation
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9-3.11
- 4GB+ RAM (for model loading)
- 2GB+ disk space

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/philipakomolafe/tomato-disease-detection.git
cd tomato-disease-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Place your model file**:
   - Add `custom_model.keras` to the `model/` directory
   - Or set `MODEL_URL` environment variable for remote model loading

4. **Start the development server**:
```bash
uvicorn infer:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API**:
   - API: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Production Deployment

#### Using Render (Recommended)

The project includes `render.yaml` for one-click deployment:

1. **Fork/Clone** this repository to your GitHub account
2. **Connect** your GitHub repository to Render
3. **Configure** environment variables:
   - `MODEL_URL`: URL to your trained model file
   - `TF_CPP_MIN_LOG_LEVEL`: Set to "2" to reduce logging
4. **Deploy** automatically using the included `render.yaml`

#### Manual Deployment

```bash
# Build and run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker infer:app --bind 0.0.0.0:8000
```

## 📡 API Reference

### Endpoints

#### `GET /` - API Information
Returns basic API information and supported disease classes.

**Response:**
```json
{
  "Message": "Welcome to the Tomato Inference API",
  "version": "1.0.0",
  "description": "API for tomato plant disease detection",
  "author": "Phil. A.O",
  "status": "online",
  "supported_class": ["Healthy", "Early blight", ...]
}
```

#### `GET /health` - Health Check
Monitor API and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `GET /classes` - Disease Classes
Get complete list of detectable disease classes.

**Response:**
```json
{
  "classes": [
    "Bacterial Spot and Speck of Tomato",
    "Early blight",
    "Grey leaf spot (fungi)",
    "Healthy",
    "Late Blight"
  ],
  "total_classes": 5
}
```

#### `POST /predict` - Disease Detection
Main endpoint for disease prediction from image uploads.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (PNG, JPG, JPEG)

**Success Response:**
```json
{
  "status": "success",
  "filename": "tomato_leaf.jpg",
  "predicted_class": "Early blight",
  "confidence_percentage": "87.45%",
  "top_predictions": [
    {
      "class": "Early blight",
      "confidence": 0.8745
    },
    {
      "class": "Late Blight",
      "confidence": 0.0892
    }
  ],
  "recommendation": "Apply fungicide containing chlorothalonil or copper..."
}
```

**Rejection Response (Non-plant image):**
```json
{
  "status": "rejected",
  "reason": "Image does not appear to contain plant or vegetation",
  "message": "Please upload an image of a tomato plant leaf",
  "validation_details": {
    "plant_confidence": 0.02,
    "detected_objects": [
      {"class": "sports_car", "confidence": 0.85}
    ]
  }
}
```

## 🧪 Usage Examples

### Python Client

```python
import requests

# Upload image for prediction
with open('tomato_leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    
result = response.json()
if result['status'] == 'success':
    print(f"Disease: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_percentage']}")
    print(f"Treatment: {result['recommendation']}")
```

### cURL

```bash
# Test health endpoint
curl -X GET "http://localhost:8000/health"

# Upload image for prediction
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@tomato_leaf.jpg"
```

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.status === 'success') {
        console.log('Disease:', data.predicted_class);
        console.log('Confidence:', data.confidence_percentage);
    }
});
```

## 📁 Project Structure

```
tomato-disease-detection/
├── 📄 infer.py              # FastAPI server & endpoints
├── 📄 feature.py            # Core ML functionality
├── 📄 requirements.txt      # Python dependencies
├── 📄 render.yaml          # Deployment configuration
├── 📄 README.md            # Project documentation
├── 📄 LICENSE              # MIT License
├── 📁 model/               # Model files directory
│   └── 📄 custom_model.keras
├── 📁 images/              # Sample images (optional)
└── 📁 log/                 # Application logs
    └── 📄 app.log
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_URL` | URL to trained model file | None | Optional |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow logging level | "2" | Optional |
| `PORT` | Server port number | 8000 | Optional |

### Model Requirements

- **Format**: Keras (.keras) or SavedModel format
- **Input Shape**: (224, 224, 3) RGB images
- **Output Classes**: 5 disease classes as defined in `CLASS_NAMES`
- **Framework**: TensorFlow 2.18+ recommended

## 🔍 Validation & Quality Assurance

### Multi-Stage Validation Pipeline

1. **File Format Validation**: Ensures uploaded files are valid image formats
2. **Plant Content Validation**: Uses MobileNetV2 to verify plant/vegetation presence
3. **Confidence Thresholding**: Filters low-confidence predictions
4. **Error Handling**: Comprehensive exception handling with informative error messages

### Model Performance Features

- **Preprocessing Standardization**: Consistent image normalization and resizing
- **Batch Processing**: Efficient tensor operations for scalability
- **Memory Management**: Optimized model loading and caching
- **Logging**: Comprehensive logging with log rotation for monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings for all functions
- Include unit tests for new features
- Update documentation for API changes

## 📋 Dependencies

### Core Requirements

```
fastapi                   # Web framework
uvicorn[standard]         # ASGI server with standard extras
tensorflow-cpu==2.18.0   # Machine learning framework (CPU optimized)
pillow                    # Image processing library
numpy                     # Numerical computing
python-multipart          # File upload handling for FastAPI
gunicorn                  # Production WSGI server
loguru                    # Enhanced logging with rotation
requests                  # HTTP library for model downloading
```

### Development Requirements

```
pytest                    # Testing framework
black                     # Code formatting
flake8                   # Code linting
httpx                    # Testing HTTP client for FastAPI
```

## 🚨 Error Handling

The API implements comprehensive error handling:

- **400 Bad Request**: Invalid file format or corrupted images
- **422 Unprocessable Entity**: Missing required parameters
- **500 Internal Server Error**: Model loading failures or processing errors
- **Custom Rejection**: Non-plant images with detailed feedback

## 📊 Performance Considerations

- **Model Loading**: Models are loaded once at startup for optimal performance
- **Memory Usage**: Approximately 500MB for MobileNetV2 + custom model
- **Response Time**: Typical prediction time ~500ms-2s depending on hardware
- **Concurrency**: Supports multiple concurrent requests with proper async handling

## 🔒 Security Considerations

- **File Size Limits**: Implement file size restrictions in production
- **Input Validation**: All inputs are validated before processing
- **Error Information**: Error messages don't expose sensitive system information
- **Resource Limits**: Configure appropriate memory and CPU limits for deployment

## 📈 Monitoring & Observability

- **Health Endpoint**: Monitor API availability and model status
- **Structured Logging**: JSON-formatted logs with log rotation
- **Performance Metrics**: Track prediction times and success rates
- **Error Tracking**: Comprehensive error logging and classification

## 📞 Support & Contact

- **Author**: Phil. A.O
- **Repository**: [tomato-disease-detection](https://github.com/philipakomolafe/tomato-disease-detection)
- **Issues**: Submit bug reports and feature requests via GitHub Issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the robust ML framework
- FastAPI developers for the excellent web framework
- MobileNetV2 authors for the efficient CNN architecture
- Plant pathology research community for disease classification insights

---

**⚠️ Disclaimer**: This tool provides automated disease detection for informational purposes. For critical agricultural decisions or severe plant diseases, always consult with professional plant pathologists or agricultural extension services.