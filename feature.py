import os
import sys
import glob
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

root = Path(__file__).parent.parent
sys.path.append(str(root))

# Define the model path
path = os.path.join(root, 'model', 'custom_model.keras')
image_dir = os.path.join(root, 'images')

img_ext = ['*.png', "*.jpg", "*.bmp", "*.jpeg"]

img_paths = []
for ext in img_ext:
    img_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

def load_model(model_path: str = path):
    return tf_load_model(model_path)


def preprocess_image(img_data, target_size=(224, 224)):
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
