from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app = FastAPI()

def create_preprocessing_model():
    """Create a model that matches the original preprocessing layers"""
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(256, 256),
        tf.keras.layers.Rescaling(1./255)
    ])

def create_core_model():
    """Create the core model architecture without preprocessing layers"""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

# Create preprocessing model and core model
preprocessing = create_preprocessing_model()
core_model = create_core_model()

# Load weights from saved model
try:
    core_model.load_weights("../saved_models/model_v1.keras")
except Exception as e:
    print(f"Error loading weights: {e}")
    # Try alternative path
    core_model.load_weights("saved_models/model_v1.keras")

# Combine preprocessing and core model
def create_inference_model():
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = preprocessing(inputs)
    outputs = core_model(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

MODEL = create_inference_model()
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize to expected dimensions
        image = image.resize((256, 256))
        # Convert to numpy array
        return np.array(image)
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image = read_file_as_image(await file.read())
        
        # Add batch dimension
        img_batch = np.expand_dims(image, 0)
        
        # Make prediction
        predictions = MODEL.predict(img_batch)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)