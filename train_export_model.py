import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuration
CLASSES = ["Healthy", "Powdery Mildew", "Leaf Rust", "Blight", "Aphids"]
IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASSES)
EPOCHS = 2
BATCH_SIZE = 32

def create_synthetic_data():
    print("Generating synthetic data for training...")
    # 100 images per class
    x_train = np.random.randint(0, 256, (NUM_CLASSES * 100, 224, 224, 3), dtype=np.uint8)
    y_train = np.repeat(np.arange(NUM_CLASSES), 100)

    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    return x_train, y_train

def train_and_export():
    x_train, y_train = create_synthetic_data()

    # Lightweight MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None, pooling='avg'
    )

    model = models.Sequential([
        base_model,
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    modelL_optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=modelL_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training lightweight model...")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Export to TFLite with quantization
    print("Exporting to quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("crop_model.tflite", "wb") as f:
        f.write(tflite_model)

    with open("labels.json", "w") as f:
        jsonL_labels = {str(i): label for i, label in enumerate(CLASSES)}
        json.dump(jsonL_labels, f)

    print("Model and labels exported successfully.")

if __name__ == "__main__":
    train_and_export()
