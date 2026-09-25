# # Pneumonia Detection from Chest X-Ray using Deep Learning on AWS
# 
# Reconstructed from the project report (Word doc) after the original notebook/source files were lost.
# Pipeline: download dataset from S3 -> preprocess/augment -> train CNN (TensorFlow/Keras) -> evaluate -> save model back to S3 -> real-time prediction.

# ## 1. Download dataset from Amazon S3

import boto3  # AWS SDK for Python to interact with AWS services

bucket = "pneumonia-project"        # Name of your S3 bucket
key = "raw-data/chest_xray.zip"     # Path to your file inside S3

s3 = boto3.client('s3')             # Creates a connection/client to Amazon S3
s3.download_file(bucket, key, "chest_xray.zip")  # Downloads file from S3 bucket -> saves locally as chest_xray.zip

print("Downloaded!")  # Confirmation message

# ## 2. Extract the dataset

import zipfile  # Used to work with zip files

with zipfile.ZipFile("chest_xray.zip", 'r') as zip_ref:  # Opens the zip file in read mode
    zip_ref.extractall("data")  # Extracts all contents into a folder named "data"

print("Extracted!")  # Confirmation

import os  # Used to interact with file system
os.listdir("data/chest_xray")  # Lists folders inside dataset directory (should show train, test, val)

# ## 3. Import / prepare data generators

# Importing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tool for image loading and preprocessing

train_dir = "data/chest_xray/train"  # path to training data
val_dir = "data/chest_xray/val"      # path to validation data

# Create Data Generators
# Training generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,        # normalize pixel values
    rotation_range=20,     # rotate images
    zoom_range=0.2,        # zoom images
    horizontal_flip=True   # flip images
)

# Validation generator (only normalization)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load data
# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),   # resize images
    batch_size=16,           # batch size
    class_mode='binary'      # NORMAL vs PNEUMONIA
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=16,
    class_mode='binary'
)

# ## 4. Build the CNN model

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ## 5. Train the model

# train
history = model.fit(
    train_data,
    epochs=2,             # keep small (fast + safe)
    steps_per_epoch=100,  # limit steps
    verbose=2             # show only epoch logs
)

# ## 6. Evaluate on validation data

# Evaluate
loss, accuracy = model.evaluate(val_data)
print("Validation Accuracy:", accuracy)

# ## 7. Save the trained model and upload it back to S3

model.save("final_pneumonia_model.h5")

import boto3

s3 = boto3.client('s3')
s3.upload_file("final_pneumonia_model.h5", "pneumonia-project", "models/final_pneumonia_model.h5")

# ## 8. Load and evaluate on the test set

# Load test data
test_data = val_datagen.flow_from_directory(
    "data/chest_xray/test",
    target_size=(150,150),
    batch_size=16,
    class_mode='binary'
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)

# ## 9. Prediction function

import numpy as np
from tensorflow.keras.preprocessing import image

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150,150))     # load image
    img_array = image.img_to_array(img) / 255.0                # normalize
    img_array = np.expand_dims(img_array, axis=0)              # reshape

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"

result = predict_pneumonia("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
print("Prediction:", result)

# ## 10. Plot training accuracy and loss

import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ## 11. Visualize a prediction against its input image

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def show_prediction(img_path):
    img = image.load_img(img_path, target_size=(150,150))

    plt.imshow(img)
    plt.axis('off')

    result = predict_pneumonia(img_path)

    plt.title(f"Prediction: {result}")
    plt.show()

show_prediction("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
