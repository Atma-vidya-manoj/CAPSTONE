import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Set the directories for images and labels
IMAGE_DIR = 'C:\\Users\\ATMA\\Downloads\\1\\Dataset\\images'
LABEL_DIR = 'C:\\Users\\ATMA\\Downloads\\1\\Dataset\\labels'
IMG_SIZE = 224  # Resize to 224x224 for YOLO input

# Load and preprocess images for YOLO
def load_images_yolo(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
    images = []
    labels = []

    for image_file in image_files:
        # Load image
        img = cv2.imread(os.path.join(image_dir, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to the input size
        images.append(img)
        
        # Load corresponding label (bounding boxes for YOLO)
        label_file = image_file.replace('.jpeg', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = f.read().strip()
            labels.append(label)

    return np.array(images), np.array(labels)

# Define YOLO model (simplified version for demonstration)
def create_yolo_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    model.add(layers.Dense(num_classes, activation='softmax'))  # For classification

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load data
X, y = load_images_yolo(IMAGE_DIR, LABEL_DIR)

# Encode labels for classification
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and compile YOLO model
yolo_model = create_yolo_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(np.unique(y_encoded)))

# Use a learning rate scheduler to adjust the learning rate during training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train the YOLO model
history = yolo_model.fit(X_train, y_train, 
                         epochs=10, 
                         validation_data=(X_val, y_val),
                         callbacks=[lr_scheduler])

# Save the YOLO model
yolo_model.save('yolo_model.h5')

# Plot training history for YOLO
def plot_history_yolo(history):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('YOLO Model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('YOLO Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot YOLO model results
plot_history_yolo(history)
