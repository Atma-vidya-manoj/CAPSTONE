import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load and Preprocess the Image Dataset (e.g., Smart City Waste Dataset)
# Assuming you have images organized into directories by class for training

# Set up an image data generator for the dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Assuming the dataset is organized as a directory with subdirectories for each class (e.g., 'plastic_bottle', 'cardboard_box', etc.)
train_generator = datagen.flow_from_directory(
    'path_to_images/',  # Path to the folder containing class folders
    target_size=(224, 224),  # Resize images to 224x224 for ResNet50
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'path_to_images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Train and Save Machine Learning Models (Logistic Regression, Random Forest, SVM, KNN)

# Example: Train a Random Forest Classifier using flattened images
# Preprocessing: Flattening images to a 1D vector for ML models

# Load and preprocess a few images for ML model training (as a simple example)
# Convert images to arrays and flatten them for traditional ML models
X_images = []
y_labels = []

for directory in ['plastic_bottle', 'cardboard_box', 'trash_bin']:  # Adjust these classes as per your dataset
    for file_name in train_generator.filenames:
        img_path = f'path_to_images/{directory}/{file_name}'
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        X_images.append(img_array.flatten())  # Flatten the image
        y_labels.append(directory)

X_images = np.array(X_images)
y_labels = np.array(y_labels)

# Standardize the features (flattened images)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_images)

# Train and save each model
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_labels, test_size=0.3, random_state=42)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')  # Save the model
    print(f"{name} model saved!")

# Evaluate the models
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

# 3. Train and Save ResNet50 for Image Classification

# Load the ResNet50 model pre-trained on ImageNet
resnet50_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3), classes=train_generator.num_classes)

# Compile the model
resnet50_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the ResNet50 model on your data
resnet50_model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Save the ResNet50 model
resnet50_model.save('resnet50_model.h5')
print("ResNet50 model saved!")

# 4. Train and Save Custom CNN Model

# Build a custom CNN architecture
custom_cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
custom_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the custom CNN model on your data
custom_cnn_model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Save the custom CNN model
custom_cnn_model.save('custom_cnn_model.h5')
print("Custom CNN model saved!")
