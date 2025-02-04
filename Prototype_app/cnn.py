import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Set the directories for images and labels
IMAGE_DIR = 'C:\\Users\\ATMA\\Downloads\\1\\Dataset\\images'
LABEL_DIR = 'C:\\Users\\ATMA\\Downloads\\1\\Dataset\\labels'
IMG_SIZE = 224  # Resize to 224x224 for CNN input

# Load and preprocess images
def load_images(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
    images = []
    labels = []

    for image_file in image_files:
        # Load image
        img = cv2.imread(os.path.join(image_dir, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to the input size
        images.append(img)
        
        # Load corresponding label
        label_file = image_file.replace('.jpeg', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = f.read().strip()
            labels.append(label)

    return np.array(images), np.array(labels)

# Visualize a few images and labels
def visualize_images(image_dir, label_dir, num_images=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
    
    plt.figure(figsize=(10, 10))
    
    for i, image_file in enumerate(image_files[:num_images]):
        # Load image
        img = cv2.imread(os.path.join(image_dir, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to the input size

        # Load corresponding label
        label_file = image_file.replace('.jpeg', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = f.read().strip()
        
        # Plot the image and label
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    
    plt.show()

# Load data
X, y = load_images(IMAGE_DIR, LABEL_DIR)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Visualize some images
visualize_images(IMAGE_DIR, LABEL_DIR)

# Define simplified data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator to the images
datagen.fit(X)

# Calculate class weights to handle class imbalance
class_weights = {i: max(np.bincount(y_encoded)) / count for i, count in enumerate(np.bincount(y_encoded))}
print("Class Weights:", class_weights)

# Define the refined CNN model
def create_refined_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    model.add(layers.Dense(num_classes, activation='softmax'))  # Use 'softmax' for multi-class classification
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and compile the refined model
refined_model = create_refined_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(np.unique(y_encoded)))

# Use a learning rate scheduler to adjust the learning rate during training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train the refined model
history = refined_model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                            epochs=10, 
                            validation_data=(X_val, y_val),
                            class_weight=class_weights,
                            callbacks=[lr_scheduler])

# Save the refined model
refined_model.save('refined_cnn_model.h5')

# Save the label encoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot the results
plot_history(history)
