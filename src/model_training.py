import os
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

def train_digit_model():
    print("--- Loading MNIST Dataset ---")
    # 1. Load Dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # 2. Preprocessing
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape to (batch_size, height, width, channels) for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("--- Building CNN Architecture ---")
    # 3. CNN Implementation
    model = models.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Helps prevent overfitting
        layers.Dense(10, activation='softmax') # 10 classes (0-9)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("--- Starting Training ---")
    # 4. Training (5 epochs is usually enough for >98% accuracy on MNIST)
    history = model.fit(x_train, y_train, 
                        epochs=5, 
                        validation_data=(x_test, y_test))

    # 5. Save the Model
    # Ensure the 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('models', 'digit_model.h5')
    model.save(model_path)
    print(f"--- Model Saved to {model_path} ---")

    # 6. Evaluation Visualization
    plt.figure(figsize=(10, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_digit_model()