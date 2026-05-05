import os
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing
import matplotlib.pyplot as plt

def train_digit_model(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=5):
    # 1. Load and Normalize Dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # 2. Tightened Augmentation (Prevents 0 from shifting into a 6/9 position)
    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,  # Reduced from 0.1
        height_shift_range=0.05, # Reduced from 0.1
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # 3. Deeper CNN Architecture
    model = models.Sequential([
        # Layer 1: More filters for better edge/tail detection
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Layer 2: Deeper feature extraction
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        # Larger Dense layer for complex feature combination
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3), # Increased dropout to prevent overfitting on loop shapes
        layers.Dense(10, activation='softmax')
    ])

    # 4. Compile and Train
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # 5. Save Artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(os.path.join('models', 'digit_model.h5'))

    # 6. Save Plot
    plt.figure(figsize=(10, 4))
    metric_name = metrics[0] if isinstance(metrics, list) else metrics
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metric_name], label='Train')
    plt.plot(history.history[f'val_{metric_name}'], label='Val')
    plt.title('Performance')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join('models', 'accuracy_plot.png'))
    plt.close()