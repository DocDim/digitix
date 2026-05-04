import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Standardizes user-uploaded images to match MNIST training data format.
    1. Grayscale
    2. Resize to 28x28
    3. Invert colors (if necessary)
    4. Normalize pixel values
    """
    # 1. Convert PIL image to OpenCV format (numpy array)
    # Ensure it's in grayscale ('L' mode in PIL)
    img = np.array(image.convert('L'))

    # 2. Thresholding / Cleaning
    # This helps remove background noise and makes the digit stand out
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Resize to 28x28
    # We use INTER_AREA for shrinking images—it preserves quality better
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. Normalization
    # Rescale pixel values from [0, 255] to [0.0, 1.0]
    img = img.astype('float32') / 255.0

    # 5. Reshape for the CNN
    # The model expects (batch_size, height, width, channels) -> (1, 28, 28, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img