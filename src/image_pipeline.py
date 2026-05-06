import cv2
import numpy as np
from PIL import Image

def center_digit(img):
    """
    Centers the digit based on its Center of Mass (Moments).
    This is crucial for distinguishing between 0, 6, and 9.
    """
    # Calculate moments to find the center of mass
    M = cv2.moments(img)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        rows, cols = img.shape
        shiftX = cols/2 - cX
        shiftY = rows/2 - cY
        
        # Apply the translation matrix to center the image
        M_shift = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        img = cv2.warpAffine(img, M_shift, (cols, rows))
    return img

def preprocess_image(image):
    """
    Standardizes user-uploaded images with Centroid alignment.
    """
    # 1. Convert PIL image to grayscale numpy array
    img = np.array(image.convert('L'))

    # 2. Thresholding / Inverting
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. Center of Mass Alignment (Prevents 0 from being misclassified as 6/9)
    img = center_digit(img)

    # 5. Normalization and Reshaping
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1)) # Shape: (1, 28, 28, 1)

    return img