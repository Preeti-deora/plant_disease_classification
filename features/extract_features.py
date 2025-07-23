import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image):
    image = image.resize((128, 128))  # Must match training
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        gray_image = rgb2gray(image_np)
    elif image_np.ndim == 2:
        gray_image = image_np
    else:
        raise ValueError("Unsupported image format")

    hog_features = hog(
        gray_image,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return hog_features

