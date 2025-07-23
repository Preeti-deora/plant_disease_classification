import cv2
import numpy as np
from skimage.feature import hog

def extract_features(img):
    img = img.resize((128, 128)).convert("RGB")
    gray_img = img.convert("L")
    feature, _ = hog(np.array(gray_img),
                     orientations=9,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2),
                     block_norm='L2-Hys',
                     visualize=True)
    return feature


