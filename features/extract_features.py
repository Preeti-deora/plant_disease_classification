import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image):

    image = cv2.resize(image, (128, 128))


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), feature_vector=True)

   
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

 
    return np.hstack([hog_features, hist])
