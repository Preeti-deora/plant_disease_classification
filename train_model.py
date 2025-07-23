import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATA_DIR = "data/images"
CSV_PATH = "data/train.csv"
MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

df = pd.read_csv(CSV_PATH)


def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  

    
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), multichannel=True, feature_vector=True)

    
    hist_features = []
    for i in range(3):  
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    return np.concatenate([hog_features, hist_features])


X = []
y = []

for i, row in df.iterrows():
    image_id = row['image_id']
    label = row['label']
    img_path = os.path.join(DATA_DIR, image_id + ".jpg")

    if os.path.exists(img_path):
        features = extract_features(img_path)
        X.append(features)
        y.append(label)
    else:
        print(f"Image not found: {img_path}")

X = np.array(X)


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(clf, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\n✅ Model saved as: {MODEL_PATH}")
print(f"✅ Label encoder saved as: {ENCODER_PATH}")
