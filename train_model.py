import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from features.extract_features import extract_features

DATA_DIR = "data/images"
CSV_PATH = "data/train.csv"

df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

joblib.dump(le, "label_encoder.pkl")

features = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(DATA_DIR, row['image_id'] + ".jpg")
    image = cv2.imread(img_path)

    if image is not None:
        feat = extract_features(image)
        features.append(feat)
        labels.append(row['label_encoded'])

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "plant_disease_model.pkl")

print("✅ Model training complete.")
