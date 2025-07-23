import cv2
from features.extract_features import extract_features
from joblib import load

model = load('models/classifier.pkl')
le = load('models/label_encoder.pkl')

def predict(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    features = extract_features_from_array(image)
    pred = model.predict([features])[0]
    return le.inverse_transform([pred])[0]

def extract_features_from_array(image):
    from features.extract_features import extract_color_histogram, extract_hog_features
    hog_feat = extract_hog_features(image)
    color_feat = extract_color_histogram(image)
    return np.hstack([hog_feat, color_feat])
