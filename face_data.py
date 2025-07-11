import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from face_preprocessing import detect_and_crop

# Constants
FACES_DIR = "faces"
MODEL_PATH = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

def build_dataset():
    X, y, label_map = [], [], {}
    cur = 0
    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue
        if uid not in label_map:
            label_map[uid] = cur
            cur += 1
        lbl = label_map[uid]
        for fn in os.listdir(user_dir):
            path = os.path.join(user_dir, fn)
            gray = detect_and_crop(path)
            if gray is None:
                continue
            X.append(gray)
            y.append(lbl)
    return np.array(X), np.array(y), label_map

def train_and_evaluate():
    """Train LBPH model and evaluate it, return metrics"""
    X, y, label_map = build_dataset()
    
    if len(X) == 0:
        return {'error': 'No training data available'}
    
    # Create reverse mapping for saving
    reverse_map = {v: k for k, v in label_map.items()}
    
    # Split data for evaluation
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train LBPH model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, y_train)
    
    # Evaluate on test set
    predictions = []
    confidences = []
    
    for i, face in enumerate(X_test):
        label, confidence = model.predict(face)
        predictions.append(label)
        confidences.append(confidence)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    avg_confidence = np.mean(confidences)
    
    # Save model and labels
    model.save(MODEL_PATH)
    with open(LABELS_MAP_PATH, 'w') as f:
        for label, user_id in reverse_map.items():
            f.write(f"{label}:{user_id}\n")
    
    return {
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'samples_trained': len(X_train),
        'samples_tested': len(X_test),
        'users_count': len(label_map)
    }
