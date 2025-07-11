import os
import numpy as np
import cv2
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from face_preprocessing import detect_and_crop

logger = logging.getLogger(__name__)

# Constants
FACES_DIR = "faces"
MODEL_PATH = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

def build_dataset():
    X, y, label_map = [], [], {}
    cur = 0
    
    logger.info(f"Building dataset from {FACES_DIR}")
    
    if not os.path.exists(FACES_DIR):
        logger.error(f"Faces directory {FACES_DIR} does not exist")
        return np.array([]), np.array([]), {}
    
    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue
            
        if uid not in label_map:
            label_map[uid] = cur
            cur += 1
            
        lbl = label_map[uid]
        user_images = 0
        
        for fn in os.listdir(user_dir):
            path = os.path.join(user_dir, fn)
            gray = detect_and_crop(path)
            if gray is None:
                logger.warning(f"Could not process image {path} for user {uid}")
                continue
            X.append(gray)
            y.append(lbl)
            user_images += 1
            
        logger.info(f"Added {user_images} images for user {uid}")
    
    logger.info(f"Dataset built: {len(X)} total images, {len(label_map)} users")
    return np.array(X), np.array(y), label_map

def train_and_evaluate():
    """Train LBPH model and evaluate it, return metrics"""
    logger.info("Starting LBPH model training and evaluation")
    
    X, y, label_map = build_dataset()
    
    if len(X) == 0:
        logger.error("No training data available")
        return {'error': 'No training data available'}
    
    # Create reverse mapping for saving
    reverse_map = {v: k for k, v in label_map.items()}
    
    # Split data for evaluation
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Split dataset: {len(X_train)} training, {len(X_test)} testing samples")
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        logger.info("Using same data for training and testing (insufficient data for split)")
    
    # Train LBPH model
    logger.info("Training LBPH Face Recognizer")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating LBPH model on test set")
    predictions = []
    confidences = []
    
    for i, face in enumerate(X_test):
        label, confidence = model.predict(face)
        predictions.append(label)
        confidences.append(confidence)
        
        # Log individual prediction details
        predicted_user = reverse_map.get(label, "Unknown")
        actual_user = reverse_map.get(y_test[i], "Unknown")
        logger.info(f"Test sample {i+1}: Predicted={predicted_user}, "
                   f"Actual={actual_user}, LBPH_confidence={confidence:.2f}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    avg_confidence = np.mean(confidences)
    
    logger.info(f"LBPH model evaluation completed: "
               f"Accuracy={accuracy:.3f}, Avg_confidence={avg_confidence:.2f}")
    
    # Save model and labels
    logger.info(f"Saving LBPH model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    logger.info(f"Saving label mappings to {LABELS_MAP_PATH}")
    with open(LABELS_MAP_PATH, 'w') as f:
        for label, user_id in reverse_map.items():
            f.write(f"{label}:{user_id}\n")
    
    metrics = {
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'samples_trained': len(X_train),
        'samples_tested': len(X_test),
        'users_count': len(label_map)
    }
    
    logger.info(f"Training completed successfully. Final metrics: {metrics}")
    return metrics
