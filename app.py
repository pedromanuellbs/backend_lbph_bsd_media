from flask import Flask, request, jsonify
import os
import cv2
import logging
import json
import time

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate, FACES_DIR, MODEL_PATH, LABELS_MAP_PATH
from google_drive_utils import download_photo_from_link, cleanup_temp_file

# Setup logging for Railway logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures logs appear in Railway
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# pastikan folder dataset ada
os.makedirs(FACES_DIR, exist_ok=True)

def save_face_image(user_id, image_file):
    """Simpan gambar upload ke folder faces/<user_id>/N.jpg"""
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len(os.listdir(user_dir))
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(dst)
    return dst

def load_model_and_labels():
    """Load LBPH model dan mapping label→user_id dari filesystem."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_MAP_PATH):
        return None, {}
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    label_map = {}
    with open(LABELS_MAP_PATH) as f:
        for line in f:
            lbl, uid = line.strip().split(":")
            label_map[int(lbl)] = uid
    return model, label_map

@app.route('/match_photos', methods=['POST'])
def match_photos():
    """Match photos from Google Drive links against trained faces"""
    try:
        data = request.get_json()
        if not data or 'photos' not in data:
            return jsonify({
                'success': False, 
                'error': 'No photos provided. Expected JSON with "photos" array.'
            }), 400

        photo_links = data['photos']
        if not isinstance(photo_links, list):
            return jsonify({
                'success': False, 
                'error': 'Photos must be an array of Google Drive links.'
            }), 400

        logger.info(f"Starting photo matching pipeline with {len(photo_links)} photos")
        
        # Load trained model
        model, label_map = load_model_and_labels()
        if model is None or not label_map:
            logger.error("Model not trained - cannot perform photo matching")
            return jsonify({
                'success': False, 
                'error': 'Model not trained yet. Please register faces first.'
            }), 400

        logger.info(f"Loaded LBPH model with {len(label_map)} registered users")
        
        # Process each photo
        results = []
        successful_matches = 0
        failed_photos = 0
        
        for i, photo_link in enumerate(photo_links):
            photo_filename = f"photo_{i+1}.jpg"
            result = {
                'photo_index': i + 1,
                'photo_link': photo_link,
                'filename': photo_filename,
                'success': False,
                'error': None,
                'match_found': False,
                'recognized_user': None,
                'lbph_confidence': None,
                'is_match': False
            }
            
            logger.info(f"Processing photo {i+1}/{len(photo_links)}: {photo_link}")
            
            try:
                # Download photo from Google Drive
                temp_path = download_photo_from_link(photo_link, photo_filename)
                if not temp_path:
                    result['error'] = 'Download error - Could not fetch photo from Google Drive'
                    logger.error(f"Download error for photo {i+1}: {result['error']}")
                    results.append(result)
                    failed_photos += 1
                    continue
                
                # Process the downloaded image
                gray = detect_and_crop(temp_path)
                
                # Clean up temp file
                cleanup_temp_file(temp_path)
                
                if gray is None:
                    result['error'] = 'Face detection error - No face detected in photo'
                    logger.error(f"Face detection error for photo {i+1}: {result['error']}")
                    results.append(result)
                    failed_photos += 1
                    continue
                
                # Perform face recognition with LBPH
                try:
                    label, confidence = model.predict(gray)
                    user_id = label_map.get(label, "Unknown")
                    
                    # LBPH uses lower confidence for better matches (threshold typically 50-80)
                    confidence_threshold = 80.0
                    is_match = confidence < confidence_threshold and user_id != "Unknown"
                    
                    result.update({
                        'success': True,
                        'match_found': is_match,
                        'recognized_user': user_id if is_match else None,
                        'lbph_confidence': float(confidence),
                        'is_match': is_match
                    })
                    
                    if is_match:
                        successful_matches += 1
                        logger.info(f"MATCH FOUND for photo {i+1} ({photo_filename}): "
                                  f"User={user_id}, LBPH_confidence={confidence:.2f}, Match=True")
                    else:
                        logger.info(f"NO MATCH for photo {i+1} ({photo_filename}): "
                                  f"User={user_id}, LBPH_confidence={confidence:.2f}, Match=False")
                    
                    results.append(result)
                    
                except Exception as e:
                    result['error'] = f'LBPH prediction error: {str(e)}'
                    logger.error(f"LBPH prediction error for photo {i+1}: {result['error']}")
                    results.append(result)
                    failed_photos += 1
                    
            except Exception as e:
                result['error'] = f'General processing error: {str(e)}'
                logger.error(f"General processing error for photo {i+1}: {result['error']}")
                results.append(result)
                failed_photos += 1
                
        # Summary statistics
        total_photos = len(photo_links)
        processed_photos = total_photos - failed_photos
        
        logger.info(f"Photo matching pipeline completed: "
                   f"Total={total_photos}, Processed={processed_photos}, "
                   f"Failed={failed_photos}, Matches={successful_matches}")
        
        return jsonify({
            'success': True,
            'total_photos': total_photos,
            'processed_photos': processed_photos,
            'failed_photos': failed_photos,
            'successful_matches': successful_matches,
            'match_rate': f"{(successful_matches / processed_photos * 100):.1f}%" if processed_photos > 0 else "0%",
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Critical error in photo matching pipeline: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Critical pipeline error: {str(e)}'
        }), 500

@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get('user_id')
    image   = request.files.get('image')
    
    logger.info(f"Face registration request for user: {user_id}")
    
    if not user_id or not image:
        logger.error("Face registration failed - Missing user_id or image")
        return jsonify({'success': False, 'error': 'Missing user_id or image'}), 400

    try:
        # simpan upload + crop/grayscale
        raw_path = save_face_image(user_id, image)
        logger.info(f"Saved uploaded image to: {raw_path}")
        
        cropped = detect_and_crop(raw_path)
        if cropped is None:
            logger.error(f"Face registration failed - No face detected in uploaded image for user {user_id}")
            return jsonify({'success': False, 'error': 'No face detected'}), 400
            
        # overwrite file dengan hasil crop grayscale 96×96
        cv2.imwrite(raw_path, cropped)
        logger.info(f"Successfully processed and saved face image for user {user_id}")

        # latih ulang seluruh dataset, hitung metrik, simpan model & label_map
        logger.info(f"Starting model retraining after registering face for user {user_id}")
        metrics = train_and_evaluate()
        
        if 'error' in metrics:
            logger.error(f"Model training failed: {metrics['error']}")
            return jsonify({'success': False, 'error': metrics['error']}), 500
            
        logger.info(f"Model retraining completed successfully. Metrics: {metrics}")
        
        return jsonify({
            'success': True,
            'message': f'Face registered for {user_id}',
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error during face registration for user {user_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'Registration failed: {str(e)}'}), 500

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    
    logger.info("Face verification request received")
    
    if not image:
        logger.error("Face verification failed - No image provided")
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        # simpan sementara & preprocess
        tmp_path = "tmp_verify.jpg"
        image.save(tmp_path)
        logger.info(f"Saved verification image to temporary file: {tmp_path}")
        
        gray = detect_and_crop(tmp_path)
        os.remove(tmp_path)
        
        if gray is None:
            logger.error("Face verification failed - No face detected in verification image")
            return jsonify({'success': False, 'error': 'No face detected'}), 400

        # predict
        model, label_map = load_model_and_labels()
        if model is None or not label_map:
            logger.error("Face verification failed - Model not trained yet")
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 400

        label, confidence = model.predict(gray)
        user_id = label_map.get(label, "Unknown")
        
        logger.info(f"Face verification completed - User: {user_id}, LBPH confidence: {confidence:.2f}")
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        logger.error(f"Error during face verification: {str(e)}")
        return jsonify({'success': False, 'error': f'Verification failed: {str(e)}'}), 500

@app.route('/')
def home():
    return "BSD Media LBPH Backend siap!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
