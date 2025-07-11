#!/usr/bin/env python3
"""
Demo script showing the photo matching pipeline functionality
"""
import logging
import json
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_data import build_dataset, train_and_evaluate
from google_drive_utils import extract_google_drive_file_id
import face_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_google_drive_parsing():
    """Demo Google Drive URL parsing"""
    print("=== Google Drive URL Parsing Demo ===")
    
    test_urls = [
        "https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/view",
        "https://drive.google.com/open?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "https://drive.google.com/uc?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        "https://invalid-url.com/file/123"
    ]
    
    for url in test_urls:
        file_id = extract_google_drive_file_id(url)
        print(f"URL: {url}")
        print(f"Extracted File ID: {file_id}")
        print()

def demo_dataset_building():
    """Demo dataset building (will show no data initially)"""
    print("=== Dataset Building Demo ===")
    
    X, y, label_map = build_dataset()
    print(f"Dataset size: {len(X)} samples")
    print(f"Users: {len(label_map)}")
    print(f"Label mapping: {label_map}")
    print()

def demo_photo_matching_pipeline():
    """Demo the complete photo matching pipeline workflow"""
    print("=== Photo Matching Pipeline Demo ===")
    
    # Simulate photo matching request
    mock_request = {
        "photos": [
            "https://drive.google.com/file/d/1example123/view",
            "https://drive.google.com/file/d/1example456/view",
            "https://drive.google.com/file/d/1example789/view"
        ]
    }
    
    print(f"Mock request: {json.dumps(mock_request, indent=2)}")
    print()
    
    # Show what would happen in the pipeline
    print("Pipeline steps that would execute:")
    print("1. Load trained LBPH model and label mappings")
    print("2. For each photo URL:")
    print("   - Extract Google Drive file ID")
    print("   - Download photo from Google Drive")
    print("   - Process image with MTCNN for face detection")
    print("   - If face detected, run LBPH face recognition")
    print("   - Log results with confidence scores")
    print("   - Clean up temporary files")
    print("3. Return summary statistics and detailed results")
    print()
    
    # Simulate expected response structure
    mock_response = {
        "success": True,
        "total_photos": 3,
        "processed_photos": 2,
        "failed_photos": 1,
        "successful_matches": 1,
        "match_rate": "50.0%",
        "results": [
            {
                "photo_index": 1,
                "photo_link": "https://drive.google.com/file/d/1example123/view",
                "filename": "photo_1.jpg",
                "success": True,
                "match_found": True,
                "recognized_user": "john_doe",
                "lbph_confidence": 45.2,
                "is_match": True
            },
            {
                "photo_index": 2,
                "photo_link": "https://drive.google.com/file/d/1example456/view",
                "filename": "photo_2.jpg",
                "success": True,
                "match_found": False,
                "recognized_user": None,
                "lbph_confidence": 95.8,
                "is_match": False
            },
            {
                "photo_index": 3,
                "photo_link": "https://drive.google.com/file/d/1example789/view",
                "filename": "photo_3.jpg",
                "success": False,
                "error": "Download error - Could not fetch photo from Google Drive"
            }
        ]
    }
    
    print(f"Expected response structure: {json.dumps(mock_response, indent=2)}")

if __name__ == "__main__":
    print("Photo Matching Pipeline Demo")
    print("=" * 50)
    print()
    
    demo_google_drive_parsing()
    demo_dataset_building()
    demo_photo_matching_pipeline()
    
    print("Demo completed successfully!")
    print("The photo matching pipeline is ready to use.")
    print("Send POST requests to /match_photos with JSON containing 'photos' array of Google Drive links.")