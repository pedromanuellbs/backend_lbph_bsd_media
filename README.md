# BSD Media LBPH Face Recognition Backend

This backend provides face recognition capabilities using MTCNN for face detection and LBPH (Local Binary Patterns Histograms) for face recognition, with support for Google Drive photo matching.

## Features

- **Face Registration**: Register faces for training the recognition model
- **Face Verification**: Verify uploaded faces against trained model
- **Photo Matching Pipeline**: Process photos from Google Drive links and match against trained faces
- **Comprehensive Logging**: Detailed logging for Railway deployment monitoring

## Endpoints

### POST /register_face
Register a new face for training.

**Form Data:**
- `user_id`: String - User identifier
- `image`: File - Face image file

**Response:**
```json
{
  "success": true,
  "message": "Face registered for user123",
  "metrics": {
    "accuracy": 0.95,
    "avg_confidence": 45.2,
    "samples_trained": 80,
    "samples_tested": 20,
    "users_count": 5
  }
}
```

### POST /verify_face
Verify a face against the trained model.

**Form Data:**
- `image`: File - Face image file

**Response:**
```json
{
  "success": true,
  "user_id": "user123",
  "confidence": 42.5
}
```

### POST /match_photos
Match photos from Google Drive links against trained faces.

**JSON Body:**
```json
{
  "photos": [
    "https://drive.google.com/file/d/1example123/view",
    "https://drive.google.com/file/d/1example456/view"
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total_photos": 2,
  "processed_photos": 2,
  "failed_photos": 0,
  "successful_matches": 1,
  "match_rate": "50.0%",
  "results": [
    {
      "photo_index": 1,
      "photo_link": "https://drive.google.com/file/d/1example123/view",
      "filename": "photo_1.jpg",
      "success": true,
      "match_found": true,
      "recognized_user": "john_doe",
      "lbph_confidence": 45.2,
      "is_match": true
    },
    {
      "photo_index": 2,
      "photo_link": "https://drive.google.com/file/d/1example456/view",
      "filename": "photo_2.jpg",
      "success": true,
      "match_found": false,
      "recognized_user": null,
      "lbph_confidence": 95.8,
      "is_match": false
    }
  ]
}
```

## Logging

The system provides comprehensive logging for:

- **Download Errors**: When photos cannot be fetched from Google Drive
- **Read Errors**: When photos cannot be loaded/read by OpenCV
- **Face Detection Errors**: When no face is detected using MTCNN
- **Matching Results**: Photo filename, match status, recognized user, LBPH confidence scores
- **LBPH/MTCNN Accuracy**: Performance metrics at each step

All logs are formatted for Railway deployment visibility.

## Google Drive Integration

Supports multiple Google Drive URL formats:
- `https://drive.google.com/file/d/FILE_ID/view`
- `https://drive.google.com/open?id=FILE_ID`
- `https://drive.google.com/uc?id=FILE_ID`

The system automatically:
1. Extracts file IDs from Google Drive URLs
2. Downloads photos to temporary files
3. Processes images with MTCNN for face detection
4. Runs LBPH face recognition if faces are detected
5. Cleans up temporary files
6. Continues processing even if individual photos fail

## Error Handling

The pipeline is designed to be robust:
- Individual photo failures don't stop the overall process
- Detailed error logging for each failure type
- Graceful handling of network issues, file format problems, and face detection failures
- Automatic cleanup of temporary files

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

The server will start on `http://0.0.0.0:8000`

## Testing

Run the demo to verify functionality:

```bash
python demo_photo_matching.py
```

## Architecture

- **app.py**: Main Flask application with endpoints
- **face_data.py**: LBPH model training and evaluation
- **face_preprocessing.py**: MTCNN face detection and preprocessing
- **google_drive_utils.py**: Google Drive integration utilities
- **demo_photo_matching.py**: Demo script showing pipeline functionality