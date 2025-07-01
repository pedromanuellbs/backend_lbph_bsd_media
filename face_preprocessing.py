from facenet_pytorch import MTCNN
from PIL import Image
import cv2

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    """
    Detect a single face from the image at img_path, crop & resize to 96×96,
    convert to grayscale, and return as a NumPy array.
    Returns None if no face is detected.
    """
    # Open image and ensure RGB
    img = Image.open(img_path).convert("RGB")
    # Detect and crop face; returns a tensor of shape (3,96,96) or (1,3,96,96)
    face = mtcnn(img)
    if face is None:
        return None

    # Remove batch dimension if present: (1,C,H,W) -> (C,H,W)
    if hasattr(face, 'dim') and face.dim() == 4:
        face = face.squeeze(0)

    # If only 1 channel, expand to 3 channels
    if face.size(0) == 1:
        face = face.expand(3, -1, -1)

    # Convert to H×W×C uint8 numpy
    rgb = face.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray
