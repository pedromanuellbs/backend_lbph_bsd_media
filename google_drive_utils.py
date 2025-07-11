import os
import requests
import logging
from urllib.parse import urlparse, parse_qs
import tempfile

logger = logging.getLogger(__name__)

def extract_google_drive_file_id(url):
    """Extract file ID from Google Drive URL"""
    try:
        # Handle different Google Drive URL formats
        if 'drive.google.com/file/d/' in url:
            # Direct file link format
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'drive.google.com/open?id=' in url:
            # Open link format  
            parsed_url = urlparse(url)
            file_id = parse_qs(parsed_url.query).get('id', [None])[0]
        elif 'drive.google.com/uc?id=' in url:
            # UC link format
            parsed_url = urlparse(url)
            file_id = parse_qs(parsed_url.query).get('id', [None])[0]
        else:
            logger.error(f"Unsupported Google Drive URL format: {url}")
            return None
        
        logger.info(f"Extracted Google Drive file ID: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Error extracting file ID from URL {url}: {str(e)}")
        return None

def download_from_google_drive(file_id, filename):
    """Download file from Google Drive using file ID"""
    try:
        # Create download URL
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        logger.info(f"Downloading file {filename} from Google Drive ID: {file_id}")
        
        # Make request with session to handle redirects
        session = requests.Session()
        response = session.get(download_url)
        
        # Check if we got a virus scan warning page
        if 'virus scan warning' in response.text.lower():
            # Try to find the confirm token
            for line in response.text.split('\n'):
                if 'confirm=' in line and 'uuid=' in line:
                    confirm_token = line.split('confirm=')[1].split('&')[0]
                    download_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={confirm_token}"
                    response = session.get(download_url)
                    break
        
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"Successfully downloaded {filename} to {temp_file.name}")
        return temp_file.name
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading file {filename}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        return None

def download_photo_from_link(photo_url, filename):
    """Download photo from Google Drive link"""
    try:
        # Extract file ID from Google Drive URL
        file_id = extract_google_drive_file_id(photo_url)
        if not file_id:
            logger.error(f"Could not extract file ID from URL: {photo_url}")
            return None
        
        # Download the file
        temp_path = download_from_google_drive(file_id, filename)
        if not temp_path:
            logger.error(f"Failed to download photo from Google Drive: {photo_url}")
            return None
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Error downloading photo from link {photo_url}: {str(e)}")
        return None

def cleanup_temp_file(temp_path):
    """Clean up temporary file"""
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {temp_path}: {str(e)}")