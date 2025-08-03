"""
Image preprocessing utilities for electrical drawings.
Handles image enhancement, noise reduction, and format conversions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging
import os
import platform
from pdf2image import convert_from_path
from PIL import Image

from ..utils.config import Config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing for electrical drawing analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to enhance image quality.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Enhance contrast
        enhanced = self._enhance_contrast(denoised)
        
        # Apply sharpening
        sharpened = self._apply_sharpening(enhanced)
        
        return sharpened
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to enhance edges.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> list:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Images
        """
        try:
            # Set Poppler path for different environments
            poppler_path = None
            
            # For Render/Linux environments, try common Poppler locations
            if platform.system() == "Linux" or os.getenv('RENDER'):
                common_paths = [
                    "/usr/bin",
                    "/usr/local/bin", 
                    "/opt/poppler/bin",
                    "/bin"
                ]
                
                # Also try to find pdftoppm using which command
                try:
                    import subprocess
                    result = subprocess.run(["which", "pdftoppm"], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        which_path = os.path.dirname(result.stdout.strip())
                        if which_path not in common_paths:
                            common_paths.insert(0, which_path)
                        logger.info(f"Found pdftoppm via which: {result.stdout.strip()}")
                except Exception as e:
                    logger.debug(f"Could not use 'which' command: {e}")
                
                for path in common_paths:
                    if os.path.exists(os.path.join(path, "pdftoppm")):
                        poppler_path = path
                        logger.info(f"Found Poppler at: {poppler_path}")
                        break
                        
                if not poppler_path:
                    logger.warning("Poppler path not found, using system PATH")
            
            # Convert PDF with or without explicit Poppler path
            if poppler_path:
                images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
                
            if not images:
                raise Exception("No pages found in PDF")
                
            logger.info(f"Converted PDF to {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            
            # Try alternative approach without DPI specification
            try:
                logger.info("Retrying PDF conversion with default settings...")
                if poppler_path:
                    images = convert_from_path(pdf_path, poppler_path=poppler_path)
                else:
                    images = convert_from_path(pdf_path)
                    
                if images:
                    logger.info(f"Alternative conversion successful: {len(images)} pages")
                    return images
                else:
                    raise Exception("No pages found in PDF")
                    
            except Exception as e2:
                logger.error(f"Alternative PDF conversion also failed: {str(e2)}")
                
                # Final attempt: try with minimal parameters and explicit PATH
                try:
                    logger.info("Final attempt: minimal conversion settings...")
                    # Try setting PATH explicitly
                    old_path = os.environ.get('PATH', '')
                    if '/usr/bin' not in old_path:
                        os.environ['PATH'] = f"/usr/bin:{old_path}"
                    
                    images = convert_from_path(pdf_path, dpi=72, first_page=1, last_page=10)
                    
                    if images:
                        logger.info(f"Minimal conversion successful: {len(images)} pages")
                        return images
                    else:
                        raise Exception("Still no pages found")
                        
                except Exception as e3:
                    logger.error(f"All PDF conversion attempts failed: {str(e3)}")
                    return []
    
    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            OpenCV image as numpy array
        """
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL format.
        
        Args:
            cv2_image: OpenCV image as numpy array
            
        Returns:
            PIL Image object
        """
        if len(cv2_image.shape) == 3:
            return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv2_image)
    
    def resize_image(self, image: np.ndarray, max_width: int = 1920, 
                    max_height: int = 1080) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(w * scale)
            new_height = int(h * scale)
            return cv2.resize(image, (new_width, new_height), 
                            interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_regions_of_interest(self, image: np.ndarray) -> dict:
        """
        Extract different regions of interest from electrical drawings.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with different regions
        """
        h, w = image.shape[:2]
        
        regions = {
            'full': image,
            'top_half': image[:h//2, :],
            'bottom_half': image[h//2:, :],
            'left_half': image[:, :w//2],
            'right_half': image[:, w//2:],
            'center': image[h//4:3*h//4, w//4:3*w//4]
        }
        
        return regions
    
    def detect_text_regions(self, image: np.ndarray) -> list:
        """
        Detect regions likely to contain text for OCR.
        
        Args:
            image: Input image
            
        Returns:
            List of text region bounding boxes
        """
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size (typical for text)
            if w > h and w > 50 and h > 10:
                text_regions.append([x, y, x + w, y + h])
        
        return text_regions
