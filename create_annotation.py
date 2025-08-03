"""
Create annotated screenshot for submission from latest processing results.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pdf2image

def create_submission_annotation():
    """Create annotated screenshot for submission."""
    
    print("🎯 Creating submission annotation...")
    
    # Load the latest detection results
    debug_dir = Path("debug")
    detection_files = list(debug_dir.glob("PDF_detection_results_*.json"))
    
    if not detection_files:
        print("❌ No detection results found!")
        return
    
    # Get the most recent detection file
    latest_detection = max(detection_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using detection results: {latest_detection.name}")
    
    # Load detection data
    with open(latest_detection, 'r') as f:
        detections = json.load(f)
    
    # Load the original PDF
    pdf_path = Path("data/PDF.pdf")
    if not pdf_path.exists():
        print("❌ PDF file not found!")
        return
    
    print("📄 Converting PDF to images...")
    try:
        # Convert first page of PDF to image
        images = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
        if not images:
            print("❌ Failed to convert PDF!")
            return
        
        # Get the first page
        page_image = images[0]
        img_array = np.array(page_image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return
    
    print("🎨 Drawing annotations...")
    
    # Get detections for page 1
    page_1_detections = [d for d in detections if d.get('source_page') == 1]
    print(f"📍 Found {len(page_1_detections)} detections on page 1")
    
    # Draw bounding boxes and labels
    for i, detection in enumerate(page_1_detections[:10]):  # Limit to first 10 for clarity
        bbox = detection.get('bounding_box', [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = bbox
        confidence = detection.get('confidence', 0)
        symbol = detection.get('symbol', 'Unknown')
        
        # Skip very low confidence detections
        if confidence < 0.3:
            continue
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw label background
        label = f"{symbol} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img_bgr, (int(x1), int(y1) - text_height - 10), 
                     (int(x1) + text_width, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(img_bgr, label, (int(x1), int(y1) - 5), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # Add title and summary
    height, width = img_bgr.shape[:2]
    
    # Add title banner
    title_height = 80
    title_img = np.zeros((title_height, width, 3), dtype=np.uint8)
    title_img[:] = (50, 50, 50)  # Dark gray
    
    # Title text
    title_text = "Emergency Lighting Detection - A1E Fixtures"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(title_text, font, font_scale, font_thickness)
    text_x = (width - text_width) // 2
    text_y = (title_height + text_height) // 2
    
    cv2.putText(title_img, title_text, (text_x, text_y), 
               font, font_scale, (255, 255, 255), font_thickness)
    
    # Add summary text
    summary_text = f"Detected: {len(page_1_detections)} fixtures | Type: A1E Emergency Lights"
    font_scale = 0.8
    (text_width, text_height), _ = cv2.getTextSize(summary_text, font, font_scale, 1)
    text_x = (width - text_width) // 2
    text_y = title_height - 15
    
    cv2.putText(title_img, summary_text, (text_x, text_y), 
               font, font_scale, (200, 200, 200), 1)
    
    # Combine title and image
    final_img = np.vstack([title_img, img_bgr])
    
    # Save the annotated image
    output_path = Path("outputs/submission_annotation.png")
    cv2.imwrite(str(output_path), final_img)
    
    print(f"✅ Annotation saved to: {output_path}")
    print(f"📏 Image size: {final_img.shape[1]}x{final_img.shape[0]}")
    print(f"📊 Detections shown: {min(len(page_1_detections), 10)}")
    
    # Also save statistics
    stats = {
        "total_detections": len(detections),
        "page_1_detections": len(page_1_detections),
        "detections_shown": min(len(page_1_detections), 10),
        "pdf_pages": max([d.get('source_page', 1) for d in detections]),
        "fixture_types": list(set([d.get('symbol', 'Unknown') for d in detections]))
    }
    
    with open("outputs/annotation_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("📈 Statistics saved to: outputs/annotation_stats.json")
    return output_path

if __name__ == "__main__":
    create_submission_annotation()
