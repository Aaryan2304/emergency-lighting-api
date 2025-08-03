"""
Simple demo script to test the emergency lighting detection system.
This script processes images from the data folder to demonstrate functionality.
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.detection import LightingDetector, ImageProcessor
from src.extraction import OCREngine, TextProcessor
from src.llm import GroupingEngine
from src.utils import config, setup_logging, get_logger

logger = get_logger(__name__)


async def demo_detection():
    """Demonstrate the detection pipeline with sample images."""
    
    setup_logging()
    logger.info("Starting Emergency Lighting Detection Demo")
    
    # Initialize components
    detector = LightingDetector(config)
    image_processor = ImageProcessor(config)
    ocr_engine = OCREngine(config)
    text_processor = TextProcessor()
    
    # Check if LLM is configured
    try:
        from src.llm.llm_backends import LLMManager
        llm_manager = LLMManager(config)
        available_backends = llm_manager.get_available_backends()
        if available_backends:
            logger.info(f"LLM backends available: {available_backends}")
            logger.info(f"Primary backend: {llm_manager.get_primary_backend()}")
            grouping_engine = GroupingEngine(config)
            llm_available = True
        else:
            logger.warning("No LLM backends available - using rule-based grouping only")
            llm_available = False
    except Exception as e:
        logger.warning(f"Failed to initialize LLM backends: {e} - using rule-based grouping only")
        llm_available = False
    
    # Process images from data folder
    data_folder = Path("data")
    
    if not data_folder.exists():
        logger.error("Data folder not found")
        return
    
    # Find image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_folder.glob(f"*{ext}"))
    
    if not image_files:
        logger.error("No image files found in data folder")
        return
    
    logger.info(f"Found {len(image_files)} image files")
    
    all_detections = []
    
    # Process each image
    for i, image_path in enumerate(image_files):
        try:
            logger.info(f"Processing {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                continue
            
            # Detect lighting fixtures
            detections = detector.detect_emergency_lights(image)
            
            # Add source information
            for detection in detections:
                detection['source_file'] = image_path.name
                detection['source_page'] = 1
                detection['source_sheet'] = f"Sheet_{i+1}"
            
            all_detections.extend(detections)
            
            # Extract text for context
            text_detections = ocr_engine.extract_text_combined(image)
            text_strings = [d['text'] for d in text_detections]
            
            # Extract symbols and specifications
            symbols = text_processor.extract_symbols_from_text(text_strings)
            specs = text_processor.extract_specifications(text_strings)
            
            logger.info(f"  - Found {len(detections)} lighting fixtures")
            logger.info(f"  - Extracted {len(symbols)} symbols: {symbols}")
            logger.info(f"  - Found specifications: {specs}")
            
            # Create annotated image for visualization
            if detections:
                vis_image = detector.visualize_detections(image, detections)
                output_path = Path("outputs") / f"annotated_{image_path.name}"
                output_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(output_path), vis_image)
                logger.info(f"  - Saved annotated image: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Summary
    logger.info(f"\nDetection Summary:")
    logger.info(f"Total detections: {len(all_detections)}")
    
    if all_detections:
        # Group by symbol
        symbol_counts = {}
        confidence_scores = []
        
        for detection in all_detections:
            symbol = detection.get('symbol', 'Unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            confidence_scores.append(detection.get('confidence', 0.0))
        
        logger.info("Symbol distribution:")
        for symbol, count in symbol_counts.items():
            logger.info(f"  - {symbol}: {count} fixtures")
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        # LLM Grouping (if available)
        if llm_available:
            try:
                logger.info("\nAttempting LLM-powered grouping...")
                
                # Create simple rulebook for demo
                demo_rulebook = {
                    'lighting_schedule': {
                        'A1E': {'description': 'Emergency Exit Light', 'mount': 'Wall'},
                        'EM': {'description': 'Emergency Light', 'mount': 'Ceiling'},
                        'LED': {'description': 'LED Fixture', 'mount': 'Recessed'}
                    },
                    'notes': ['Emergency lighting must be connected to emergency power']
                }
                
                # Group fixtures using new LLM manager
                grouped_results = await llm_manager.group_fixtures(all_detections)
                
                logger.info("LLM Grouping Results:")
                for group_name, group_data in grouped_results.items():
                    count = group_data.get('count', 0) if isinstance(group_data, dict) else len([d for d in all_detections if d.get('symbol') == group_name])
                    description = group_data.get('description', 'Emergency lighting fixture') if isinstance(group_data, dict) else 'Emergency lighting fixture'
                    logger.info(f"  - {group_name}: {count} fixtures ({description})")
                
            except Exception as e:
                logger.error(f"LLM grouping failed: {str(e)}")
        
        # Save results
        results = {
            'total_detections': len(all_detections),
            'symbol_counts': symbol_counts,
            'average_confidence': avg_confidence,
            'detections': all_detections[:10]  # Save first 10 for size
        }
        
        results_path = Path("outputs") / "demo_results.json"
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_path}")
    
    else:
        logger.warning("No lighting fixtures detected in any images")
    
    logger.info("Demo completed!")


def create_sample_test_image():
    """Create a sample test image with simple shapes for testing."""
    logger.info("Creating sample test image...")
    
    # Create a white image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some dark rectangles to simulate emergency lighting fixtures
    rectangles = [
        (100, 100, 150, 130),  # Small rectangle
        (200, 150, 280, 180),  # Medium rectangle
        (350, 200, 420, 240),  # Larger rectangle
        (500, 300, 550, 320),  # Thin rectangle
    ]
    
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 50), -1)  # Dark gray fill
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)      # Black border
    
    # Add some text labels
    texts = [
        ("A1E", (120, 95)),
        ("EM", (230, 145)),
        ("EXIT", (370, 195)),
        ("LED", (520, 295))
    ]
    
    for text, (x, y) in texts:
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add a title
    cv2.putText(image, "Emergency Lighting Demo", (250, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save the image
    output_path = Path("data") / "sample_test.png"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    
    logger.info(f"Sample test image created: {output_path}")


async def main():
    """Main demo function."""
    
    # Create sample image if data folder is empty
    data_folder = Path("data")
    
    if not data_folder.exists() or not list(data_folder.glob("*.png")):
        create_sample_test_image()
    
    # Run the detection demo
    await demo_detection()


if __name__ == "__main__":
    asyncio.run(main())
