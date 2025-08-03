"""
Bounding box utilities for detection and spatial analysis.
Handles box operations, overlaps, and spatial relationships.
"""

import numpy as np
from typing import List, Tuple, Dict
import math


class BoundingBoxUtils:
    """Utility class for bounding box operations."""
    
    @staticmethod
    def calculate_iou(box1: List[int], box2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def calculate_distance(box1: List[int], box2: List[int]) -> float:
        """
        Calculate distance between centers of two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Euclidean distance between centers
        """
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    @staticmethod
    def expand_bbox(bbox: List[int], margin: int, image_shape: Tuple[int, int]) -> List[int]:
        """
        Expand bounding box by margin while staying within image bounds.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2]
            margin: Margin to add in pixels
            image_shape: Image shape (height, width)
            
        Returns:
            Expanded bounding box
        """
        h, w = image_shape
        return [
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(w, bbox[2] + margin),
            min(h, bbox[3] + margin)
        ]
    
    @staticmethod
    def merge_nearby_boxes(boxes: List[List[int]], distance_threshold: float = 50) -> List[List[int]]:
        """
        Merge bounding boxes that are close to each other.
        
        Args:
            boxes: List of bounding boxes
            distance_threshold: Maximum distance for merging
            
        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []
        
        merged = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
                
            # Start a new merged box
            min_x1, min_y1 = box1[0], box1[1]
            max_x2, max_y2 = box1[2], box1[3]
            used[i] = True
            
            # Find nearby boxes to merge
            for j, box2 in enumerate(boxes):
                if i == j or used[j]:
                    continue
                    
                distance = BoundingBoxUtils.calculate_distance(box1, box2)
                if distance <= distance_threshold:
                    min_x1 = min(min_x1, box2[0])
                    min_y1 = min(min_y1, box2[1])
                    max_x2 = max(max_x2, box2[2])
                    max_y2 = max(max_y2, box2[3])
                    used[j] = True
            
            merged.append([min_x1, min_y1, max_x2, max_y2])
        
        return merged
    
    @staticmethod
    def filter_overlapping_boxes(boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Filter out overlapping bounding boxes using Non-Maximum Suppression.
        
        Args:
            boxes: List of detection dictionaries with 'bounding_box' and 'confidence'
            iou_threshold: IoU threshold for overlap
            
        Returns:
            Filtered list of boxes
        """
        if not boxes:
            return []
        
        # Sort by confidence score (descending)
        sorted_boxes = sorted(boxes, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        while sorted_boxes:
            # Keep the box with highest confidence
            current = sorted_boxes.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            remaining = []
            for box in sorted_boxes:
                iou = BoundingBoxUtils.calculate_iou(
                    current['bounding_box'], 
                    box['bounding_box']
                )
                if iou <= iou_threshold:
                    remaining.append(box)
            
            sorted_boxes = remaining
        
        return keep
    
    @staticmethod
    def find_nearby_text(fixture_bbox: List[int], text_boxes: List[Dict], 
                        distance_threshold: float = 100) -> List[Dict]:
        """
        Find text boxes near a lighting fixture.
        
        Args:
            fixture_bbox: Lighting fixture bounding box
            text_boxes: List of text detection dictionaries
            distance_threshold: Maximum distance for association
            
        Returns:
            List of nearby text boxes
        """
        nearby_texts = []
        
        for text_box in text_boxes:
            text_bbox = text_box['bounding_box']
            distance = BoundingBoxUtils.calculate_distance(fixture_bbox, text_bbox)
            
            if distance <= distance_threshold:
                text_box['distance'] = distance
                nearby_texts.append(text_box)
        
        # Sort by distance
        nearby_texts.sort(key=lambda x: x['distance'])
        
        return nearby_texts
    
    @staticmethod
    def calculate_bbox_area(bbox: List[int]) -> int:
        """
        Calculate area of a bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Area in pixels
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(0, width * height)
    
    @staticmethod
    def is_bbox_valid(bbox: List[int], min_size: int = 10) -> bool:
        """
        Check if bounding box is valid.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            min_size: Minimum width/height
            
        Returns:
            True if valid, False otherwise
        """
        if len(bbox) != 4:
            return False
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return width >= min_size and height >= min_size and width > 0 and height > 0
    
    @staticmethod
    def normalize_bbox(bbox: List[int], image_shape: Tuple[int, int]) -> List[float]:
        """
        Normalize bounding box coordinates to [0, 1] range.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: Image shape (height, width)
            
        Returns:
            Normalized bounding box
        """
        h, w = image_shape
        return [
            bbox[0] / w,
            bbox[1] / h,
            bbox[2] / w,
            bbox[3] / h
        ]
