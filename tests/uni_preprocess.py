"""
Unified Preprocessing Test for CarROICrop
Tests vehicle detection, cropping, and bbox adjustment with visualizations.

Usage:
    python tests/uni_preprocess.py

Configuration:
    Edit the variables below to customize behavior
"""

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, '/workspace')
sys.path.insert(0, 'mmdetection')

from mmdet.apis import init_detector, inference_detector


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Dataset paths (will try each until it finds one that exists)
DATASET_PATHS = [
    # '/Data/coco/train2017/',
    # '/Data/coco/val2017/',
    '/Data/coco/test2017/']

# Annotation file (COCO format)
ANNOTATION_PATHS = [
    # '/Data/coco/annotations/annotations_train.json',
    # '/Data/coco/annotations/annotations_val.json',
    '/Data/coco/annotations/annotations_test.json',

]

# Output directory for test results
OUTPUT_DIR = 'tests/preprocess_test_output'

# Vehicle detector (RTMDet-tiny for speed)
VEHICLE_DETECTOR_CONFIG = 'mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
VEHICLE_DETECTOR_CHECKPOINT = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# COCO vehicle class IDs
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck (0-indexed)
COCO_CLASS_NAMES = {
    1: 'bicycle',
    2: 'car', 
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Detection settings
SCORE_THRESHOLD = 0.3
SELECTION_METHOD = 'highest_confidence'  # 'highest_confidence' or 'largest_area'

# Crop settings
PADDING_RATIO = 0.05  # 5% - Optimal for car damage detection (tight but safe)
SQUARE_CROP = True
MIN_CROP_SIZE = 100

# Test settings
PERCENT_TO_VISUALIZE = 0.1  # 1% of dataset
MAX_IMAGES = 100  # Maximum images to process
DEVICE = 'cpu'  # 'cuda:0' or 'cpu'


# ============================================================================
# Helper Functions
# ============================================================================

def find_dataset():
    """Find the first existing dataset path."""
    for path in DATASET_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find dataset in any of: {DATASET_PATHS}")


def find_annotations():
    """Find the first existing annotation file."""
    for path in ANNOTATION_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find annotations in any of: {ANNOTATION_PATHS}")


def load_coco_annotations(ann_file):
    """Load COCO annotations."""
    import json
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Build image_id -> annotations mapping
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Build image_id -> image_info mapping
    img_id_to_info = {img['id']: img for img in data['images']}
    
    return img_to_anns, img_id_to_info, data['categories']


def detect_vehicle(detector, img, score_threshold=0.3):
    """
    Detect vehicles in image and return the best one.
    
    Returns:
        bbox (np.ndarray): [x1, y1, x2, y2] or None if no vehicle found
        score (float): confidence score
        class_id (int): detected class ID
    """
    result = inference_detector(detector, img)
    
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    # Filter for vehicle classes and score threshold
    valid_mask = np.zeros(len(labels), dtype=bool)
    for vehicle_class in VEHICLE_CLASSES:
        valid_mask |= (labels == vehicle_class)
    valid_mask &= (scores >= score_threshold)
    
    if not np.any(valid_mask):
        return None, 0.0, -1
    
    valid_bboxes = bboxes[valid_mask]
    valid_scores = scores[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Select based on method
    if SELECTION_METHOD == 'highest_confidence':
        idx = np.argmax(valid_scores)
    else:  # largest_area
        areas = (valid_bboxes[:, 2] - valid_bboxes[:, 0]) * \
                (valid_bboxes[:, 3] - valid_bboxes[:, 1])
        idx = np.argmax(areas)
    
    return valid_bboxes[idx], valid_scores[idx], valid_labels[idx]


def apply_padding(bbox, img_shape, padding_ratio=0.1, square_crop=True, min_size=100):
    """
    Apply padding to bbox and optionally make it square.
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_shape: (H, W) or (H, W, C)
        padding_ratio: fraction of bbox size to add as padding
        square_crop: whether to make crop square
        min_size: minimum crop dimension
    
    Returns:
        crop_bbox: [x1, y1, x2, y2] padded and clipped to image
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    
    # Calculate current dimensions
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # Add padding
    pad_w = bbox_w * padding_ratio
    pad_h = bbox_h * padding_ratio
    
    x1 = x1 - pad_w
    y1 = y1 - pad_h
    x2 = x2 + pad_w
    y2 = y2 + pad_h
    
    # Make square if needed
    if square_crop:
        crop_w = x2 - x1
        crop_h = y2 - y1
        max_dim = max(crop_w, crop_h, min_size)
        
        # Center the crop
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        x1 = center_x - max_dim / 2
        y1 = center_y - max_dim / 2
        x2 = center_x + max_dim / 2
        y2 = center_y + max_dim / 2
    
    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return np.array([x1, y1, x2, y2])


def adjust_bboxes(bboxes, crop_bbox, labels=None):
    """
    Adjust bboxes from original image coordinates to crop coordinates.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] in original image
        crop_bbox: [x1, y1, x2, y2] of the crop region
        labels: Optional list of labels corresponding to bboxes
    
    Returns:
        adjusted_bboxes: List of [x1, y1, x2, y2] in crop coordinates
        adjusted_labels: List of labels (only if labels provided)
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    adjusted = []
    adjusted_labels = []
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Adjust to crop coordinates
        new_x1 = x1 - cx1
        new_y1 = y1 - cy1
        new_x2 = x2 - cx1
        new_y2 = y2 - cy1
        
        # Clip to crop boundaries
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(cx2 - cx1, new_x2)
        new_y2 = min(cy2 - cy1, new_y2)
        
        # Only keep if bbox still has positive area
        if new_x2 > new_x1 and new_y2 > new_y1:
            adjusted.append([new_x1, new_y1, new_x2, new_y2])
            if labels is not None:
                adjusted_labels.append(labels[i])
    
    if labels is not None:
        return adjusted, adjusted_labels
    return adjusted


def draw_bboxes(img, bboxes, labels=None, category_map=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image with class labels."""
    img_vis = img.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
        
        if labels is not None and i < len(labels):
            # Get class name from category map if available
            if category_map is not None and labels[i] in category_map:
                label_text = f"{category_map[labels[i]]}"
            else:
                label_text = f"ID:{labels[i]}"
            
            # Draw background for text
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            
            # Draw text
            cv2.putText(img_vis, label_text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_vis


# ============================================================================
# Main Test Function
# ============================================================================

def test_preprocess():
    """Test CarROICrop preprocessing with visualizations."""
    
    print("="*70)
    print("CarROICrop Preprocessing Test")
    print("="*70)
    
    # Find dataset
    dataset_path = find_dataset()
    ann_file = find_annotations()
    print(f"\n✓ Dataset: {dataset_path}")
    print(f"✓ Annotations: {ann_file}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'original'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'cropped'), exist_ok=True)
    print(f"✓ Output: {OUTPUT_DIR}")
    
    # Load annotations
    print("\n⏳ Loading annotations...")
    img_to_anns, img_id_to_info, categories = load_coco_annotations(ann_file)
    print(f"✓ Loaded {len(img_id_to_info)} images")
    
    # Create category ID to name mapping
    category_map = {cat['id']: cat['name'] for cat in categories}
    
    # Initialize vehicle detector
    print(f"\n⏳ Initializing vehicle detector (RTMDet-tiny on {DEVICE})...")
    detector = init_detector(
        VEHICLE_DETECTOR_CONFIG,
        VEHICLE_DETECTOR_CHECKPOINT,
        device=DEVICE
    )
    print("✓ Detector ready")
    
    # Get all image files
    all_images = sorted([f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))])
    
    # Sample images
    num_to_process = min(int(len(all_images) * PERCENT_TO_VISUALIZE), MAX_IMAGES)
    sampled_images = random.sample(all_images, num_to_process)
    
    print(f"\n📊 Processing {num_to_process}/{len(all_images)} images ({PERCENT_TO_VISUALIZE*100:.1f}%)")
    print(f"Selection method: {SELECTION_METHOD}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print("="*70 + "\n")
    
    # Statistics
    stats = {
        'total': 0,
        'vehicle_detected': 0,
        'no_vehicle': 0,
        'saved': 0
    }
    
    # Process images
    for img_name in tqdm(sampled_images, desc="Processing"):
        stats['total'] += 1
        
        # Load image
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Find corresponding annotations
        img_id = int(img_name.split('.')[0].lstrip('0') or '0')
        anns = img_to_anns.get(img_id, [])
        
        # Convert COCO annotations to bboxes [x1, y1, x2, y2]
        original_bboxes = []
        original_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: [x, y, width, height]
            original_bboxes.append([x, y, x + w, y + h])
            original_labels.append(ann['category_id'])
        
        # Detect vehicle
        vehicle_bbox, vehicle_score, vehicle_class = detect_vehicle(
            detector, img, SCORE_THRESHOLD
        )
        
        if vehicle_bbox is None:
            stats['no_vehicle'] += 1
            continue
        
        stats['vehicle_detected'] += 1
        
        # Apply padding and get crop bbox
        crop_bbox = apply_padding(
            vehicle_bbox,
            img.shape,
            PADDING_RATIO,
            SQUARE_CROP,
            MIN_CROP_SIZE
        )
        
        # Crop image
        cx1, cy1, cx2, cy2 = map(int, crop_bbox)
        cropped_img = img[cy1:cy2, cx1:cx2]
        
        # Adjust bboxes to crop coordinates (with labels)
        adjusted_bboxes, adjusted_labels = adjust_bboxes(original_bboxes, crop_bbox, original_labels)
        
        # Visualize original with vehicle detection
        img_with_vehicle = img.copy()
        vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
        cv2.rectangle(img_with_vehicle, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)  # Red for vehicle
        vehicle_name = COCO_CLASS_NAMES.get(vehicle_class, f"class_{vehicle_class}")
        cv2.putText(img_with_vehicle, f"{vehicle_name}: {vehicle_score:.2f}",
                   (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw original bboxes with class labels
        img_with_bboxes = draw_bboxes(img_with_vehicle, original_bboxes, original_labels, category_map, (0, 255, 0), 2)
        
        # Draw crop region
        cv2.rectangle(img_with_bboxes, (cx1, cy1), (cx2, cy2), (255, 0, 0), 3)  # Blue for crop
        
        # Visualize cropped with adjusted bboxes and class labels
        cropped_with_bboxes = draw_bboxes(cropped_img, adjusted_bboxes, adjusted_labels, category_map, (0, 255, 0), 2)
        
        # Save visualizations
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, 'original', img_name),
            img_with_bboxes
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, 'cropped', img_name),
            cropped_with_bboxes
        )
        
        stats['saved'] += 1
    
    # Print results
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Total processed:     {stats['total']}")
    print(f"Vehicle detected:    {stats['vehicle_detected']} ({stats['vehicle_detected']/stats['total']*100:.1f}%)")
    print(f"No vehicle:          {stats['no_vehicle']} ({stats['no_vehicle']/stats['total']*100:.1f}%)")
    print(f"Visualizations saved: {stats['saved']}")
    print(f"\n✓ Output directory: {OUTPUT_DIR}")
    print(f"  - original/: Original images with vehicle detection (red), crop region (blue), and ground truth bboxes (green)")
    print(f"  - cropped/:  Cropped images with adjusted bboxes (green)")
    print("="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    test_preprocess()

