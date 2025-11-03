"""
Debug script to test CarROICrop transform
"""
import sys
import os
sys.path.insert(0, '/car_damage')
sys.path.insert(0, '/car_damage/mmdetection')

import cv2
import numpy as np
from mmdet.datasets.transforms import CarROICrop

print("=" * 80)
print("DEBUGGING CARROICROP")
print("=" * 80)

# Test 1: Can we import CarROICrop?
print("\n[TEST 1] Importing CarROICrop...")
try:
    from mmdet.datasets.transforms import CarROICrop
    print("✓ CarROICrop imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Can we instantiate it?
print("\n[TEST 2] Instantiating CarROICrop with cuda device...")
try:
    transform = CarROICrop(
        detector_config='mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
        detector_checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
        score_threshold=0.3,
        padding_ratio=0.05,
        device='cuda',  # Using CUDA
        fallback_to_original=True
    )
    print("✓ CarROICrop instantiated")
    print(f"  Device: {transform.device}")
    print(f"  Fallback enabled: {transform.fallback_to_original}")
except Exception as e:
    print(f"✗ Failed to instantiate: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load a real image from the dataset
print("\n[TEST 3] Loading a sample image...")
try:
    img_path = '/Data/coco/train2017/000001.jpg'
    if not os.path.exists(img_path):
        print(f"✗ Image not found: {img_path}")
        sys.exit(1)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"✗ Failed to read image")
        sys.exit(1)
    
    print(f"✓ Image loaded: {img.shape}")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    sys.exit(1)

# Test 4: Try to initialize the detector (lazy loading)
print("\n[TEST 4] Initializing detector (this may take a while)...")
try:
    detector = transform.detector  # This triggers lazy initialization
    if detector is None:
        print("✗ Detector is None - initialization failed silently")
    else:
        print("✓ Detector initialized successfully")
        print(f"  Detector type: {type(detector)}")
except Exception as e:
    print(f"✗ Failed to initialize detector: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Try the transform on the image
print("\n[TEST 5] Running transform on image...")
try:
    # Create a results dict like MMDetection would
    results = {
        'img': img,
        'img_shape': img.shape[:2],
        'ori_shape': img.shape[:2],
        'gt_bboxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
        'gt_labels': np.array([0], dtype=np.int64)
    }
    
    print(f"  Input: img_shape={results['img_shape']}, gt_bboxes={results['gt_bboxes'].shape}")
    
    # Run transform
    output = transform.transform(results)
    
    if output is None:
        print("✗ Transform returned None")
    else:
        print("✓ Transform succeeded")
        print(f"  Output img_shape: {output['img_shape']}")
        print(f"  Crop bbox: {output.get('crop_bbox', 'N/A')}")
        print(f"  Output gt_bboxes: {output.get('gt_bboxes', np.array([])).shape}")
        
except Exception as e:
    print(f"✗ Transform failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test vehicle detection directly
print("\n[TEST 6] Testing vehicle detection directly...")
try:
    vehicle_bbox = transform._detect_vehicles(img)
    if vehicle_bbox is None:
        print("✗ No vehicle detected (or detector failed)")
        print("  This is OK if the image doesn't contain a vehicle")
        print("  With fallback_to_original=True, it should use the original image")
    else:
        print(f"✓ Vehicle detected: {vehicle_bbox}")
except Exception as e:
    print(f"✗ Vehicle detection failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)

