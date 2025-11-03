"""
Bbox transformation utilities for inference.
Handles coordinate mapping from model predictions back to original image coordinates.
"""
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any


def adapt_bboxes_to_original_image(
    pred_instances,
    img_meta: Optional[Dict[str, Any]] = None,
    crop_info: Optional[Dict[str, Any]] = None,
    original_shape: Optional[Tuple[int, int]] = None,
    current_shape: Optional[Tuple[int, int]] = None
):
    """
    Adapt predicted bounding boxes back to original image coordinates.
    
    This function handles multiple transformation scenarios:
    1. Resize transformations (handled by MMDetection's scale_factor in img_meta)
    2. Crop transformations (e.g., from CarROICrop)
    3. Manual transformations
    
    Args:
        pred_instances: Prediction instances containing bboxes (can be DetDataSample.pred_instances)
        img_meta: Image metadata from MMDetection containing scale_factor, ori_shape, etc.
        crop_info: Dictionary containing crop information with keys:
            - 'crop_applied': bool, whether crop was applied
            - 'crop_bbox': [x1, y1, x2, y2], crop coordinates in original image
        original_shape: (H, W) of original image (optional, can be derived from img_meta)
        current_shape: (H, W) of current/processed image (optional, can be derived from img_meta)
    
    Returns:
        Transformed pred_instances with bboxes adapted to original image coordinates
    
    Notes:
        - MMDetection automatically handles resize transformations during inference
          by rescaling bboxes using the scale_factor stored in metadata
        - This function primarily handles additional transformations like cropping
        - If using MMDetection's inference_detector(), bboxes are already rescaled to 
          the input image size, so we only need to handle crop offsets
    """
    if pred_instances is None:
        return pred_instances
    
    # Check if empty (handle both length and bbox array size)
    if hasattr(pred_instances, '__len__'):
        if len(pred_instances) == 0:
            return pred_instances
    elif hasattr(pred_instances, 'bboxes'):
        if len(pred_instances.bboxes) == 0:
            return pred_instances
    
    # Get bboxes (support both tensor and numpy)
    bboxes = pred_instances.bboxes
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cuda().numpy()
    
    # Step 1: Handle scale factor from MMDetection (usually already applied)
    # The inference_detector() function in MMDetection automatically rescales
    # predictions back to the original input image size using img_meta['scale_factor']
    # So in most cases, we don't need to do anything here.
    
    # However, if you're working with raw model outputs or want to verify:
    if img_meta is not None and 'scale_factor' in img_meta:
        scale_factor = img_meta['scale_factor']
        # Check if scale_factor is a tuple/list or single value
        if isinstance(scale_factor, (list, tuple, np.ndarray)):
            if len(scale_factor) == 2:
                # (scale_w, scale_h)
                scale_w, scale_h = scale_factor
            elif len(scale_factor) == 4:
                # (scale_w, scale_h, scale_w, scale_h) - XYXY format
                scale_w, scale_h = scale_factor[0], scale_factor[1]
            else:
                scale_w = scale_h = scale_factor[0]
        else:
            scale_w = scale_h = scale_factor
        
        # Note: MMDetection's inference_detector already applies this automatically
        # This is here for reference and edge cases
        # bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / scale_w
        # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / scale_h
    
    # Step 2: Handle crop offset (e.g., from CarROICrop)
    # This is the main transformation we need to apply manually
    if crop_info is not None and crop_info.get('crop_applied', False):
        crop_bbox = crop_info['crop_bbox']
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        
        # Add crop offset to bbox coordinates
        # The predictions are in the cropped image space,
        # so we need to shift them back to original image space
        bboxes[:, [0, 2]] += crop_x1  # x coordinates
        bboxes[:, [1, 3]] += crop_y1  # y coordinates
    
    # Step 3: Handle any additional manual transformations
    if original_shape is not None and current_shape is not None:
        # Manual resize handling (if needed)
        orig_h, orig_w = original_shape
        curr_h, curr_w = current_shape
        
        # Calculate scale factors
        scale_w = orig_w / curr_w
        scale_h = orig_h / curr_h
        
        # Apply scale
        bboxes[:, [0, 2]] *= scale_w
        bboxes[:, [1, 3]] *= scale_h
    
    # Update pred_instances with transformed bboxes
    if isinstance(pred_instances.bboxes, torch.Tensor):
        pred_instances.bboxes = torch.from_numpy(bboxes).to(pred_instances.bboxes.device)
    else:
        pred_instances.bboxes = bboxes
    
    return pred_instances


def get_crop_info_from_meta(img_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract crop information from image metadata.
    
    Args:
        img_meta: Image metadata dictionary
        
    Returns:
        Dictionary with crop_applied and crop_bbox, or None if no crop info found
    """
    if 'crop_bbox' in img_meta and img_meta['crop_bbox'] is not None:
        return {
            'crop_applied': True,
            'crop_bbox': img_meta['crop_bbox']
        }
    return None


def verify_bbox_coordinates(bboxes: np.ndarray, img_shape: Tuple[int, int], clip: bool = True) -> np.ndarray:
    """
    Verify and optionally clip bbox coordinates to image boundaries.
    
    Args:
        bboxes: Array of bboxes in [x1, y1, x2, y2] format, shape (N, 4)
        img_shape: (H, W) of the image
        clip: Whether to clip coordinates to image boundaries
        
    Returns:
        Verified (and optionally clipped) bboxes
    """
    H, W = img_shape
    
    if clip:
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, W)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, H)
    
    # Check for invalid bboxes
    invalid_mask = (bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1])
    if invalid_mask.any():
        print(f"[WARNING] Found {invalid_mask.sum()} invalid bboxes (x2 <= x1 or y2 <= y1)")
    
    return bboxes


def transform_bboxes_simple(
    bboxes: np.ndarray,
    crop_offset: Optional[Tuple[float, float]] = None,
    scale_factor: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Simple bbox transformation with crop offset and scale factor.
    
    Args:
        bboxes: Array of bboxes in [x1, y1, x2, y2] format, shape (N, 4)
        crop_offset: (offset_x, offset_y) to add to coordinates
        scale_factor: (scale_x, scale_y) to multiply coordinates
        
    Returns:
        Transformed bboxes
    """
    bboxes = bboxes.copy()
    
    # Apply scale first
    if scale_factor is not None:
        scale_x, scale_y = scale_factor
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
    
    # Then apply crop offset
    if crop_offset is not None:
        offset_x, offset_y = crop_offset
        bboxes[:, [0, 2]] += offset_x
        bboxes[:, [1, 3]] += offset_y
    
    return bboxes

