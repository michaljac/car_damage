# Copyright (c) OpenMMLab. All rights reserved.
"""
CarROICrop Transform
On-the-fly vehicle detection and ROI cropping for training, evaluation, and inference.
Detects vehicles (car, bicycle, motorcycle, truck, bus) and crops to largest vehicle region.
"""

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.config import Config
from mmengine.runner import load_checkpoint

# NOTE: Don't import init_detector here to avoid circular import
# It will be imported lazily inside the method that needs it
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample


@TRANSFORMS.register_module()
class CarROICrop(BaseTransform):
    """Crop image to largest vehicle ROI using a pre-trained detector.
    
    This transform detects vehicles (car, bicycle, motorcycle, truck, bus) in the image
    using a pre-trained model, selects the largest detection, crops the image to that
    region with optional padding, and adjusts all bounding box annotations to the new
    coordinate system.
    
    The transform is designed to work "on the fly" during training, evaluation, and
    inference - no preprocessed dataset is created.
    
    Vehicle Classes (COCO):
        - bicycle (id=2)
        - car (id=3)
        - motorcycle (id=4)
        - bus (id=6)
        - truck (id=8)
    
    Required Keys:
        - img (np.ndarray): Input image
        - img_shape (Tuple[int, int]): Image shape (H, W)
        - gt_bboxes (np.ndarray): Ground truth bboxes in [x1, y1, x2, y2] format
        - gt_labels (np.ndarray): Ground truth labels
    
    Modified Keys:
        - img (np.ndarray): Cropped image
        - img_shape (Tuple[int, int]): New image shape
        - ori_shape (Tuple[int, int]): Updated to cropped shape
        - gt_bboxes (np.ndarray): Adjusted bboxes in cropped coordinates
        - gt_labels (np.ndarray): Updated labels (removes out-of-bounds)
        - crop_bbox (np.ndarray): The crop region applied [x1, y1, x2, y2]
    
    Args:
        detector_config (str): Path to pre-trained detector config file.
            Default: 'rtmdet_tiny_8xb32-300e_coco' (will use MMDet's model zoo)
        detector_checkpoint (str): Path to pre-trained detector checkpoint.
            Default: None (will download from MMDet model zoo)
        score_threshold (float): Confidence threshold for vehicle detection.
            Default: 0.3
        padding_ratio (float): Padding ratio around detected vehicle bbox.
            E.g., 0.1 means add 10% padding on each side.
            Default: 0.1
        square_crop (bool): Whether to pad the crop to square dimensions.
            Useful for models that expect square inputs (e.g., RTMDet, Faster R-CNN).
            Default: True
        min_crop_size (int): Minimum crop size (prevents too small crops).
            Default: 100
        device (str): Device to run detector on ('cuda' or 'cuda').
            Default: 'cuda:0'
        fallback_to_original (bool): If no vehicle detected, use original image.
            Default: True
        vehicle_classes (list): COCO class IDs to treat as vehicles.
            Default: [2, 3, 4, 6, 8] (bicycle, car, motorcycle, bus, truck)
        selection_method (str): Method to select vehicle when multiple detected.
            Options: 'largest_area' (default) or 'highest_confidence'
            Default: 'largest_area'
    """
    
    def __init__(
        self,
        detector_config: str = 'rtmdet_tiny_8xb32-300e_coco',
        detector_checkpoint: Optional[str] = None,
        score_threshold: float = 0.3,
        padding_ratio: float = 0.1,
        square_crop: bool = True,
        min_crop_size: int = 100,
        device: str = 'cuda:0',
        fallback_to_original: bool = True,
        vehicle_classes: list = [2, 3, 4, 6, 8],  # bicycle, car, motorcycle, bus, truck
        selection_method: str = 'largest_area',  # 'largest_area' or 'highest_confidence'
    ):
        self.detector_config = detector_config
        self.detector_checkpoint = detector_checkpoint
        self.score_threshold = score_threshold
        self.padding_ratio = padding_ratio
        self.square_crop = square_crop
        self.min_crop_size = min_crop_size
        self.device = device
        self.fallback_to_original = fallback_to_original
        self.vehicle_classes = vehicle_classes
        self.selection_method = selection_method
        
        # Validate selection method
        if self.selection_method not in ['largest_area', 'highest_confidence']:
            raise ValueError(
                f"selection_method must be 'largest_area' or 'highest_confidence', "
                f"got '{self.selection_method}'"
            )
        
        # Lazy initialization of detector (on first use)
        self._detector = None
        
    @property
    def detector(self):
        """Lazy load detector only when needed."""
        if self._detector is None:
            self._init_detector()
        return self._detector
    
    def _init_detector(self):
        """Initialize the pre-trained vehicle detector."""
        try:
            # Lazy import to avoid circular dependency
            from mmdet.apis import init_detector
            
            # Handle MMDet model zoo configs
            if not self.detector_config.endswith('.py'):
                # Try to find config in MMDet model zoo
                from mmdet import model_zoo
                # For now, use the config file directly if provided
                config_path = self.detector_config
            else:
                config_path = self.detector_config
            
            # Initialize detector
            self._detector = init_detector(
                config_path,
                self.detector_checkpoint,
                device=self.device
            )
            
            print(f"[CarROICrop] Initialized vehicle detector: {self.detector_config}")
            print(f"[CarROICrop] Vehicle classes: {self.vehicle_classes}")
            
        except Exception as e:
            warnings.warn(
                f"Failed to initialize detector: {e}. "
                f"CarROICrop will fallback to original images."
            )
            self._detector = None
    
    def _detect_vehicles(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect vehicles in the image and return selected vehicle bbox.
        
        Selection method is determined by self.selection_method:
        - 'largest_area': Select vehicle with largest bounding box area
        - 'highest_confidence': Select vehicle with highest detection confidence
        
        Args:
            img (np.ndarray): Input image in BGR format
            
        Returns:
            Optional[np.ndarray]: Selected vehicle bbox in [x1, y1, x2, y2] format,
                                  or None if no vehicle detected
        """
        if self._detector is None:
            return None
        
        try:
            # Run inference
            from mmdet.apis import inference_detector
            result = inference_detector(self._detector, img)
            
            # Extract predictions
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cuda().numpy()  # [N, 4]
            scores = pred_instances.scores.cuda().numpy()  # [N]
            labels = pred_instances.labels.cuda().numpy()  # [N]
            
            # Filter by vehicle classes and score threshold
            vehicle_mask = np.isin(labels, self.vehicle_classes)
            score_mask = scores >= self.score_threshold
            valid_mask = vehicle_mask & score_mask
            
            if not valid_mask.any():
                return None
            
            # Get valid vehicle detections
            valid_bboxes = bboxes[valid_mask]
            valid_scores = scores[valid_mask]
            
            # Select vehicle based on configured method
            if self.selection_method == 'largest_area':
                # Select by largest bbox area
                areas = (valid_bboxes[:, 2] - valid_bboxes[:, 0]) * \
                        (valid_bboxes[:, 3] - valid_bboxes[:, 1])
                selected_idx = np.argmax(areas)
            elif self.selection_method == 'highest_confidence':
                # Select by highest confidence score
                selected_idx = np.argmax(valid_scores)
            else:
                # Fallback to largest area (should never reach here due to validation)
                areas = (valid_bboxes[:, 2] - valid_bboxes[:, 0]) * \
                        (valid_bboxes[:, 3] - valid_bboxes[:, 1])
                selected_idx = np.argmax(areas)
            
            selected_bbox = valid_bboxes[selected_idx]
            
            return selected_bbox
            
        except Exception as e:
            warnings.warn(f"Vehicle detection failed: {e}")
            return None
    
    def _compute_crop_bbox(
        self,
        vehicle_bbox: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Compute the final crop bbox with padding and optional square adjustment.
        
        Args:
            vehicle_bbox (np.ndarray): Vehicle bbox in [x1, y1, x2, y2] format
            img_shape (Tuple[int, int]): Image shape (H, W)
            
        Returns:
            np.ndarray: Crop bbox in [x1, y1, x2, y2] format
        """
        H, W = img_shape
        x1, y1, x2, y2 = vehicle_bbox
        
        # Calculate current bbox dimensions
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Add padding
        pad_w = bbox_w * self.padding_ratio
        pad_h = bbox_h * self.padding_ratio
        
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(W, x2 + pad_w)
        crop_y2 = min(H, y2 + pad_h)
        
        # Make square if requested
        if self.square_crop:
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            # Determine target size (use the smaller dimension to ensure it fits)
            target_size = max(crop_w, crop_h)
            
            # Center the crop
            center_x = (crop_x1 + crop_x2) / 2
            center_y = (crop_y1 + crop_y2) / 2
            
            # Calculate new crop bounds
            new_x1 = center_x - target_size / 2
            new_x2 = center_x + target_size / 2
            new_y1 = center_y - target_size / 2
            new_y2 = center_y + target_size / 2
            
            # Adjust if crop exceeds image boundaries
            # Shift horizontally if needed
            if new_x1 < 0:
                shift = -new_x1
                new_x1 = 0
                new_x2 = min(W, new_x2 + shift)
            elif new_x2 > W:
                shift = new_x2 - W
                new_x2 = W
                new_x1 = max(0, new_x1 - shift)
            
            # Shift vertically if needed
            if new_y1 < 0:
                shift = -new_y1
                new_y1 = 0
                new_y2 = min(H, new_y2 + shift)
            elif new_y2 > H:
                shift = new_y2 - H
                new_y2 = H
                new_y1 = max(0, new_y1 - shift)
            
            # If still can't fit a perfect square (image too small), use largest possible square
            actual_w = new_x2 - new_x1
            actual_h = new_y2 - new_y1
            
            if actual_w != actual_h:
                # Use the smaller dimension to ensure perfect square within bounds
                final_size = min(actual_w, actual_h)
                
                # Re-center with the constrained size
                if actual_w > actual_h:
                    # Width is larger, reduce it
                    center_x = (new_x1 + new_x2) / 2
                    new_x1 = center_x - final_size / 2
                    new_x2 = center_x + final_size / 2
                else:
                    # Height is larger, reduce it
                    center_y = (new_y1 + new_y2) / 2
                    new_y1 = center_y - final_size / 2
                    new_y2 = center_y + final_size / 2
            
            crop_x1, crop_x2 = new_x1, new_x2
            crop_y1, crop_y2 = new_y1, new_y2
        
        # Ensure minimum crop size
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_w < self.min_crop_size or crop_h < self.min_crop_size:
            # Fallback to original image if crop is too small
            return np.array([0, 0, W, H], dtype=np.float32)
        
        return np.array([crop_x1, crop_y1, crop_x2, crop_y2], dtype=np.float32)
    
    def _crop_image(
        self,
        img: np.ndarray,
        crop_bbox: np.ndarray
    ) -> np.ndarray:
        """Crop image to the specified bbox.
        
        Args:
            img (np.ndarray): Input image
            crop_bbox (np.ndarray): Crop bbox in [x1, y1, x2, y2] format
            
        Returns:
            np.ndarray: Cropped image
        """
        x1, y1, x2, y2 = crop_bbox.astype(int)
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img
    
    def _adjust_bboxes(
        self,
        bboxes: np.ndarray,
        crop_bbox: np.ndarray,
        crop_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust all bounding boxes to the cropped coordinate system.
        
        This function transforms all bboxes to the new cropped coordinates and
        filters out bboxes that fall completely outside the crop region.
        
        Args:
            bboxes (np.ndarray): Original bboxes in [x1, y1, x2, y2] format, shape [N, 4]
            crop_bbox (np.ndarray): Crop region [x1, y1, x2, y2]
            crop_shape (Tuple[int, int]): Cropped image shape (H, W)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Adjusted bboxes in cropped coordinates
                - Valid indices mask
        """
        if len(bboxes) == 0:
            return bboxes, np.array([], dtype=bool)
        
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        crop_h, crop_w = crop_shape
        
        # Adjust bbox coordinates relative to crop
        adjusted_bboxes = bboxes.copy()
        adjusted_bboxes[:, [0, 2]] -= crop_x1  # Adjust x coordinates
        adjusted_bboxes[:, [1, 3]] -= crop_y1  # Adjust y coordinates
        
        # Clamp to crop boundaries
        adjusted_bboxes[:, [0, 2]] = np.clip(adjusted_bboxes[:, [0, 2]], 0, crop_w)
        adjusted_bboxes[:, [1, 3]] = np.clip(adjusted_bboxes[:, [1, 3]], 0, crop_h)
        
        # Calculate bbox areas to filter out invalid boxes
        widths = adjusted_bboxes[:, 2] - adjusted_bboxes[:, 0]
        heights = adjusted_bboxes[:, 3] - adjusted_bboxes[:, 1]
        areas = widths * heights
        
        # Keep bboxes with area > 1 (at least 1 pixel)
        valid_mask = areas > 1.0
        
        return adjusted_bboxes, valid_mask
    
    def transform(self, results: dict) -> dict:
        """Transform function to crop image to vehicle ROI.
        
        Args:
            results (dict): Result dict containing image and annotations
            
        Returns:
            dict: Result dict with cropped image and adjusted annotations
        """
        try:
            img = results['img']
            img_shape = results['img_shape']  # (H, W)
            
            # Detect largest vehicle in the image
            vehicle_bbox = self._detect_vehicles(img)
            
            # If no vehicle detected and fallback is enabled, return original
            if vehicle_bbox is None:
                if self.fallback_to_original:
                    # No cropping needed, return as-is
                    results['crop_bbox'] = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)
                    return results
                else:
                    # Skip this sample (return None will be handled by pipeline)
                    warnings.warn("No vehicle detected and fallback disabled. Returning empty result.")
                    return None
            
            # Compute crop bbox with padding and square adjustment
            crop_bbox = self._compute_crop_bbox(vehicle_bbox, img_shape)
            
            # Crop the image
            cropped_img = self._crop_image(img, crop_bbox)
            crop_h, crop_w = cropped_img.shape[:2]
            
            # Update image in results
            results['img'] = cropped_img
            results['img_shape'] = (crop_h, crop_w)
            results['ori_shape'] = (crop_h, crop_w)
            results['crop_bbox'] = crop_bbox
            
            # Adjust all bounding boxes if they exist
            if 'gt_bboxes' in results:
                gt_bboxes = results['gt_bboxes']
                
                # Handle different bbox formats (numpy array or tensor)
                if isinstance(gt_bboxes, torch.Tensor):
                    gt_bboxes = gt_bboxes.cuda().numpy()
                
                # Convert from any format to [x1, y1, x2, y2] if needed
                # Assuming gt_bboxes are already in [x1, y1, x2, y2] format (standard in MMDet)
                
                adjusted_bboxes, valid_mask = self._adjust_bboxes(
                    gt_bboxes,
                    crop_bbox,
                    (crop_h, crop_w)
                )
                
                # Update bboxes and labels
                results['gt_bboxes'] = adjusted_bboxes[valid_mask]
                
                # Also update labels if they exist
                if 'gt_labels' in results:
                    gt_labels = results['gt_labels']
                    if isinstance(gt_labels, torch.Tensor):
                        gt_labels = gt_labels.cuda().numpy()
                    results['gt_labels'] = gt_labels[valid_mask]
                
                # Update other annotation fields if they exist
                for key in ['gt_bboxes_labels', 'gt_ignore_flags', 'gt_masks']:
                    if key in results:
                        value = results[key]
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            if isinstance(value, torch.Tensor):
                                value = value.cuda().numpy()
                            if len(value) == len(valid_mask):
                                results[key] = value[valid_mask]
            
            return results
            
        except Exception as e:
            # Catch any errors and fallback to original image
            warnings.warn(f"CarROICrop failed with error: {e}. Falling back to original image.")
            img_shape = results.get('img_shape', results['img'].shape[:2])
            results['crop_bbox'] = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)
            return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(detector_config={self.detector_config}, '
        repr_str += f'detector_checkpoint={self.detector_checkpoint}, '
        repr_str += f'score_threshold={self.score_threshold}, '
        repr_str += f'padding_ratio={self.padding_ratio}, '
        repr_str += f'square_crop={self.square_crop}, '
        repr_str += f'vehicle_classes={self.vehicle_classes})'
        return repr_str

