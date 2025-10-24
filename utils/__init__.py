"""
Utility functions for the car damage detection project.
"""
from .bbox_utils import (
    adapt_bboxes_to_original_image,
    verify_bbox_coordinates,
    transform_bboxes_simple,
    get_crop_info_from_meta
)

__all__ = [
    'adapt_bboxes_to_original_image',
    'verify_bbox_coordinates',
    'transform_bboxes_simple',
    'get_crop_info_from_meta'
]

