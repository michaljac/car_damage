import os
import numpy as np
import cv2
import torch
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes

@TRANSFORMS.register_module()
class CarROICrop(BaseTransform):
    """Crop image around the first vehicle (category_id=vehicle_class_id),
    adjust all GT boxes accordingly, and optionally save debug images.

    Assumes bboxes are already in [x1, y1, x2, y2] format (from LoadAnnotations).
    """

    def __init__(self, vehicle_class_id=7, save_debug=False):
        self.vehicle_class_id = vehicle_class_id
        self.save_debug = save_debug

    # --- helper: extract numpy boxes ---
    @staticmethod
    def _to_numpy_boxes(gt_bboxes):
        """Convert HorizontalBoxes or tensor/array to numpy."""
        if hasattr(gt_bboxes, 'tensor'):
            return gt_bboxes.tensor.cpu().numpy()
        elif hasattr(gt_bboxes, 'numpy'):
            return gt_bboxes.numpy()
        else:
            return np.asarray(gt_bboxes, dtype=np.float32)

    # --- helper: clip boxes to crop bounds ---
    @staticmethod
    def _clip_boxes(boxes, w, h):
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        return boxes

    def old_transform(self, results: dict) -> dict:
        """Crop image around first vehicle bbox; fallback to full image if no vehicle."""
        img = results['img']
        gt_labels = results.get('gt_bboxes_labels', results.get('gt_labels'))
        gt_bboxes_raw = results['gt_bboxes']

        # Convert to numpy safely (handles torch, HorizontalBoxes, etc.)
        if hasattr(gt_bboxes_raw, "tensor"):
            gt_b = gt_bboxes_raw.tensor.detach().cpu().numpy().copy()
        elif isinstance(gt_bboxes_raw, torch.Tensor):
            gt_b = gt_bboxes_raw.detach().cpu().numpy().copy()
        else:
            gt_b = np.asarray(gt_bboxes_raw, dtype=np.float32)

        if gt_b.size == 0:
            return results  # no boxes at all

        # --- find first vehicle
        vehicle_mask = gt_labels == self.vehicle_class_id
        if not np.any(vehicle_mask):
            # fallback to original full image
            H, W = img.shape[:2]
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        vehicle_bbox = gt_b[vehicle_mask][0].astype(int)
        x1c, y1c, x2c, y2c = vehicle_bbox

        H, W = img.shape[:2]
        # clamp to image boundaries
        x1c = int(np.clip(x1c, 0, W - 1))
        y1c = int(np.clip(y1c, 0, H - 1))
        x2c = int(np.clip(x2c, x1c + 1, W))
        y2c = int(np.clip(y2c, y1c + 1, H))

        if y2c <= y1c or x2c <= x1c:
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        cropped = img[y1c:y2c, x1c:x2c]
        h_crop, w_crop = cropped.shape[:2]

        adj = gt_b.copy()
        adj[:, [0, 2]] -= x1c
        adj[:, [1, 3]] -= y1c
        adj[:, [0, 2]] = np.clip(adj[:, [0, 2]], 0, w_crop)
        adj[:, [1, 3]] = np.clip(adj[:, [1, 3]], 0, h_crop)

        keep = (adj[:, 2] - adj[:, 0] > 1) & (adj[:, 3] - adj[:, 1] > 1)
        if not np.any(keep):
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        adj = adj[keep]
        adj_labels = gt_labels[keep]

        results['img'] = cropped
        results['img_shape'] = cropped.shape
        results['gt_bboxes'] = HorizontalBoxes(torch.as_tensor(adj, dtype=torch.float32))
        results['gt_bboxes_labels'] = adj_labels
        results['crop_bbox'] = np.array([x1c, y1c, x2c, y2c], dtype=np.float32)

        return results

    def transform(self, results: dict) -> dict:
        """Crop image around vehicle bbox if present (GT from annotations)."""
        img = results['img']
        H, W = img.shape[:2]

        # Only crop if ground-truth boxes exist (training/eval)
        gt_bboxes_raw = results.get('gt_bboxes', None)
        gt_labels = results.get('gt_bboxes_labels', results.get('gt_labels', None))

        # --- CASE 1: No annotations (inference) ---
        if gt_bboxes_raw is None or gt_labels is None:
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        # Convert to numpy
        if hasattr(gt_bboxes_raw, "tensor"):
            gt_b = gt_bboxes_raw.tensor.detach().cpu().numpy().copy()
        elif isinstance(gt_bboxes_raw, torch.Tensor):
            gt_b = gt_bboxes_raw.detach().cpu().numpy().copy()
        else:
            gt_b = np.asarray(gt_bboxes_raw, dtype=np.float32)

        # --- CASE 2: find vehicle ---
        vehicle_mask = gt_labels == self.vehicle_class_id
        if not np.any(vehicle_mask):
            # fallback: no vehicle found
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        vehicle_bbox = gt_b[vehicle_mask][0].astype(int)
        x1c, y1c, x2c, y2c = np.clip(vehicle_bbox, [0, 0, 1, 1], [W, H, W, H])

        if y2c <= y1c or x2c <= x1c:
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        cropped = img[y1c:y2c, x1c:x2c]
        h_crop, w_crop = cropped.shape[:2]

        # Adjust boxes and labels to crop
        adj = gt_b.copy()
        adj[:, [0, 2]] -= x1c
        adj[:, [1, 3]] -= y1c
        adj[:, [0, 2]] = np.clip(adj[:, [0, 2]], 0, w_crop)
        adj[:, [1, 3]] = np.clip(adj[:, [1, 3]], 0, h_crop)

        keep = (adj[:, 2] - adj[:, 0] > 1) & (adj[:, 3] - adj[:, 1] > 1)
        if not np.any(keep):
            results['crop_bbox'] = np.array([0, 0, W, H], dtype=np.float32)
            return results

        results['img'] = cropped
        results['img_shape'] = cropped.shape
        results['gt_bboxes'] = HorizontalBoxes(torch.as_tensor(adj[keep], dtype=torch.float32))
        results['gt_bboxes_labels'] = gt_labels[keep]
        results['crop_bbox'] = np.array([x1c, y1c, x2c, y2c], dtype=np.float32)

        return results


    # --- new helper: reproject crop and boxes back to original image ---
    @staticmethod
    def map_back_to_original(original_img, crop_bbox, adjusted_bboxes, color=(0,255,0)):
        vis = original_img.copy()
        x1, y1, _, _ = crop_bbox.astype(int)
        for x1b, y1b, x2b, y2b in adjusted_bboxes.astype(int):
            cv2.rectangle(vis, (x1+x1b, y1+y1b), (x1+x2b, y1+y2b), color, 2)
        return vis

    def __repr__(self) -> str: return f"{self.__class__.__name__}(vehicle_class_id={self.vehicle_class_id}, save_debug={self.save_debug})"