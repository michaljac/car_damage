# mmdet/datasets/pipelines/crop_car_roi.py
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import cv2

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.logging import MMLogger

# Lazy imports only when needed (for test-time prepass)
_prepass_model = None
_prepass_cfg = None
_prepass_ckpt = None

def _xyxy_expand(x1, y1, x2, y2, w, h, expand=1.10) -> Tuple[int,int,int,int]:
    cx = 0.5*(x1+x2)
    cy = 0.5*(y1+y2)
    bw = (x2-x1) * expand
    bh = (y2-y1) * expand
    nx1 = max(0, int(round(cx-bw/2)))
    ny1 = max(0, int(round(cy-bh/2)))
    nx2 = min(w, int(round(cx+bw/2)))
    ny2 = min(h, int(round(cy+bh/2)))
    return nx1, ny1, nx2, ny2

def _largest_box(boxes: np.ndarray) -> Optional[np.ndarray]:
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return boxes[np.argmax(areas)]

def _init_prepass_model_if_needed(det_config: str, det_ckpt: str, device: str):
    global _prepass_model, _prepass_cfg, _prepass_ckpt
    if _prepass_model is not None:
        return
    from mmdet.apis import init_detector
    _prepass_cfg = det_config
    _prepass_ckpt = det_ckpt
    _prepass_model = init_detector(det_config, det_ckpt, device=device)

def _infer_car_boxes(img, score_thr: float):
    from mmdet.apis import inference_detector
    # Returns list of class arrays; concat and filter to car class id if needed.
    result = inference_detector(_prepass_model, img)
    # MMDet v3’s RTMDet returns DetDataSample; handle both
    bboxes = None
    labels = None
    if hasattr(result, 'pred_instances'):
        ins = result.pred_instances
        bboxes = ins.bboxes.cpu().numpy()
        labels = ins.labels.cpu().numpy()
        scores = ins.scores.cpu().numpy()
    else:
        # legacy structure (list[np.ndarray] per class)
        # Flatten with labels
        all_b = []
        all_l = []
        for cls_idx, arr in enumerate(result):
            if arr is None or len(arr)==0:
                continue
            all_b.append(arr[:,:4])
            all_l.append(np.full((len(arr),), cls_idx))
        if all_b:
            bboxes = np.concatenate(all_b, axis=0)
            labels = np.concatenate(all_l, axis=0)
            scores = np.ones((len(labels),), dtype=np.float32)
        else:
            bboxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            scores = np.zeros((0,), dtype=np.float32)

    keep = scores >= score_thr
    return bboxes[keep], labels[keep], scores[keep]

@TRANSFORMS.register_module()
class CropCarROI:
    """
    On-the-fly crop-to-car ROI for training (by GT) and test/val (by prepass detector).
    - Train: filters GT labels to car_class_id, picks largest car, crops image, remaps/ clips all GT boxes into crop frame.
    - Test/Val: runs a lightweight prepass detector to get car boxes, chooses largest, crops image before the main model.

    Args:
        mode: 'train' or 'test'
        car_class_id: int class id for 'car' in your dataset (for TRAIN GT filtering)
        expand_ratio: float expansion around the car box
        min_side, max_side: optional size clamps after crop (applied later by Resize anyway; kept for safety)
        keep_if_no_car: if True and no car found, return original image; if False, center-crop to a square.
        prepass:
            det_config: path to prepass car-detector config (only used when mode='test')
            det_ckpt: path to checkpoint
            device: 'cuda:0' or 'cpu'
            score_thr: prepass score threshold
            car_label_ids: optional list of label ids that count as 'car' in prepass (default: any)
    """
    def __init__(
        self,
        mode: str,
        car_class_id: int = 0,
        expand_ratio: float = 1.10,
        min_side: int = 0,
        max_side: int = 0,
        keep_if_no_car: bool = True,
        prepass: Optional[dict] = None
    ):
        assert mode in ('train', 'test')
        self.mode = mode
        self.car_class_id = car_class_id
        self.expand_ratio = expand_ratio
        self.min_side = min_side
        self.max_side = max_side
        self.keep_if_no_car = keep_if_no_car

        self.prepass = prepass or {}
        self._logger = MMLogger.get_current_instance()

        if self.mode == 'test':
            det_cfg = self.prepass.get('det_config', '')
            det_ckpt = self.prepass.get('det_ckpt', '')
            device = self.prepass.get('device', 'cuda:0')
            if not (os.path.exists(det_cfg) and os.path.exists(det_ckpt)):
                raise FileNotFoundError(
                    f'CropCarROI(test): prepass config/ckpt not found: {det_cfg}, {det_ckpt}')
            _init_prepass_model_if_needed(det_cfg, det_ckpt, device)
            self.score_thr = float(self.prepass.get('score_thr', 0.3))
            self.prepass_car_label_ids = self.prepass.get('car_label_ids', None)

    def _crop_apply(self, img: np.ndarray, crop_xyxy: Tuple[int,int,int,int], results: dict):
        x1, y1, x2, y2 = crop_xyxy
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # Shift / clip GT boxes for training
        if self.mode == 'train' and 'gt_bboxes' in results:
            b = results['gt_bboxes'].astype(np.float32)
            # shift
            b[:, [0,2]] -= x1
            b[:, [1,3]] -= y1
            # clip
            b[:, 0::2] = np.clip(b[:, 0::2], 0, cw - 1)
            b[:, 1::2] = np.clip(b[:, 1::2], 0, ch - 1)
            # drop degenerate
            valid = (b[:,2] > b[:,0]) & (b[:,3] > b[:,1])
            for k in ('gt_bboxes', 'gt_bboxes_labels'):
                if k in results:
                    results[k] = results[k][valid]
            if 'gt_ignore_flags' in results:
                results['gt_ignore_flags'] = results['gt_ignore_flags'][valid]
            results['gt_bboxes'] = b[valid]

        results['img'] = crop
        results['img_shape'] = crop.shape
        results['ori_shape'] = crop.shape
        results['pad_shape'] = crop.shape

        # Some downstream components expect HorizontalBoxes
        if 'gt_bboxes' in results and not isinstance(results['gt_bboxes'], HorizontalBoxes):
            results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        return results

    def transform(self, results: dict):
        img = results['img']
        h, w = img.shape[:2]

        if self.mode == 'train':
            gt_bboxes = results.get('gt_bboxes', None)
            gt_labels = results.get('gt_bboxes_labels', None)
            if gt_bboxes is None or gt_labels is None or len(gt_labels) == 0:
                return results if self.keep_if_no_car else results

            # pick largest GT car
            car_inds = np.where(gt_labels == self.car_class_id)[0]
            if len(car_inds) == 0:
                return results if self.keep_if_no_car else results
            car_boxes = np.asarray(gt_bboxes)[car_inds]
            car = _largest_box(car_boxes)
            x1, y1, x2, y2 = map(int, car)
            nx1, ny1, nx2, ny2 = _xyxy_expand(x1, y1, x2, y2, w, h, self.expand_ratio)
            return self._crop_apply(img, (nx1, ny1, nx2, ny2), results)

        # TEST/VAL: run prepass detector, select largest car-like box
        bboxes, labels, scores = _infer_car_boxes(img, self.score_thr)
        if bboxes is None or len(bboxes) == 0:
            return results if self.keep_if_no_car else results

        sel = np.arange(len(bboxes))
        if self.prepass_car_label_ids is not None:
            mask = np.isin(labels, np.array(self.prepass_car_label_ids))
            sel = np.where(mask)[0]
            if len(sel) == 0:
                # fallback: use *any* bbox
                sel = np.arange(len(bboxes))

        cand = bboxes[sel]
        car = _largest_box(cand)
        if car is None:
            return results if self.keep_if_no_car else results

        x1, y1, x2, y2 = map(int, car[:4])
        nx1, ny1, nx2, ny2 = _xyxy_expand(x1, y1, x2, y2, w, h, self.expand_ratio)
        return self._crop_apply(img, (nx1, ny1, nx2, ny2), results)
