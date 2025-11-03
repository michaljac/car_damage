import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
import mmcv
from mmcv.ops import nms
from mmengine import Config
from mmdet.datasets.transforms.car_roi_crop import CarROICrop
from mmdet.apis import init_detector, inference_detector

if __name__ == "__main__":

    # config params
    cfg_path = "configs/inference_cfg.py"
    cfg = Config.fromfile(cfg_path)
    checkpoint = cfg.get("load_from", None)
    device = cfg.get("device", "cuda:0")
    classes_names = cfg['classes_names']
    conf_thr = cfg['conf_thr']
    iou_thr = cfg['iou_thr'] 
    data_root = cfg.get("data_root", "/Data/coco/")
    output_dir = getattr(cfg, "output_dir", "/car_damage/results/")
    test_data = cfg.get("test_dataloader", {}).get("dataset", {})
    input_dir = os.path.join(data_root, test_data.get("data_prefix", {}).get("img", "test2017/"))
    os.makedirs(output_dir, exist_ok=True)

    # inference model
    model = init_detector(cfg_path, checkpoint, device=device)

    if os.path.isdir(input_dir):
        img_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
    else:
        img_paths = [input_dir]

 
    for img_path in tqdm(img_paths):
        result = inference_detector(model, img_path)
        original_img = mmcv.imread(img_path)
        crop_bbox = result.metainfo.get("crop_bbox", None)

        if hasattr(result, "pred_instances") and result.pred_instances is not None:
            bboxes = result.pred_instances.bboxes.detach().cpu().numpy()
            scores = result.pred_instances.scores.detach().cpu().numpy()
            labels = result.pred_instances.labels.detach().cpu().numpy()

            # Filter by confidence
            keep = scores > conf_thr
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

            # Apply NMS
            if len(bboxes) > 0:
                bboxes_t = torch.from_numpy(bboxes).float()
                scores_t = torch.from_numpy(scores).float()

                dets, keep_idx = nms(bboxes_t, scores_t, iou_thr)

                bboxes = bboxes_t[keep_idx].numpy()
                scores = scores_t[keep_idx].numpy()
                labels = labels[keep_idx.numpy()]

            vis = CarROICrop.map_back_to_original(original_img, crop_bbox, bboxes)

            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.astype(int)
                color = (255, 0, 0)
                text = f"{classes_names[label]} {score:.2f}"  # or f"ID {label} {score:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        else:
            vis = original_img

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis)

    print("Inference completed. Results saved to:", output_dir)
