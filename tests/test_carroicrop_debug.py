import numpy as np
import cv2
import torch
import sys, os

# ✅ 1. Add mmdetection repo first in sys.path
sys.path.insert(0, '/workspace/mmdetection')

from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

# ✅ 2. Explicitly import CarROICrop to ensure it's registered
# from mmdet.datasets.transforms.car_roi_crop import CarROICrop
# from mmdet.datasets import transforms #
from mmdet.datasets.transforms.car_roi_crop import CarROICrop
from mmengine.dataset import Compose
from mmdet.structures.bbox import HorizontalBoxes


# ✅ 3. Define manual pipeline
manual_pipeline = [
    dict(type='CarROICrop', vehicle_class_id=7, save_debug=True),
]
pipeline = Compose(manual_pipeline)


# ✅ 4. Prepare sample
img_path = '/Data/coco/test2017/000012.jpg'
img = cv2.imread(img_path)

sample = {
    'img_path': img_path,
    'img': img,
    'img_shape': img.shape,
    'ori_shape': img.shape,
    # COCO loader converts xywh -> xyxy, so we directly use xyxy here
    'gt_bboxes': HorizontalBoxes(torch.tensor([
        [0.0, 0.0, 966.93, 564.42],   # Vehicle
        [32.61, 221.06, 364.33, 553.88],  # Damage 1
        [687.22, 367.63, 861.35, 518.61], # Damage 2
    ], dtype=torch.float32)),
    'gt_bboxes_labels': np.array([7, 6, 6], dtype=np.int64),
}

print(f"📥 Input sample keys: {list(sample.keys())}")
print(f"📥 Input gt_bboxes shape: {sample['gt_bboxes'].tensor.shape}")
print(f"📥 Input gt_bboxes (xyxy):\n{sample['gt_bboxes'].tensor}")
print(f"📥 Input gt_labels: {sample['gt_bboxes_labels']}")
print(f"📥 Input image shape: {sample['img'].shape}")

# ✅ 5. Run the pipeline
result = pipeline(sample)

print(f"\n📤 Output keys: {list(result.keys())}")
print(f"📤 Output gt_bboxes type: {type(result['gt_bboxes'])}")
if isinstance(result['gt_bboxes'], HorizontalBoxes):
    print(f"📤 Output gt_bboxes tensor shape: {result['gt_bboxes'].tensor.shape}")
else:
    print(f"📤 Output gt_bboxes shape: {result['gt_bboxes'].shape}")
print(f"📤 Output gt_labels: {result['gt_bboxes_labels']}")
print(f"📤 Output image shape: {result['img'].shape}")

# ✅ 6. Prepare for visualization
os.makedirs("examples/debug_check", exist_ok=True)
colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]

# --- Original image ---
if isinstance(sample['gt_bboxes'], HorizontalBoxes):
    input_boxes = sample['gt_bboxes'].tensor.cpu().numpy()
else:
    input_boxes = np.asarray(sample['gt_bboxes'], dtype=float)

vis_original = sample['img'].copy()
for idx, (x1, y1, x2, y2) in enumerate(input_boxes.astype(int)):
    color = colors[idx % len(colors)]
    cv2.rectangle(vis_original, (x1, y1), (x2, y2), color, 3)
    label = f"Label:{sample['gt_bboxes_labels'][idx]}"
    cv2.putText(vis_original, label, (x1, max(15, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite("examples/debug_check/03_original_with_bboxes.jpg", vis_original)
print("✅ Saved examples/debug_check/03_original_with_bboxes.jpg")

# --- Cropped image ---
if isinstance(result['gt_bboxes'], HorizontalBoxes):
    output_boxes = result['gt_bboxes'].tensor.cpu().numpy()
else:
    output_boxes = np.asarray(result['gt_bboxes'], dtype=float)

vis_cropped = result['img'].copy()
for idx, (x1, y1, x2, y2) in enumerate(output_boxes.astype(int)):
    color = colors[idx % len(colors)]
    cv2.rectangle(vis_cropped, (x1, y1), (x2, y2), color, 3)
    label = f"Label:{result['gt_bboxes_labels'][idx]}"
    cv2.putText(vis_cropped, label, (x1, max(15, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite("examples/debug_check/04_cropped_with_adjusted_bboxes.jpg", vis_cropped)
print("✅ Saved examples/debug_check/04_cropped_with_adjusted_bboxes.jpg")

# ✅ 7. Visualize pasted-back result (using CarROICrop helper)
crop_bbox = result['crop_bbox']
CarROICrop.paste_crop_back(
    original_img=img,
    cropped_img=result['img'],
    crop_bbox=crop_bbox,
    adjusted_bboxes=output_boxes
)
vis_back = CarROICrop.paste_crop_back(img, result['img'], crop_bbox, output_boxes)
cv2.imwrite("examples/debug_check/05_paste_back_on_original.jpg", vis_back)
print("✅ Saved examples/debug_check/05_paste_back_on_original.jpg")

print("\n🎯 Summary:")
print(f"   - Input bboxes: {len(input_boxes)}")
print(f"   - Output bboxes: {len(output_boxes)}")
print(f"   - Original shape: {sample['img'].shape}")
print(f"   - Cropped shape: {result['img'].shape}")
