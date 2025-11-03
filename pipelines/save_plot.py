import os, json, pickle, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# CONFIG
ann_file   = '/Data/coco/annotations/annotations_test.json'
pred_pkl   = 'work_dirs/rtmdet_s_car_roi/eval_results.pkl'
out_dir    = 'work_dirs/rtmdet_s_car_roi/eval_analysis'
os.makedirs(out_dir, exist_ok=True)

classes = ('dent','scratch','crack','glass shatter','lamp broken','tire flat')

# LOAD PREDICTIONS
results = pickle.load(open(pred_pkl, 'rb'))

coco_json_results = []
for det in results:
    if not isinstance(det, dict) or 'pred_instances' not in det:
        continue

    preds = det['pred_instances']
    bboxes = np.array(preds['bboxes'])
    scores = np.array(preds['scores'])
    labels = np.array(preds['labels'])

    # robust image ID extraction
    img_id = None
    for key in ['img_id', 'image_id', 'id']:
        if key in det:
            img_id = det[key]
            break
    if img_id is None:
        # try nested metadata
        meta = det.get('metainfo', det.get('data_sample', {}))
        img_id = meta.get('img_id', meta.get('ori_id', -1))
    img_id = int(img_id)

    # skip empty images
    if bboxes.size == 0:
        continue

    for box, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        coco_json_results.append({
            'image_id': img_id,
            'category_id': int(label) + 1,  # COCO 1-based
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'score': float(score)
        })

# save JSON
pred_json = os.path.join(out_dir, 'predictions.json')
with open(pred_json, 'w') as f:
    json.dump(coco_json_results, f)
print(f"✅ Saved valid COCO predictions JSON: {pred_json}")

# COCO EVAL
coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(pred_json)

e = COCOeval(coco_gt, coco_dt, 'bbox')
e.params.iouThrs = np.array([0.5])   # evaluate at IoU=0.5 (AP50)
e.evaluate()
e.accumulate()
e.summarize()

# PER-CLASS AP
cat_ids = coco_gt.getCatIds()
ap_per_class = []
for i, catId in enumerate(cat_ids):
    precision = e.eval['precision'][:, :, i, 0, 2]  # IoU x recall x class x area x maxdet
    precision = precision[precision > -1]
    ap = np.mean(precision) if precision.size else float('nan')
    ap_per_class.append(ap)

# PLOT
plt.figure(figsize=(10, 4))
sns.barplot(x=classes, y=ap_per_class)
plt.ylabel('AP@0.5')
plt.title('Per-class Average Precision (IoU ≥ 0.5)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{out_dir}/per_class_AP50.png')

for cls, ap in zip(classes, ap_per_class):
    print(f'{cls:<15}  AP50={ap:.3f}')

print(f"\nEvaluation complete. Plots saved to {out_dir}")
