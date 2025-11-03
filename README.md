# Vehicle Damage Detection with Dynamic ROI Cropping

<img src="utils/AP_per_class.jpg" width="600">


## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Quick Start Summary
<div>

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michaljac/car_damage.git
   cd car_damage
   ```

2. **Build the image**
- **Windows:**
```bash
docker build -t car_damage:v1 -f Dockerfile .
```

3. **Run the container**
- **Windows:**

```bash
docker run --rm -v "$(pwd):/car_damage" -v "$(pwd)/../Data/car_damage:/Data" car_damage:v1
```

4. **Prepare dataset**
Download dataset zip from Google Drive.
Place it inside the container so you'll have:

```bash
.../car_damage:/car_damage
.../Data/car_damage:/Data
```

After extracting, rename the folder "Dataset" → "coco", so the structure is:

```
/Data/coco
├── train2017/
├── val2017/
├── test2017/
└── annotations/
    ├── annotations_train.json
    ├── annotations_val.json
    └── annotations_test.json
```

4. **Preprocess**
             
1.
```bash
python pipelines/preprocess.py
git clone https://github.com/open-mmlab/mmdetection.git
```

2. move car_roi_crop.py from pipelines/car_roi_crop.py to mmdetection/mmdet/datasets/transforms/car_roi_crop.py

5. *** train ****

- train with rtmdet_s model integrate with Wandb

```bash
PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH \
python /car_damage/mmdetection/tools/train.py \
/car_damage/configs/rtmdet_s_car_roi.py \
--launcher none \
--cfg-options visualizer.vis_backends.1.init_kwargs.project="car_damage_detection"
```

- train with faster_rcnn model

```bash
PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH python /car_damage/mmdetection/tools/train.py /car_damage/configs/faster_rcnn_car_roi.py
```

6. *** inference ****
```bash
PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH python /car_damage/pipelines/inference.py /car_damage/configs/inference_cfg.py
```

7. *** benchmark ****
a. run:

```bash
PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH \
python /car_damage/mmdetection/tools/test.py \
/car_damage/configs/eval_cfg.py \
work_dirs/rtmdet_s_car_roi/epoch_150.pth \
--show-dir work_dirs/rtmdet_s_car_roi/eval_vis
```

b. saving the plots:
from the benchmark step, copy the relative path of the json file and paste it in "save_plot.py" file and run:
```bash
python pipelines/save_plot.py
```

## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Project Structure

```
car_damage/
├── Dockerfile
├── requirements.txt
├── README.md
│
├── configs/
│   ├── preprocess_cfg.yaml
│   ├── inference_cfg.py                # no LoadAnnotations (no GT)
│   ├── eval_cfg.py
│   ├── faster_rcnn_car_roi.py
│   ├── rtmdet_s_car_roi.py
│   └── rtmdet_tiny_8xb32-300e_coco.py  # car detector (ROI stage)
│
├── mmdetection/
│   ├── mmdet/
│   │   └── datasets/transforms/
│   │       └── car_roi_crop.py         # custom "on-the-fly" ROI transform
│   └── tools/
│       ├── train.py                    # training
│       └── test.py                     # evaluation
│
├── pipelines/
│   ├── preprocess.py
│   ├── inference.py
│   └── save_plot.py                    # evaluation metrics + plots
│
├── tests/
│   ├── uni_preprocess.py
│   ├── uni_train.py
│   ├── uni_inference.py
│   └── uni_evaluate.py
│
├── utils/
└── work_dirs/                          # model checkpoints & results