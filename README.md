# Vehicle Damage Detection with Dynamic ROI Cropping

## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Quick Start Summary
<div>

**🚀 Get Started in 4 Steps:**

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

2. **Build the container**
- **Windows:**

```bash
docker run --rm -v "$(pwd):/car_damage" -v "$(pwd)/../Data/car_damage:/Data" car_damage:v1
```

download dataset zip from google drive.
put it in container so you'll have 
.../car_damage:/car_damage
.../Data/damage:/Data

after extract the dataset zip, and rename the folder "Dataset" to "coco", so the structure is:
/Data/coco
where the images are in: /Data/coco/train2017 (val2017, test2017)
and the annotations are in: //Data/coco/annotations


## <img src="" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Project Structure

```
car_damage/                    # Project root
├── Dockerfile               
├── configs/   
|___|__preprocess_cfg.yaml                  # Configs
|   |__inference_cfg.py                     # w/o loadannotations (GT)
|   |__eval_cfg.py
│   ├──faster_rcnn_car_roi.py                 
│   │__rtmdet_s_car_roi.py
|   |__rtmdet_tiny_8xb32-300e_coco.py      # for inference cars
├── mmdetection/                     
│     |__mmdet/                
│         └── datasets/
|                |__transforms/
|                       |___car_roi_crop.py    # the class "on the fly"
|     |__tools/
|           |__train.py                     # train script
|           |__test.py                      # evaluation script
|
├── pipelines/               
│   ├── preprocess.py    
|   |__ inference.py

├── tests/                  # Test suite
│   ├── uni_preprocess.py    
│   └── uni_train.py
|   |__ uni_inference.py
|   |__uni_evaluate.py                 
├── utils/                  # helpers
├── workdirs/                 # all the runs (unuploaded to git)
├── requirements.txt         # Python dependencies
└── README.md              
```
*** preprocess ****
1. run pipelines/preprocess.py
2. git clone https://github.com/open-mmlab/mmdetection.git
3. move car_roi_crop.py from pipelines/car_roi_crop.py to mmdetection/mmdet/datasets/transforms/car_roi_crop.py

*** train ****
- train with rtmdet_s model integrate with Wandb

* PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH python /car_damage/mmdetection/tools/train.py /car_damage/configs/rtmdet_s_car_roi.py 

* PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH \
python /car_damage/mmdetection/tools/train.py \
  /car_damage/configs/rtmdet_s_car_roi.py \
  --launcher none \
  --cfg-options visualizer.vis_backends.1.init_kwargs.project="car_damage_detection"

- train with faster_rcnn model

PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH python /car_damage/mmdetection/tools/train.py /car_damage/configs/faster_rcnn_car_roi.py

*** inference ****

PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH python /car_damage/pipelines/inference.py /car_damage/configs/inference_cfg.py

*** benchmark ****
run: 
1. PYTHONPATH=/car_damage/mmdetection:$PYTHONPATH \
python /car_damage/mmdetection/tools/test.py \
/car_damage/configs/eval_cfg.py \
work_dirs/rtmdet_s_car_roi/epoch_150.pth \
--show-dir work_dirs/rtmdet_s_car_roi/eval_vis

2. saving the plots
from the benchmark step, copy the relative path of the json file and paste it in "save_plot.py" file and run:
python pipelines/save_plot.py