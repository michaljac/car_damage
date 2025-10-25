## 🚗 Project Report: Vehicle Damage Detection with Dynamic ROI Cropping

### **1. Model Selection**
Several pretrained backbones were evaluated — **RTMDet-tiny**, **RTMDet-s**, and **Faster R-CNN** — all initialized from COCO weights.  
After comparative experiments, **RTMDet-s** achieved the highest performance, especially for **small-scale damages such as scratches**.  
Its stronger multi-scale feature extraction led to higher precision and recall across most categories.

---

### **2. Preprocessing and ROI Handling**

#### **Data Preprocessing**
To ensure data integrity and compatibility with MMDetection:
- Skipped **incorrect references**, **mislabeled IDs**, and **duplicate annotations**.  
- Clipped bounding boxes that exceeded image dimensions.  
- Reorganized the dataset into **COCO format** under the `coco/` directory.

#### **Dynamic Car ROI Cropping (“On the Fly”)**
A key part of the pipeline was a custom preprocessing transform that dynamically crops vehicle regions during both **training** and **inference**.

- Used a pretrained detector to infer on raw data and **detect vehicle-like objects**, filtering relevant COCO classes:  
  *car, bus, truck, motorcycle,* and *bicycle*.
- For each detection:
  - Cropped the detected car region with slight **padding** to preserve context.  
  - Adjusted all bounding boxes relative to the new cropped image dimensions.
- During training/inference, each image passed through this transform in real time.
- After inference, bounding boxes were **mapped back** to the original full-image coordinates.

The transform implementation is located in:  
`mmdetection/mmdet/datasets/transforms/car_roi_crop.py`

---

### **3. Evaluation Results**
Evaluation metrics (Precision, Recall, mAP, etc.) were plotted per category.  
Due to **class imbalance** and **annotation inconsistencies**, results varied across categories.  
While the preprocessing pipeline validated the structure of each annotation, it did not automatically correct semantically mislabeled but valid categories.  
Despite this, **RTMDet-s consistently outperformed other models**, particularly in fine-grained damage detection.

---

### **4. Challenges and Limitations**
Training was performed on **CPU** due to the absence of a local GPU.  
Although experiments on **Google Colab** and **Kaggle** were attempted, integration between **PyTorch** and **MMDetection** caused dependency conflicts that prevented stable GPU acceleration.  
Consequently, all training and inference were executed on CPU-based models, leading to **slower training** and **reduced model performance** compared to GPU setups.

---

### **Summary**
| Component | Description |
|------------|-------------|
| **Best Model** | RTMDet-s |
| **ROI Strategy** | Dynamic car ROI cropping during training/inference |
| **Transform File** | `mmdetection/mmdet/datasets/transforms/car_roi_crop.py` |
| **Evaluation Metrics** | Precision, Recall, mAP (per category) |
| **Limitation** | CPU-only execution due to missing GPU support |

---

**Next Steps**
- Migrate the setup to a **GPU cloud environment** (RunPod, Lambda, or Paperspace) for full-speed MMDetection training.
- Extend preprocessing to automatically reclassify mislabeled categories.
- Experiment with **data augmentation** on cropped ROIs for enhanced small-object robustness.
