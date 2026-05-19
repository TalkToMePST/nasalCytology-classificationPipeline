This repository contains the Jupyter notebooks used for data preprocessing, object segmentation, mask matching, and deep learning classification models to identify cell types in nasal mucosa cytology images.

## Repository Structure

### 1. Segmentation & Mask Matching (SAM)
These notebooks leverage foundation models to localize cells and compute region-based validation metrics.
* **`SAM_Kaggle_Test.ipynb`**
  Implements inference using the Segment Anything Model (SAM) framework. It evaluates cell segmentation quality by computing Mean Intersection over Union (IoU) metrics against target regions.
* **`fork-of-mask-bb-matcher.ipynb`**
  Utilizes the Segment Anything 2 (SAM 2) library to align generated segmentation masks with object bounding boxes. Includes visualization routines that apply alpha-blending to overlay transparent masks over the original cytology images.

### 2. Classification Models & Comparative Baselines
These notebooks handle structural classification training pipelines, address extreme dataset imbalances, and compute validation performance breakdowns.

* **`Classwise_Copies_IncepV4.ipynb`**
  The primary classification network script. Implements an Inception-v4 architecture utilizing a class-aware sample replication strategy to balance highly skewed minority classes. Features a 4-fold cross-validation training and assessment loop.
* **`Classwise_Copies_R50-Verification.ipynb`**
  Implements and trains the ResNet50 baseline classification network, generating overall accuracy metrics alongside detailed per-class performance verification.
* **`Classwise_Copies_ViT-Verification.ipynb`**
  Implements and evaluates the Vision Transformer (ViT) baseline model framework, reporting independent per-class validation metrics.
* **`Classwise_Copies_MNET.ipynb` / `MNET_ALB_PLUS.ipynb`**
  MobileNet classification setups configured for image regularization. Integrates the `Albumentations` library to apply complex pixel-level and spatial augmentations to improve model generalization.
* **`Mask_MNET_Class.ipynb`**
  A dedicated MobileNet configuration designed to train exclusively on masked cell regions, forcing the model to optimize for localized morphology and internal cellular structures rather than background context.
* **`mnet_50_test.ipynb`**
  Evaluation script that loads trained MobileNet checkpoints and executes inference on holdout test datasets to record final model precision and accuracy.

## Environment Requirements
* Python 3.10+
* PyTorch (with CUDA support recommended)
* Segment Anything 2 (SAM 2)
* Albumentations
* OpenCV (cv2)
* Scikit-Learn
* Pandas, NumPy, and Matplotlib
