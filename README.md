# ML-VisionRice-Dev
Development notebooks and scripts for an ML pipeline for rice plant counting using computer vision (YOLO).

This repository contains the full history of experiments, data processing, and prototype development that led to the production-ready code in the main repository.


## Thesis and Research

This development work is based on the thesis "Application of Computer Vision Models for Plant Counting in Rice Crops" by Prof. Alejandro Manuel Lloveras, for the Specialization in Artificial Intelligence at the Faculty of Engineering, University of Buenos Aires (FIUBA).

**Author:** Prof. Alejandro Manuel Lloveras

**Advisor:**
  * Ing. Juan Ignacio Cavalieri (FIUBA)
  * Dr. Marcel Bentancor (UdelaR)
  
**Examining Committee:**
  * Dr. Ing. Facundo Luciana (UNT/FIUBA)
  * Esp. Ing. Alfonso Rafel (UNR/FIUBA)
  * Ing. Martín Horn (UNLU/FIUBA)

**Date:** June 2025, Autonomous City of Buenos Aires, Argentina

## Project Description

This project involves the development of an artificial vision pipeline designed for automating the counting of plants in rice crops. The solution leverages deep learning techniques, primarily the YOLO (You Only Look Once) architecture, for object detection and counting.

The development phase, captured within this repository, focused on a comprehensive approach, including:

* **Data Analysis and Spatial Analysis:** Understanding the characteristics and spatial distribution of the data.
* **Image Preprocessing:** Applying techniques like image denoising and contrast enhancement (e.g., CLAHE) to improve image quality.
* **Vegetation Indices:** Using vegetation indices to aid in the detection process.
* **Data Augmentation:** Employing libraries like Albumentations for fast and flexible image augmentation to improve model robustness.
* **Computer Vision and Deep Learning:** Application of supervised and unsupervised learning, Convolutional Neural Network (CNN) architectures, and transfer learning.
* **Hyperparameter Optimization:** Fine-tuning the model for optimal performance.

## Folder Structure

This repository is organized to keep all development experiments, scripts, and notebooks separate from the final production-ready code.

```
ML-VisionRice-Dev/
├── Dataset processing/
│   ├── apply_blendedPCA_dataset.py
│   ├── apply_clahe_dataset.py
│   ├── apply_exGreen_dataset.py
│   ├── dataset_histogram.py
│   ├── DatasetSplitter.ipynb
│   ├── fake_dataset_builder.py
│   └── merge_datasets.py
├── Experiments/
│   ├── 0 Results_Mixed_1_(exp_48).ipynb
│   └── ...
│   ├── Inference/
│   ├── YOLOv_inference_L_(exp_19-22).ipynb
│   └── ...
│   ├── Training/
│   ├── 0 Mixed_experiments_1_(exp_48).ipynb
│   └── ...
├── Geotags/
│   ├── gotagged_analysis.ipynb
│   ├── geotagged_merged_data.csv
│   ├── geotagged_images_0.5km.csv
│   └── ...
├── Image Preprocessing/
│   ├── apply_blendedPCA_dataset.py
│   ├── Batch_rename.ipynb
│   ├── BBScaler.ipynb
│   ├── BurnBlend.py
│   ├── color_variance.py
│   ├── colorPCA_BurnBlend.py
│   ├── colorPCA.py
│   ├── DataAugmentation.ipynb
│   ├── exGreen.py
│   └── ImageSplitter.ipynb
├── Line detection/
│   ├── cropLineDetection.ipynb
│   ├── dfLineDetector manual.ipynb
│   └── dfLineDetector.ipynb
├── RetinaNet/
│   └── Annotations_Format_Convert.ipynb
├── Tagging/
│   ├── BBintoPoints.ipynb
│   ├── BBVerifier.ipynb
│   ├── images_splitter.py
│   ├── label_points.jpg
│   ├── points_visualization.jpg
│   ├── TagEditor.ipynb
│   └── tiles_splitter.py
├── Vegetation index/
├── YOLO/
├── circleROI_analysis.ipynb
├── metadata.py
├── n_samples.ipynb
└── README.md
````

-   Each main folder contains scripts, notebooks, or data related to a specific processing step.
-   `Experiments/` holds notebooks with results and experiment logs for various model runs.
-   `Dataset processing/` and `Image Preprocessing/` contain scripts for preparing, cleaning, and augmenting datasets.
-   `Tagging/`, `Line detection/`, `RetinaNet/`, and `YOLO/` are dedicated to annotation, detection, and model-specific tasks, including different architectures explored during development.
-   CSV and notebook files at the root are used for initial data analysis and management.

## Getting Started
### Prerequisites

The project requires **Python 3.x** and the following libraries, which can be installed via `pip` or `conda`:

* `albumentations`
* `opencv-python` and `opencv-python-headless` (for OpenCV functionalities)
* `torch`, `torchaudio`, and `torchvision` (for PyTorch)
* `ultralytics` and `ultralytics-thop` (for YOLOv8)
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `scikit-image`
* `scipy`
* `Pillow` (PIL)
* `tqdm`
* `PyYAML`
* `requests`
* `seaborn`
* `plotly`
* `geopy`
* `ExifRead`
* `jupyter_client` and `ipykernel` (for Jupyter notebooks)

### Installation
You can set up the environment in two ways:
- using `pip` with `requirements.txt`
- or using `conda` with `environment.yml`.

#### Option 1: Using pip

```bash
# Clone this repository
git clone [https://github.com/MLsound/ML-VisionRice-Dev.git](https://github.com/MLsound/ML-VisionRice-Dev.git)
cd ML-VisionRice-Dev
# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Conda

If you prefer to use Conda, you can create and activate an environment using the provided `environment.yml` file:

```bash
# Clone this repository
git clone [https://github.com/MLsound/ML-VisionRice-Dev.git](https://github.com/MLsound/ML-VisionRice-Dev.git)
cd ML-VisionRice-Dev
# Create and activate the conda environment
conda env create -f environment.yml
conda activate tenv
```
---

## Main Repository (Final User Tool)

For the final, production-ready code and user tool, please refer to the main repository:

  * **[ML-VisionRice on GitHub](https://github.com/MLsound/ML-VisionRice)**

