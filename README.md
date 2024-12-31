<p align="center" style="font-weight: bold;">
  <h1>🍥⛩️ JutsuClassifier ⛩️🍥</h1>
</p>

This repository contains a project for training, validating, and classifying Naruto's hand signs using two models:

  * **Faster R-CNN**: A deep learning model specialized in object detection.
  * **FastAI**: A framework for image classification using transfer learning.

Both models are designed to classify Naruto hand signs based on the provided datasets. This project includes pipelines for training, validating, and real-time prediction, and allows seamless switching between the two models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f700031-b759-49f8-ae6f-4aac19dc1f0f" width="500"/>
</p>

---

## 🍜 Differences Between Faster R-CNN and FastAI Models

### Faster R-CNN:
- **Focus**: Object detection.
- **Capabilities**: Detects and localizes hand signs in an image, providing bounding boxes and confidence scores.
- **Dataset**: Requires annotations in COCO format for bounding boxes and labels.

### FastAI:
- **Focus**: Image classification.
- **Capabilities**: Predicts the class of a hand sign from an input image without localization.
- **Dataset**: Organized folder structure with subfolders named after class labels.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
#----- Clone this repository to your local machine
git clone https://github.com/d1akon/JutsuClassifier.git
cd JutsuClassifier
```

### 2. Set Up Python Environment
Ensure Python 3.8 or later is installed.

```bash
#----- Create and activate a virtual environment
python -m venv mlops-env
source mlops-env/bin/activate  # For Linux/macOS
mlops-env\Scripts\activate   # For Windows

#----- Install required dependencies
pip install -r requirements.txt
```

### 3. Configure `config.yaml`
Edit the `config/config.yaml` file to specify paths and model parameters:

- **Model Selection**:
  - Set `model.type` to either `faster_rcnn` or `fastai`.
- **Dataset Paths**:
  - Update placeholders like `FASTER_RCNN_TRAINING_DATASET_PATH` to your dataset paths.

Example:
```yaml
model:
  type: "faster_rcnn"  # Options: "faster_rcnn", "fastai"

paths:
  train_data: "FASTER_RCNN_TRAINING_DATASET_PATH"
  train_annotations: "FASTER_RCNN_TRAINING_ANNOTATIONS_PATH"
  val_data: "FASTER_RCNN_VALIDATION_DATASET_PATH"
  val_annotations: "FASTER_RCNN_VALIDATION_ANNOTATIONS_PATH"
  frcnn_model_save_path: "FASTER_RCNN_MODEL_SAVE_PATH"
  frcnn_model_load_path: "FASTER_RCNN_MODEL_LOAD_PATH"
  fastai_data: "FASTAI_TRAINING_DATASET_PATH"
  fastai_model_save_path: "FASTAI_MODEL_SAVE_PATH"
  fastai_model_load_path: "FASTAI_MODEL_LOAD_PATH"

rcnn_training:
  batch_size: 4
  num_epochs: 5
  learning_rate: 0.005

fastai_training:
  learning_rate: 3e-3
  epochs: 10
```

---

## ♨️ Running the Pipelines

### 1. Train Pipeline
Train the selected model based on the configuration in `config.yaml`.

```bash
python src/pipelines/train_pipeline.py
```

### 2. Validate Pipeline
Validate the trained model and display performance metrics.

```bash
python src/pipelines/validate_pipeline.py
```

### 3. Predict Pipeline
Run real-time predictions using a webcam or video feed.

```bash
python src/pipelines/predict_pipeline.py
```

---

## 📕 Notes

1. **Model Switching**:
   - Change the `model.type` in `config.yaml` to switch between `faster_rcnn` and `fastai`.

2. **Dataset Formats**:
   - **Faster R-CNN**: Requires datasets in COCO format with annotations.
   - **FastAI**: Requires a folder structure with images organized into subfolders by class name.

3. **Dependencies**:
   - Ensure all dependencies in `requirements.txt` are installed.
   - For GPU acceleration, install the appropriate CUDA and cuDNN versions compatible with your PyTorch installation.

4. **Visualization**:
   - Utilize visualization scripts in the `utils` folder to verify dataset integrity (e.g., bounding boxes).

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

