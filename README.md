# 50.035 Computer Vision Project - Group 5: Sign Language Translator

## Project Overview
This project focuses on building a computer vision model to recognize and translate static alphabet gestures from sign language into alphabets. The system prioritizes recall to ensure accurate recognition of gestures in real-time scenarios.

## Folder Structure

### 1. `models/`
Contains pre-trained models in `.h5` format, used for predicting gestures:
- **MobileNet_landmarks_only.h5**: MobileNet trained using only landmarks data.
- **doubleCNN_landmarks.h5**: A double CNN architecture trained on landmarks data.
- **multiheadcnn.h5**: A multi-head CNN model integrating various input features.
- **resent_images_only.h5**: ResNet trained solely on image data.
- **resent_landmarks_only.h5**: ResNet trained only on landmarks data.
- **resent_landmarkswithhands.h5**: ResNet trained on landmarks data combined with hand segmentation.

### 2. `scripts/`
Includes Python scripts for various stages of the project:

- **Data Collection**:
  - `Approach1.ipynb`: Jupyter Notebook for data collection using Approach 1.
  - `Approach2.py`: Script for data collection using Approach 2.

- **Model Training**:
  - `MobileNet-landmarks.ipynb`: Training script for MobileNet using landmarks data.
  - `doubleCNN_model.ipynb`: Training script for the double CNN model.
  - `multiheadcnn_chewon_edit.ipynb`: Training script for the multi-head CNN model.
  - `resnet-imageonly.ipynb`: Training script for ResNet using image data only.
  - `resnet-landmarks.ipynb`: Training script for ResNet using landmarks data only.
  - `resnet-landmarkswithhands.ipynb`: Training script for ResNet using landmarks combined with hand segmentation.

### 3. `ui/`
Contains scripts for the user interface:
- `live_dection+translation.ipynb`: Jupyter Notebook for live gesture detection and translation.
- `testing_predictions_live.py`: Script for testing predictions with live input or pre-recorded data.

## Dataset
The dataset for this project was collected by the team. For access or further information, please contact us directly.

## Evaluation Metric
The F2 score is used as the primary evaluation metric, emphasizing recall to reduce missed gestures. Model 3 (`multiheadcnn.h5`) was selected for deployment due to its highest F2 score, making it the most suitable for robust gesture recognition.

## Dependencies
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Git Setup
Before cloning the repository, ensure Git Large File Storage (LFS) is installed:
```bash
git lfs install
```
