# GazeBST_Utrain

## Requirements
Package | Version
| :--- | :---
python | >=3.10.8
opencv | >=4.6.0
pytorch | >=1.13.0
torchvision | >=0.14.0
pillow | >=9.2.0
matplotlib | >=3.6.2
ipython | >=8.7.0
ipykernel | >=6.19.2
numpy | >=1.23.5

:exclamation: environment.yml & requirements.txt are for MacOS Vetura Version 13.0.1

## Usage

### Video Inference
```
python <Utrain_Baseline_Eye/Utrain_Baseline_Fullface>/video_test.py
```
:exclamation: May need to change device index in cv2.VideoCapture() if more than one cameras is connected.

```
cap = cv2.VideoCapture(0)
```

### Image Inference 
python <Utrain_Baseline_Eye/Utrain_Baseline_Fullface>/image_test.py

## Performance

#### Train and test on Columbia Gaze Dataset
| Model | Accuracy, mean angle error (deg) | Resolution | Params |
| ----- | ----------------------------| ---------- | ------ |
| Utrain (eye) | 4.26 | 36 x 60 | 3.08 M |
| Utrain (eye) advers | - | 36 x 60 | 3.08 M |
| Utrain (cropped face) | 4.99 | 80 x 112 | 3.10 M |
| Utrain (cropped face) + advers | 4.15 | 80 x 112 | 3.10 M |
| Utrain (full face) | - | 224 x 224 | 3.19 M |
| Utrain (full face) + advers | 3.86 | 224 x 224 | 3.19 M |