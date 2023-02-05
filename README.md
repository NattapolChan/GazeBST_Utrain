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
