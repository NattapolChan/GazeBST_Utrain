import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from UnsupervisedGaze_model import *
import math
import time

openCV_PATH = '/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Eye/OpenCV_Localisation_Model/'
eye_cascade_left = cv2.CascadeClassifier(openCV_PATH+'haarcascade_lefteye_2splits.xml')
eye_cascade_right = cv2.CascadeClassifier(openCV_PATH+'haarcascade_righteye_2splits.xml')
face_cascade = cv2.CascadeClassifier(openCV_PATH+'haarcascade_face.xml')

pretrained_PATH = '/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Eye/Pretrained_Model/'
first_read = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
model = GazeRepresentationLearning()
model.load_state_dict(torch.load('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Eye/Pretrained_Model/baseline_error_eye=3-2198.pth', map_location=torch.device('cpu')))
model.eval()

def crop_center(img,cropy,cropx):
    _, y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:, starty+10:starty+cropy+10,startx:startx+cropx]

def find_abs_angle_difference(a, b):
    cos_theta = torch.cos(a/180 * math.pi) * torch.cos(b/180 * math.pi) 
    theta = torch.acos(cos_theta)
    return torch.abs(theta * 180 / math.pi)

cur_time = 0
prev_time = 0
fps = 0

plot_each_frame = True
fig, ax = plt.subplots()
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(36,60))
])


while(ret):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray,5,1,1)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    cur_time = time.time()
    if cur_time-prev_time > 1:
        prev_time = time.time()
        print(f'current FPS = {fps}')
        fps = 0
    fps += 1

    if len(faces) > 0:
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face_gray = img_gray[y:y+h,x:x+w]
            face = img[y:y+h,x:x+w]
            eyes_left = eye_cascade_left.detectMultiScale(face_gray)
            if len(eyes_left) > 0:
                rgbPermutation = [2,1,0]
                cv2.rectangle(img, (x+eyes_left[0][0],y+eyes_left[0][1]), (x+eyes_left[0][0]+eyes_left[0][2], y+eyes_left[0][1] + eyes_left[0][3]), (0,255,0), 2)
                eyes_input = face[eyes_left[0][1]: eyes_left[0][1] + eyes_left[0][3], eyes_left[0][0]: eyes_left[0][0] + eyes_left[0][2], :]
                eyes_input = np.transpose(eyes_input, (2, 0, 1))
                eyes_input_torch = torch.from_numpy(eyes_input) / 255
                eyes_input_torch = eyes_input_torch.to(torch.float32)
                eyes_input_torch = preprocess(eyes_input_torch)
                eyes_input_torch = eyes_input_torch.view(1, 3, 36, 60)
                with torch.no_grad():
                    eyes_input_torch = eyes_input_torch[:, rgbPermutation, :, :]
                    output = model(eyes_input_torch)
                    output = np.array(output)
                    size = 100
                    if plot_each_frame:
                        eyes_input_torch = eyes_input_torch.view(1, 3, 36, 60)
                        a = eyes_input_torch
                        a = torch.permute(a, (0,2,3,1)) * 0.5 + 0.5
                        plt.imshow(a[0])
                        yaw = str(float(output[0][0]))[:5]
                        pitch = str(float(output[0][1]))[:5]
                        plt.title(f'Vertical = {yaw} Horizontal = {pitch}')
                        plt.show()
    cv2.imshow("image", img)
    a = cv2.waitKey(1)
    if a & 0xFF == ord('q'):
        break
    elif a==ord('s') and first_read:
        first_read = False
cap.release()
cv2.destroyAllWindows()