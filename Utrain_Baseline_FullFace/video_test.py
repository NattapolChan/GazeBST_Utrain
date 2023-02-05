import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from UnsupervisedGaze_model import *
import math
import time

eye_cascade_left = cv2.CascadeClassifier('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Fullface/haarcascade_lefteye_2splits.xml')
eye_cascade_right = cv2.CascadeClassifier('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Fullface/haarcascade_righteye_2splits.xml')
face_cascade = cv2.CascadeClassifier('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Fullface/haarcascade_face.xml')

first_read = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
model = GazeRepresentationLearning_fullface()
model = torch.load('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_FullFace/ADV_UnsupervisedBaseline_log_6/full_log/baseline_epoch=56_loss=4.044334411621094_batch_size=16_weight_decay=0.0001.pt')
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

plot_each_frame = False
fig, ax = plt.subplots()

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
                cv2.rectangle(img, (x+eyes_left[0][0],y+eyes_left[0][1]), (x+eyes_left[0][0]+eyes_left[0][2], y+eyes_left[0][1] + eyes_left[0][3]), (0,255,0), 2)
                eyes_input = face[eyes_left[0][1]: eyes_left[0][1] + eyes_left[0][3], eyes_left[0][0]: eyes_left[0][0] + eyes_left[0][2], :]
                eyes_input = np.transpose(eyes_input, (2, 0, 1))
                eyes_input = crop_center(eyes_input, 224, 224)
                eyes_input = (eyes_input - eyes_input.mean(axis=(0,1,2), keepdims=True)) / eyes_input.std(axis=(0,1,2), keepdims=True)
                if eyes_input.shape != (3,224,224) :
                    continue
                eyes_input = np.resize(eyes_input, (1, 3, 224, 224))
                eyes_input_torch = torch.from_numpy(eyes_input)
                eyes_input_torch = eyes_input_torch.to(torch.float32)
                with torch.no_grad():
                    output = model(eyes_input_torch)
                    output = np.array(output)
                    size = 100
                    point_from = [eyes_input.shape[3]//2, eyes_input.shape[2]//2]
                    point_to = [eyes_input.shape[3]//2 + size * math.sin(math.pi/180 * output[0][0]) * math.cos(math.pi/180 * output[0][1]), eyes_input.shape[2]//2 + size * math.sin(math.pi/180 * output[0][1])]
                    if plot_each_frame:
                        ax.imshow(eyes_input[0,0,:,:])
                        ax.arrow(point_from[0], point_from[1], point_to[0]-point_from[0], point_to[1]-point_from[1], color="white", linewidth=3)
                        yaw = str(float(output[0][0]))[:5]
                        pitch = str(float(output[0][1]))[:5]
                        ax.set_title(f'{yaw=} {pitch=}')
                        plt.ion()
                        plt.show()
                    cv2.arrowedLine(img, (x + eyes_left[0][0] + int(point_from[0]), y + eyes_left[0][1] + int(point_from[1]) + 15), ( x + eyes_left[0][0]+ int(point_to[0]), y + eyes_left[0][1] + int(point_to[1]) + 15), color = (0,255,0), thickness = 2)
    cv2.imshow("image", img)
    a = cv2.waitKey(1)
    if a & 0xFF == ord('q'):
        break
    elif a==ord('s') and first_read:
        first_read = False
cap.release()
cv2.destroyAllWindows()