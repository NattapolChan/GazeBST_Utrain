import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from UnsupervisedGaze_model import *
import math
import time
from blazeface import BlazeFace

openCV_PATH = '/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_FullFace/OpenCV_Localisation_Model/'
eye_cascade_left = cv2.CascadeClassifier(openCV_PATH+'haarcascade_lefteye_2splits.xml')
eye_cascade_right = cv2.CascadeClassifier(openCV_PATH+'haarcascade_righteye_2splits.xml')
face_cascade = cv2.CascadeClassifier(openCV_PATH+'haarcascade_face.xml')

first_read = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()

videoShape = (img.shape[0], img.shape[1])

font = cv2.FONT_HERSHEY_SIMPLEX

blazeface = BlazeFace()
blazeface.load_weights('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_FullFace/blazeface/blazeface.pth')
blazeface.load_anchors('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_FullFace/blazeface/anchors.npy')

model = GazeRepresentationLearning_fullface()
model.load_state_dict(torch.load('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_FullFace/Pretrained_Model/adv_Utrain_fullface_error=3.49.pth', map_location=torch.device('cpu')))
model.eval()

def crop_center(img,cropy,cropx):
    _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:, starty+10:starty+cropy+10,startx:startx+cropx]

def find_abs_angle_difference(a, b):
    cos_theta = torch.cos(a/180 * math.pi) * torch.cos(b/180 * math.pi) 
    theta = torch.acos(cos_theta)
    return torch.abs(theta * 180 / math.pi)

preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(224,224))
])
preprocess_face = transforms.Compose([
    transforms.CenterCrop(size=(min(videoShape),min(videoShape))),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(128,128))
])

cur_time = 0
prev_time = 0
fps = 0

plot_each_frame = False
fig, ax = plt.subplots()
histFps = 0

while(ret):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray,5,1,1)
    faceTensor = np.transpose(img, (2, 0, 1))
    faceTensor = torch.from_numpy(faceTensor) / 255
    faceTensor = preprocess_face(faceTensor)
    faceTensor = faceTensor.view(1, 3, 128, 128)
    blazefaceOutput = blazeface.predict_on_batch(faceTensor * 127.5 + 127.5)
    print(blazefaceOutput)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    cur_time = time.time()
    if cur_time-prev_time > 1:
        prev_time = time.time()
        histFps = fps
        fps = 0
    fps += 1
    timeEach = {}
    now = time.time()
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            prev = time.time()
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face_gray = img_gray[y:y+h,x:x+w]
            face = img[y:y+h,x:x+w]
            eyes_input = face
            eyes_input = np.transpose(eyes_input, (2, 0, 1))
            eyes_input_torch = torch.from_numpy(eyes_input) / 255
            eyes_input_torch = eyes_input_torch.to(torch.float32)
            eyes_input_torch = preprocess(eyes_input_torch)
            eyes_input_torch = eyes_input_torch.view(1, 3, 224, 224)
            with torch.no_grad():
                rgbPermutation = [2,1,0]
                eyes_input_torch = eyes_input_torch[:, rgbPermutation, :, :]
                timeEach['preProcess'] = time.time() - prev
                prev = time.time()
                output = model(eyes_input_torch)
                timeEach['modelInference'] = time.time() - prev
                prev = time.time()
                eyes_input_torch = eyes_input_torch.view(1, 3, 224, 224)
                output = np.array(output)
                size = 100
                point_to = [-size * math.sin(math.pi/180 * output[0][0]) * math.cos(math.pi/180 * output[0][1]), size * math.sin(math.pi/180 * output[0][1])]
                yaw = str(float(output[0][0]))[:5]
                pitch = str(float(output[0][1]))[:5]
                timeEach['postProcess'] = time.time() - prev
                prev = time.time()
                if plot_each_frame:
                    a = eyes_input_torch
                    a = torch.permute(a, (0,2,3,1)) * 0.5 + 0.5
                    plt.imshow(a[0,:,:,:])
                    plt.arrow(112, 112, point_to[1], point_to[0], color="red", linewidth=3)
                    plt.title(f'Vertical : {yaw=} Horizontal : {pitch=}')
                    plt.show()
                cv2.arrowedLine(img, (x + w//2, y + h//2), ( x + w//2 + int(point_to[1]), y + h//2 + int(point_to[0])), color = (255, 0 ,0), thickness = 4, tipLength=0.5)
                cv2.putText(img, f'V : {yaw}   H : {pitch}', org = (5, 50), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color = (255, 0, 0), thickness=2)
                overlaySize = []
                if face.shape[0] > 200:
                    img[70:270, 0:200] = face[h//2 - 100 : h//2 + 100, w//2 -100 : w//2 + 100]
    timeEach['total'] = time.time() - now
    count = 0
    for key in timeEach:
        cv2.putText(img, f'{key} processed in {str(1000 * timeEach[key])[:5]} ms', org = (5, 290 + count * 30), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1.5, color = (255, 0, 0), thickness=2)
        count += 1
    if histFps > 0:
        cv2.putText(img, f'FPS : {histFps} s', org = (5, 290 + count * 30), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color = (0, 255), thickness=2)
    cv2.imshow("image", img)
    a = cv2.waitKey(1)
    if a & 0xFF == ord('q'):
        break
    elif a==ord('s') and first_read:
        first_read = False
cap.release()
cv2.destroyAllWindows()