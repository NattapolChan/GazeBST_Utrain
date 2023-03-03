import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from UnsupervisedGaze_model import * 
import os

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(36, 60)),
])

model = GazeRepresentationLearning()
model.load_state_dict(torch.load('/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Eye/Pretrained_Model/baseline_error_eye_adv=4-2264.pth', map_location=torch.device('cpu')))
model.eval()

path = '/Users/nattapolchanpaisit/GazeBST/Utrain_Baseline_Eye/images'
dirs = os.scandir(path)
images = []
for each in dirs:
    if each.is_file() and each.path[-5:] == '.jpeg':
        with Image.open(each.path) as image:
            images.append(preprocess(image))
imagesTensor = torch.stack(images)

with torch.no_grad():
    outputs = model(imagesTensor)

fig, axes = plt.subplots(3,5,figsize=(20, 7))
count = 0
imagesTensor = imagesTensor.permute(0,2,3,1)
imagesTensor = imagesTensor * 0.5 + 0.5
for i in range(imagesTensor.size(0)):
    axes[count//5, count%5].axis('off')
    axes[count//5, count%5].set_title(f' V : {str(float(outputs[count,0]))[:5]}  H : {str(float(outputs[count,1]))[:5]}')
    im3 = axes[count //5, count%5].imshow(imagesTensor[count, :, :, :], aspect=1)
    count += 1
plt.tight_layout()
fig.suptitle('Sample Image test : Utrain_EYE_ADV on Sample Image (trained on Columbia Gaze)  Res = (3,36,60)')
plt.subplots_adjust(top=0.92)
plt.show()