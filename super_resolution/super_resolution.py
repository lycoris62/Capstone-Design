import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import glob

CUR_DIR = os.path.abspath('.')


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(64, 32, 1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


model = SRCNN()
model.load_state_dict(torch.load(os.getcwd() + "/super_resolution/srcnn_model.pt", map_location="cpu"))
model.eval()


def super_res(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = model(img)
    img = img.squeeze(0)
    img = img.detach().numpy()
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def super_resolution(path):
    object_list = glob.glob(f"{path}/objects_original/*")

    if len(object_list) == 0:
        return
    for object_path in object_list:
        img = cv2.imread(object_path)
        img = super_res(img)
        path_name = object_path.replace("objects_original", "objects_super")
        cv2.imwrite(path_name, img)

