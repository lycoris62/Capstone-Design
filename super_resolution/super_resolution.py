import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2

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

def super_res(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float()
    frame = frame.unsqueeze(0)
    frame = model(frame)
    frame = frame.squeeze(0)
    frame = frame.detach().numpy()
    frame = frame.transpose(1, 2, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame
