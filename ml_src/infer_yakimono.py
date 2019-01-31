import json
from skimage import io
from scipy.misc import imresize

import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = "model/yakimono_proto.pt"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*29*29, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 29*29*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model = model.eval()

    img_name = 'data/arita/0001.jpg'
    image = io.imread(img_name)
    image = imresize(image, (128, 128), interp='nearest')
    image = image/255

    _input = torch.Tensor(image).reshape((1, 3, 128, 128))

    output = model(_input)
    labs = ["arita", "mino", "seto"]
    dic = {}
    for pred, lab in zip(F.softmax(output, dim=1)[0], labs):
        dic[lab] = float(pred)
    print(json.dumps(dic))
