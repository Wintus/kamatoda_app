import json
from skimage import io
from scipy.misc import imresize
from net import Net
import torch
import torch.nn.functional as F

from flask import Flask
app = Flask(__name__)

MODEL_PATH = '../ml_src/model/yakimono_proto.pt'
IMG_NAME = '../ml_src/data/arita/0001.jpg'
LABS = ["arita", "mino", "seto"]


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/yakimono")
def yakimono():
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.eval()

    image = io.imread(IMG_NAME)
    image = imresize(image, (128, 128), interp='nearest')
    image = image/255
    _input = torch.Tensor(image).reshape((1, 3, 128, 128))

    output = model(_input)
    dic = {}
    for pred, lab in zip(F.softmax(output, dim=1)[0], LABS):
        dic[lab] = float(pred)
    return json.dumps(dic)


if __name__ == '__main__':
    app.run()
