import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from skimage import io
import pickle
from tqdm import tqdm

LABEL_IDX = 1
IMG_IDX = 2
root = 'data'
lab_dic = {
    '有田焼': 0,
    '瀬戸焼': 1,
    '美濃焼': 2,
}


class MyDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, img_name):
        image = io.imread(img_name)
        return image


path_lst = []
lab_lst = []
for yakimono in ['有田焼', '美濃焼', '瀬戸焼']:
    picts = ['{0}/{1}/{2}'.format(root, yakimono, i)
             for i in os.listdir('{0}/{1}'.format(root, yakimono))]
    path_lst.extend(picts)
    lab_lst.extend([lab_dic[yakimono] for _ in range(len(picts))])

imgDataset = MyDataset()
img_lst = []
for path in tqdm(path_lst):
    img_lst.append(imgDataset[path])

x_train, x_test, y_train, y_test = train_test_split(
    img_lst, lab_lst, test_size=0.2, random_state=0
)

with open('data/pkl/dataset', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
