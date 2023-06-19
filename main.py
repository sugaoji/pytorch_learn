import d2l
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
print(torch.cuda.is_available())


class MyData(Dataset):
    def __init__(self, dir, label_dir):
        self.dir = dir
        self.label_dir = label_dir
        self.path = os.path.join(dir,label_dir)
        self.image_list = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.path,image_name)
        image = Image.open(image_path)
        label = self.label_dir
        result = (image, label)
        return result

    def __len__(self):
        return len(self.image_list)

dir = "C:\\Users\\sugao\\Desktop\\研究生相关\\pytorch学习\\dataset\\train"
label_dir = "ants"
mydata_ants = MyData(dir,label_dir)
img,label = mydata_ants[1]
print(img)
img.show()



