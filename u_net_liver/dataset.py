import torch.utils.data as data
import PIL.Image as Image
import os
import pydicom
import SimpleITK as sitk
from torchvision.transforms import transforms
import numpy as np


def make_dataset(root):

    # imgs=[]
    # GT_paths = root + '_GT/'
    # image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
    # for img in image_paths:
    #     mask = GT_paths + 'ISIC_' + img[-9:-4] + '.png'
    #     imgs.append((img, mask))
    # return imgs

    # 获取所有病人

    imgs = []
    patients = os.listdir(root)
    for i, patient in enumerate(patients):
        path = root + patient + '/' + 'train'
        GT_paths = path + '_GT/'
        image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        for img in image_paths:
            mask = GT_paths + 'liver_GT_' + img[-9:-4] + '.png'
            img_test = Image.open(mask)
            img_test = np.array(img_test)
            if 1 in img_test:
                imgs.append((img, mask))
    return imgs

class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
