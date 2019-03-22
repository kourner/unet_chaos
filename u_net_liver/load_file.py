import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import pylab
import os
import torch
from torchvision.transforms import transforms

def make_dataset(root):

    # root = ...../train
    #
    # imgs=[]
    # n=len(os.listdir(root))//2
    # for i in range(n):
    #     img=os.path.join(root,"%03d.png"%i)
    #     mask=os.path.join(root,"%03d_mask.png"%i)
    #     imgs.append((img,mask))
    # return imgs

    # 获取所有病人文件

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


def getitem(index):
    imgs = make_dataset(root)
    x_path, y_path =imgs[index]

    img_y = Image.open(y_path)
    img_y = transforms.ToTensor()(img_y)
    print(img_y)
    if self.transform is not None:
        img_x = self.transform(img_x)
    if self.target_transform is not None:
        img_y = self.target_transform(img_y)
    return img_x, img_y


if __name__=='__main__':
    root = 'D:\long-term\dataset/new/train\CT/'
    imgs = make_dataset(root)
    print(len(imgs))