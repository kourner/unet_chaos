import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from loss import SegmentationLosses


train_root = 'D:\long-term\dataset/new/train\CT/'
val_root = 'D:\long-term\dataset/new/val/'


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

#参数解析
parse=argparse.ArgumentParser()



def train_model(model, criterion, optimizer, dataload_train,dataload_val, num_epochs=20):
    for epoch in range(num_epochs):

        model.train(True)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload_train.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload_train:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # loss = SegmentationLosses()
            loss_dice, loss_bce, loss = criterion.dice_bce_loss(outputs,labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d   train_dice_loss:%0.3f   train_bce_loss:%0.3f" % (step, (dt_size - 1) // dataload_train.batch_size + 1, loss_dice.item(),loss_bce.item() ))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))


        #=========================================Validation==============================================
        model.train(False)
        model.eval()

        epoch_dice_loss = 0
        epoch_bce_loss = 0
        step = 0

        for x, y in dataload_val:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss_dice, loss_bce, loss = criterion.dice_bce_loss(outputs,labels)
            epoch_dice_loss += loss_dice.item()
            epoch_bce_loss += loss_bce.item()
        print("==================================================")
        print("epoch %d val_dice_loss:%0.3f   val_dice_loss:%0.3f" % (epoch, epoch_dice_loss/step, epoch_bce_loss/step))
        print("==================================================")

    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
def train():
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size

    criterion = SegmentationLosses(cuda=True)

    optimizer = optim.Adam(model.parameters())

    liver_dataset_train = LiverDataset(train_root, transform=x_transforms,target_transform=y_transforms)
    liver_dataset_val = LiverDataset(val_root, transform=x_transforms, target_transform=y_transforms)

    dataloaders_train = DataLoader(liver_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloaders_val = DataLoader(liver_dataset_val, batch_size=1, shuffle=False, num_workers=4)

    train_model(model, criterion, optimizer, dataload_train=dataloaders_train, dataload_val=dataloaders_val)



#显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp,map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    args.action = 'train'

    if args.action=="train":
        train()

    elif args.action=="test":
        test()
