from config import *
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def DataInit(dataDir):
    trainset = CIFAR10(root=dataDir, train=True, download=False, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = CIFAR10(root=dataDir, train=False, download=False, transform=test_transform)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    return trainloader, testloader


# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def ImgShow(Loader):
    # 获取随机数据
    dataiter = iter(Loader)
    images, labels = dataiter.next()

    # 展示图像
    imshow(torchvision.utils.make_grid(images))
    # 显示图像标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
