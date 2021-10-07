import torchvision.transforms as transforms
import torch

train_transform = transforms.Compose([
    # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
    transforms.RandomCrop(32, padding=4),

    # 按0.5的概率水平翻转图片
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

classes = ('plane',
           'car',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck'
           )

dataDir = './data/'
modelDir = '.\\model'

stepSize = 1000

epoches = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
