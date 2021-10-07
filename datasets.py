from config import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


trainset = CIFAR10(root=dataDir, train=True, download=False, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10(root=dataDir, train=False, download=False, transform=test_transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
