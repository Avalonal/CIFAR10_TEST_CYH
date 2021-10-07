from main import *
import torch
from Net import Net
import torch.optim as optim
import torch.nn as nn
import os

def Train(trainloader, device):
    for epoch in range(epoches):  # 多批次循环

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % stepSize == stepSize-1:  # 每3批次打印一次， 在 gpu 上训练请调大此参数
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / stepSize))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net, os.path.join(modelDir, 'model_00.pkl'))


if __name__ == '__main__':
    trainLoader, testLoader = DataInit(dataDir)

    print(device)

    #net = Net()
    #net.to(device)
    net = torch.load(os.path.join(modelDir, 'model_00.pkl'))
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    Train(trainLoader, device)
