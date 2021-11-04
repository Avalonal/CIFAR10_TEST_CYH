from datasets import *
import torch
from Net import Net
import torch.optim as optim
import torch.nn as nn
import os

def Train(trainloader, device):
    for epoch in range(epoches):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % stepSize == stepSize-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / stepSize))
                running_loss = 0.0

        if epoch % 10 == 9:
            torch.save(net, os.path.join(modelDir, model_name))

    torch.save(net, os.path.join(modelDir, model_name))
    print('Finished Training')


if __name__ == '__main__':

    print(device)

    # net = Net()
    # net.to(device)
    net = torch.load(os.path.join(modelDir, model_name))
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    Train(trainloader, device)
