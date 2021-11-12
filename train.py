from datasets import *
import torch
from Net import Net
import torch.optim as optim
import torch.nn as nn
import os

def Train(model, trainloader, device):
    for epoch in range(epoches):
        running_step = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_step += 1

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / running_step))

        if epoch % 10 == 9:
            torch.save(model, os.path.join(modelDir, model_name))

    torch.save(model, os.path.join(modelDir, model_name))
    print('Finished Training')


if __name__ == '__main__':

    print(device)

    # model = Net()
    # model.to(device)
    model = torch.load(os.path.join(modelDir, model_name))
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    Train(model, trainloader, device)
