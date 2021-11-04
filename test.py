from datasets import *
import torch
import os


def Test(testloader):
    net = torch.load(os.path.join(modelDir, model_name))

    print(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images is %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    Test(testloader)