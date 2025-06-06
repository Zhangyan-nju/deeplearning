from torch import nn
from torch.nn import Conv2d
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])

# kernel = torch.tensor([[1, 2, 1],
#                       [0, 1, 0],
#                       [2, 1, 0]])

# input = torch.reshape(input, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))

# print(input.shape)
# print(kernel.shape)

# output = F.conv2d(input, kernel, stride=1, padding=1)
# print(output)


dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)
        

class cov_demo(nn.Module):
    def __init__(self):
        super(cov_demo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        input = self.conv1(input)
        return input

writer = SummaryWriter("logs")
cov_demo = cov_demo()
step = 0
# for data in dataloader:
#     imgs, targets = data
#     output = cov_demo(imgs)
#     print(imgs.shape)
#     print(output.shape)
#     break
#     output = torch.reshape(output, (-1, 3, 32, 32))
#     writer.add_images("input", imgs, step)
#     writer.add_images("output", output, step)
#     step += 1


class maxpool_demo(nn.Module):
    def __init__(self):
        super(maxpool_demo, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3 ,padding=0, dilation=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
    
maxpool_demo = maxpool_demo()

for data in dataloader:
    imgs, targets = data
    output = maxpool_demo(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1   
