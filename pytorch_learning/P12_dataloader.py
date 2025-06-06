import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备的测试集
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())          

# 加载数据集
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
img, target = test_set[0]
# print(img.shape)
print(target)


for epoch in range(2):
    writer = SummaryWriter("logs")
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets[0])
        writer.add_images("epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()
