from torchvision import transforms
from PIL import Image                   
import cv2
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_cv2 = cv2.imread(img_path)

# ToTensor
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img_PIL)
img_tensor = trans_to_tensor(img_cv2)

writer = SummaryWriter("logs")
writer.add_image("cv2", img_cv2, dataformats="HWC")
writer.add_image("ToTensor", img_tensor)

# Normalize
# ImageNet 的均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
trans_norm = transforms.Normalize(mean, std)
img_norm = trans_norm(img_tensor)

writer.add_image("Normalize", img_norm) 

print(img_PIL.size)
# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_PIL)
print(img_resize.size)
# 将 PIL Image 转换为 tensor
img_resize_tensor = trans_to_tensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 0)

# Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_to_tensor])
img_compose = trans_compose(img_PIL)
print(img_compose.shape)
writer.add_image("Resize", img_compose, 1) 

# RandomCrop
trans_randomcrop = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_to_tensor])
for i in range(10):
    img_randomcrop = trans_compose_2(img_PIL)
    writer.add_image("RandomCrop", img_randomcrop, i)

writer.close()

