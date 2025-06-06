from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from PIL import Image

writer = SummaryWriter("logs")

# 添加标量数据
for i in range(100):
    writer.add_scalar("y = 3*x", 3*i, i)

image_path = "dataset/train/bees_image"
image_nlist =  os.listdir(image_path)
idx = 0
for image_name in image_nlist:
    img_PIL =  Image.open(os.path.join(image_path, image_name))
    img_array = np.array(img_PIL)
    writer.add_image("test", img_array, idx, dataformats='HWC')
    idx += 1

writer.close()