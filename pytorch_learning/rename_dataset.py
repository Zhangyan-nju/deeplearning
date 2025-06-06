import os

root_dir = "dataset/train"
target_dir = "bees_image"
img_path = os.path.join(root_dir, target_dir)
label = target_dir.split('_')[0]
out_dir = "bees_label"

len = 0
# 获取所有图片文件
for filename in os.listdir(img_path):
    if filename.endswith('.jpg'):
        len += 1
        file_name = filename.split('.jpg')[0]
        # 创建标签文件
        with open(os.path.join(root_dir, out_dir, f'{file_name}.txt'), 'w') as f:
            f.write(label)
print(len)