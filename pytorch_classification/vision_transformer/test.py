'''
这个程序是为了直观看看数据增强中随机裁剪的效果
'''

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
#加这一段代码防止报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 读取图片
image_path = 'D:/Desktop/Snipaste_2024-05-23_18-29-07.jpg' # 替换为你的图片路径
original_image = Image.open(image_path)

# 定义transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 应用transforms
transformed_image = transform(original_image)

# 转换张量回PIL图像以便于显示
transformed_image_pil = transforms.ToPILImage()(transformed_image)

# 原始图片尺寸
original_width, original_height = original_image.size

# 根据原始图片宽高比确定子图布局
if original_width > original_height:
    # 如果图片宽度大于高度，则设置宽度比为1，高度按比例缩放
    width_ratio = 1
    height_ratio = original_height / original_width
    plt.figure(figsize=(10, 8 * height_ratio))  # 设置图像大小
else:
    # 如果图片高度大于或等于宽度，则设置高度比为1，宽度按比例缩放
    height_ratio = 1
    width_ratio = original_width / original_height
    plt.figure(figsize=(10 * width_ratio, 8))  # 设置图像大小

# 创建一个图形和两个子图
plt.subplot(2, 1, 1)  # 2行1列的第一个子图
plt.imshow(original_image)
plt.title("Original Image")
plt.axis('off')  # 不显示坐标轴

plt.subplot(2, 1, 2)  # 2行1列的第二个子图
plt.imshow(transformed_image_pil)
plt.title("Transformed Image")
plt.axis('off')  # 不显示坐标轴

# 显示图形
plt.tight_layout()  # 调整布局以适应整个窗口
plt.show()