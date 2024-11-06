import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil

# 配置路径
MODEL_PATH = './model/150_model.keras'  # 训练好的模型路径
CLASS_INDICES_PATH = './model/150_class_indices.json'  # 类别名称映射文件路径
IMAGE_FOLDER = './未分类'  # 未分类图片路径
OUTPUT_FOLDER = './分类结果'  # 分类结果路径
IMG_SIZE = (150, 150)  # 与训练时保持一致

# 加载模型
model = load_model(MODEL_PATH)

# 加载类别索引文件
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# 反转类别索引映射，以便通过预测索引找到类别名称
class_labels = {v: k for k, v in class_indices.items()}

# 初始化分类计数字典
class_counts = {label: 0 for label in class_labels.values()}

# 图像预处理函数
def preprocess_image(image_path):
    # 加载图像并调整大小
    img = load_img(image_path, target_size=IMG_SIZE)
    # 转换图像为数组，并增加一个维度以匹配模型输入格式
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # 归一化图像数组
    img_array /= 255.0
    return img_array

# 对图片进行分类
def classify_image(image_path):
    # 预处理图像
    img_array = preprocess_image(image_path)
    # 预测类别
    predictions = model.predict(img_array)
    # 获取概率最高的类别索引
    predicted_class_index = np.argmax(predictions[0])
    # 获取对应的类别标签
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# 创建输出文件夹
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 遍历未分类图片文件夹并进行分类
for filename in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    
    # 确保处理的是图片文件
    if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        # 分类图片
        predicted_class = classify_image(image_path)
        
        # 更新类别计数
        class_counts[predicted_class] += 1
        
        # 为每个类别创建对应的文件夹
        class_folder = os.path.join(OUTPUT_FOLDER, predicted_class)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # 将图片移动到对应类别的文件夹
        shutil.move(image_path, os.path.join(class_folder, filename))
        print(f"{filename} 已分类为 {predicted_class}")

# 打印分类统计结果
print("\n分类统计结果:")
for category, count in class_counts.items():
    print(f"类别 {category}: {count} 张图片")

print("图片分类完成。")
