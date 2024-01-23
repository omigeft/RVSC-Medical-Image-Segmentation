# 数据文件目录结构：
# ../TrainingSet
# ../TestSet/
#     /Test1Set
#     /Test2Set
#     /Test1SetContours
#     /Test2SetContours

import os
import shutil
import pydicom
from PIL import Image
import numpy as np
import cv2
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    GaussNoise, ElasticTransform, RandomResizedCrop
)
from tqdm import tqdm
from pathlib import Path


def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_dcm_to_png(dicom_path, png_path):
    """将 DICOM 文件转换为 PNG 图像"""
    dcm_image = pydicom.dcmread(dicom_path).pixel_array
    im = Image.fromarray(dcm_image)
    im = im.convert('L')  # 转换为灰度图像
    im.save(png_path)


def create_mask_from_contour(contour_path, mask_path, image_shape):
    """根据轮廓文件创建 mask 图像"""
    # 读取轮廓点并转换为 OpenCV 可以理解的格式
    contours = []
    with open(contour_path, 'r') as file:
        contour = []
        for line in file:
            x, y = map(lambda v: int(round(float(v))), line.split())
            contour.append([x, y])
        contours.append(np.array(contour, dtype=np.int32))

    # 创建一个空白的 mask，并用轮廓点填充区域
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, 255)  # 填充轮廓内部

    Image.fromarray(mask).save(mask_path)


def copy_and_process_files(base_dir, dataset_folder, target_folder, start_index, end_index):
    # 创建目标目录
    target_dir_imgs = os.path.join(base_dir, target_folder, 'imgs')
    target_dir_i_masks = os.path.join(base_dir, target_folder, 'i-masks')  # 内轮廓 mask
    target_dir_o_masks = os.path.join(base_dir, target_folder, 'o-masks')  # 外轮廓 mask
    ensure_directory_exists(target_dir_imgs)
    ensure_directory_exists(target_dir_i_masks)
    ensure_directory_exists(target_dir_o_masks)

    # 遍历dataset_folder文件夹
    for i in range(start_index, end_index + 1):
        patient_folder = f'patient{i:02d}'
        list_file_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}list.txt')

        if not os.path.exists(list_file_path):
            print(f"List file not found: {list_file_path}")
            continue

        # 读取并处理list文件中的每一行
        with open(list_file_path, 'r') as file:
            for line in file:
                line = line.strip().replace('.\\', '').replace('\\', '/')
                contour_filename = line.split('/')[-1]

                # 处理 DICOM 文件并保存为 PNG
                dicom_filename = contour_filename.replace('-icontour-manual.txt', '.dcm').replace('-ocontour-manual.txt', '.dcm')
                dicom_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}dicom', dicom_filename)
                png_filename = dicom_filename.replace('.dcm', '.png')
                png_path = os.path.join(target_dir_imgs, png_filename)
                if os.path.exists(dicom_path):
                    convert_dcm_to_png(dicom_path, png_path)
                    print(f"Converted and saved: {png_path}")
                else:
                    print(f"Dicom file not found: {dicom_path}")
                    continue  # 如果 DICOM 文件不存在，跳过后续处理

                # 根据是 i-contour 还是 o-contour 创建并保存 mask
                mask_path = os.path.join(target_dir_i_masks if 'icontour' in contour_filename else target_dir_o_masks, png_filename)
                contour_path = os.path.join(base_dir, dataset_folder, line)
                if os.path.exists(contour_path) and os.path.exists(png_path):
                    image_shape = Image.open(png_path).size[::-1]  # 获取图像尺寸 (高度, 宽度)
                    create_mask_from_contour(contour_path, mask_path, image_shape)
                    print(f"Created and saved mask: {mask_path}")
                else:
                    print(f"Contour file or PNG file not found: {contour_path} or {png_path}")


def copy_contours(base_dir, dataset_folder, source_folder, target_folder, start_index, end_index):
    source_dir = os.path.join(base_dir, dataset_folder, source_folder)
    target_base_dir = os.path.join(base_dir, dataset_folder, target_folder)

    # 确保源和目标基础目录存在
    if not os.path.exists(source_dir) or not os.path.exists(target_base_dir):
        print("Source or target base directory does not exist.")
        return

    # 遍历指定范围内的文件夹
    for i in range(start_index, end_index + 1):
        source_folder = os.path.join(source_dir, f"P{i:02d}contours-manual")
        target_folder = os.path.join(target_base_dir, f"patient{i:02d}")

        # 检查源文件夹是否存在
        if not os.path.exists(source_folder):
            print(f"Source folder not found: {source_folder}")
            continue

        # 创建目标文件夹（如果不存在）
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 复制文件夹
        shutil.copytree(source_folder, os.path.join(target_folder, f"P{i:02d}contours-manual"), dirs_exist_ok=True)
        print(f"Copied {source_folder} to {target_folder}")


def augment_data(img_path, mask_i_path, mask_o_path, output_dir, image_name, transform):
    # 读取图像和掩码
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask_i = cv2.imread(str(mask_i_path), cv2.IMREAD_GRAYSCALE)
    mask_o = cv2.imread(str(mask_o_path), cv2.IMREAD_GRAYSCALE)

    # 保存原始图像和掩码
    cv2.imwrite(str(output_dir / 'imgs' / image_name), image)
    cv2.imwrite(str(output_dir / 'i-masks' / image_name), mask_i)
    cv2.imwrite(str(output_dir / 'o-masks' / image_name), mask_o)

    for i in range(10):
        # 应用增强
        augmented = transform(image=image, masks=[mask_i, mask_o])
        image_aug, mask_i_aug, mask_o_aug = augmented['image'], augmented['masks'][0], augmented['masks'][1]

        # 保存增强后的图像和掩码
        cv2.imwrite(str(output_dir / 'imgs' / f'aug{i}_{image_name}'), image_aug)
        cv2.imwrite(str(output_dir / 'i-masks' / f'aug{i}_{image_name}'), mask_i_aug)
        cv2.imwrite(str(output_dir / 'o-masks' / f'aug{i}_{image_name}'), mask_o_aug)


def augmentation(img_dir, mask_i_dir, mask_o_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'imgs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'i-masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'o-masks').mkdir(parents=True, exist_ok=True)

    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=45, p=0.5),
        RandomBrightnessContrast(p=0.3),
        GaussNoise(p=0.2),
        ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        RandomResizedCrop(height=256, width=216, scale=(0.3, 1.0), p=0.5)
    ])

    for img_name in tqdm(os.listdir(img_dir), desc='Augmenting images'):
        if img_name.endswith('.png'):
            img_path = img_dir / img_name
            mask_i_path = mask_i_dir / img_name
            mask_o_path = mask_o_dir / img_name

            augment_data(img_path, mask_i_path, mask_o_path, output_dir, img_name, transform)


if __name__ == '__main__':
    # 调用函数
    base_dir = '..'  # 当前目录作为基本目录
    copy_and_process_files(base_dir, 'TrainingSet', 'train_data', 1, 16)
    copy_contours(base_dir, 'TestSet', "Test1SetContours", "Test1Set", 17, 32)
    copy_contours(base_dir, 'TestSet', "Test2SetContours", "Test2Set", 33, 48)
    copy_and_process_files(base_dir, 'TestSet/Test1Set', 'test1_data', 17, 32)
    copy_and_process_files(base_dir, 'TestSet/Test2Set', 'test2_data', 33, 48)

    print("数据文件处理和转换完成。")

    augmentation(Path('../train_data/imgs'), Path('../train_data/i-masks'), Path('../train_data/o-masks'),
                 Path('../train_data_aug'))

    print("数据增广完成")