# Please manually organize dataset files into the structure as follows before you run the data preprocess script:
# ../TrainingSet
# ../TestSet/
#   - /Test1Set
#   - /Test2Set
#   - /Test1SetContours
#   - /Test2SetContours

import argparse
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
    # Ensure that the directory exists, if not, create the directory
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_dcm_to_png(dicom_path, png_path):
    # Convert DICOM files to PNG images
    dcm_image = pydicom.dcmread(dicom_path).pixel_array
    im = Image.fromarray(dcm_image)
    im = im.convert('L')  # 转换为灰度图像
    im.save(png_path)


def create_mask_from_contour(contour_path, mask_path, image_shape):
    # Create a mask image based on the profile file
    # Read contour points and convert them to a format that OpenCV can understand
    contours = []
    with open(contour_path, 'r') as file:
        contour = []
        for line in file:
            x, y = map(lambda v: int(round(float(v))), line.split())
            contour.append([x, y])
        contours.append(np.array(contour, dtype=np.int32))

    # Create a blank mask and fill the area with contour points
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, 255)  # Fill the interior of the contour with white

    Image.fromarray(mask).save(mask_path)


def copy_and_process_files(base_dir, dataset_folder, target_folder, start_index, end_index):
    # Create target directories
    target_dir_imgs = os.path.join(base_dir, target_folder, 'imgs')
    target_dir_i_masks = os.path.join(base_dir, target_folder, 'i-masks')  # Endocardial contour mask
    target_dir_o_masks = os.path.join(base_dir, target_folder, 'o-masks')  # Epicardial contour mask
    ensure_directory_exists(target_dir_imgs)
    ensure_directory_exists(target_dir_i_masks)
    ensure_directory_exists(target_dir_o_masks)

    # Traverse dataset_folder
    for i in range(start_index, end_index + 1):
        patient_folder = f'patient{i:02d}'
        list_file_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}list.txt')

        if not os.path.exists(list_file_path):
            print(f"List file not found: {list_file_path}")
            continue

        # Read and process each line in the list file
        with open(list_file_path, 'r') as file:
            for line in file:
                line = line.strip().replace('.\\', '').replace('\\', '/')
                contour_filename = line.split('/')[-1]

                # Process DICOM files and save them as PNG
                dicom_filename = contour_filename.replace('-icontour-manual.txt', '.dcm').replace('-ocontour-manual.txt', '.dcm')
                dicom_path = os.path.join(base_dir, dataset_folder, patient_folder, f'P{i:02d}dicom', dicom_filename)
                png_filename = dicom_filename.replace('.dcm', '.png')
                png_path = os.path.join(target_dir_imgs, png_filename)
                if os.path.exists(dicom_path):
                    convert_dcm_to_png(dicom_path, png_path)
                    print(f"Converted and saved: {png_path}")
                else:
                    print(f"Dicom file not found: {dicom_path}")
                    continue  # If the DICOM file does not exist, skip subsequent processing

                # Create and save a mask based on whether it is i-contour or o-contour
                mask_path = os.path.join(target_dir_i_masks if 'icontour' in contour_filename else target_dir_o_masks, png_filename)
                contour_path = os.path.join(base_dir, dataset_folder, line)
                if os.path.exists(contour_path) and os.path.exists(png_path):
                    image_shape = Image.open(png_path).size[::-1]  # Obtain image size (height, width)
                    create_mask_from_contour(contour_path, mask_path, image_shape)
                    print(f"Created and saved mask: {mask_path}")
                else:
                    print(f"Contour file or PNG file not found: {contour_path} or {png_path}")


def copy_contours(base_dir, dataset_folder, source_folder, target_folder, start_index, end_index):
    source_dir = os.path.join(base_dir, dataset_folder, source_folder)
    target_base_dir = os.path.join(base_dir, dataset_folder, target_folder)

    # Ensure that the source and target base directories exist
    if not os.path.exists(source_dir) or not os.path.exists(target_base_dir):
        print("Source or target base directory does not exist.")
        return

    # Traverse folders within a specified range
    for i in range(start_index, end_index + 1):
        source_folder = os.path.join(source_dir, f"P{i:02d}contours-manual")
        target_folder = os.path.join(target_base_dir, f"patient{i:02d}")

        # Check if the source folder exists
        if not os.path.exists(source_folder):
            print(f"Source folder not found: {source_folder}")
            continue

        # Create target folder if it does not exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # copy the folder
        shutil.copytree(source_folder, os.path.join(target_folder, f"P{i:02d}contours-manual"), dirs_exist_ok=True)
        print(f"Copied {source_folder} to {target_folder}")


def augment_data(img_path, mask_i_path, mask_o_path, output_dir, image_name, transform, times):
    # Read images and masks
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask_i = cv2.imread(str(mask_i_path), cv2.IMREAD_GRAYSCALE)
    mask_o = cv2.imread(str(mask_o_path), cv2.IMREAD_GRAYSCALE)

    # Save the original images and masks
    cv2.imwrite(str(output_dir / 'imgs' / image_name), image)
    cv2.imwrite(str(output_dir / 'i-masks' / image_name), mask_i)
    cv2.imwrite(str(output_dir / 'o-masks' / image_name), mask_o)

    for i in range(times):
        # Applying augmentations to the images and masks
        augmented = transform(image=image, masks=[mask_i, mask_o])
        image_aug, mask_i_aug, mask_o_aug = augmented['image'], augmented['masks'][0], augmented['masks'][1]

        # Save the augmented images and masks
        cv2.imwrite(str(output_dir / 'imgs' / f'aug{i}_{image_name}'), image_aug)
        cv2.imwrite(str(output_dir / 'i-masks' / f'aug{i}_{image_name}'), mask_i_aug)
        cv2.imwrite(str(output_dir / 'o-masks' / f'aug{i}_{image_name}'), mask_o_aug)


def augmentation(img_dir, mask_i_dir, mask_o_dir, output_dir, times):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'imgs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'i-masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'o-masks').mkdir(parents=True, exist_ok=True)

    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=45, p=0.5),
        RandomBrightnessContrast(p=0.2),
        GaussNoise(p=0.2),
        ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        RandomResizedCrop(height=256, width=216, scale=(0.3, 1.0), p=0.5)
    ])

    for img_name in tqdm(os.listdir(img_dir), desc='Augmenting images'):
        if img_name.endswith('.png'):
            img_path = img_dir / img_name
            mask_i_path = mask_i_dir / img_name
            mask_o_path = mask_o_dir / img_name

            augment_data(img_path, mask_i_path, mask_o_path, output_dir, img_name, transform, times)


def get_args():
    parser = argparse.ArgumentParser(description='Data preprocessing and augmentation')

    parser.add_argument('--times', '-t', type=int, default=4, help='Augmentation times')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    base_dir = '..'
    copy_and_process_files(base_dir, 'TrainingSet', 'train_data', 1, 16)
    copy_contours(base_dir, 'TestSet', "Test1SetContours", "Test1Set", 17, 32)
    copy_contours(base_dir, 'TestSet', "Test2SetContours", "Test2Set", 33, 48)
    copy_and_process_files(base_dir, 'TestSet/Test1Set', 'test1_data', 17, 32)
    copy_and_process_files(base_dir, 'TestSet/Test2Set', 'test2_data', 33, 48)

    print("Copying and processing files finished.")

    augmentation(Path('../train_data/imgs'), Path('../train_data/i-masks'), Path('../train_data/o-masks'),
                 Path('../train_data_aug'), args.times)

    print("Data augmentation finished.")