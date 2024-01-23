import numpy as np
import pydicom
import matplotlib.pyplot as plt

# training set:
#   images: ./TrainingSet/patient01/P01dicom/P01-0000.dcm
#   contours: ./TrainingSet/patient01/P01contours-manual/P01-0080-icontour-manual.txt
#   paths of contours: ./TrainingSet/patient01/P01list.txt


def read_contour_file(filename):
    """
    读取包含轮廓坐标的文件
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        contour = [list(map(float, line.split())) for line in lines]
    return np.array(contour)


def plot_contour_on_dicom(dicom_file, contour_file):
    """
    在 DICOM 图像上绘制轮廓
    """
    # 读取 DICOM 文件
    dcm = pydicom.dcmread(dicom_file)
    image = dcm.pixel_array

    # 读取轮廓文件
    contour = read_contour_file(contour_file)

    # 显示 DICOM 图像
    plt.imshow(image, cmap=plt.cm.bone)

    # 在图像上绘制轮廓
    plt.plot(contour[:, 0], contour[:, 1], 'r.')  # 使用红点绘制轮廓

    # 显示带有轮廓的图像
    plt.title("DICOM Image with Contour")
    plt.show()


if __name__ == "__main__":
    # 加载 DICOM 文件
    dicom_filename = 'TrainingSet/patient01/P01dicom/P01-0080.dcm'
    contour_filename = 'TrainingSet/patient01/P01contours-manual/P01-0080-ocontour-manual.txt'
    plot_contour_on_dicom(dicom_filename, contour_filename)