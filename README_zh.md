[English README](README.md)

## 准备

创建工作目录，并拉取代码。

```
mkdir ws
cd ws
git clone https://github.com/omigeft/RVSC-Medical-Image-Segmentation.git src
```


在Ubuntu 22.04、CUDA 12.1、Python=3.10、torch=2.1.2、torchvision=0.16.2上测试。其他类似版本也应适用。

要安装其他所需的软件包，请运行：

```sh
pip install -r requirements.txt
```

要使培训过程可视化，您需要在[Weights&Biases](https://wandb.ai/)上注册一个帐户。然后运行以下命令并按照说明进行登录。

```sh
wandb login
```

## 数据预处理

解压数据集到`ws`目录下，数据集文件结构如下：

```
/TrainingSet
/TestSet/
    /Test1Set
    /Test2Set
    /Test1SetContours
    /Test2SetContours
```

进入源代码目录，运行`data_preprocess.py`，自动处理RVSC数据集为可训练的格式并进行数据增广。

```sh
cd src
python data_preprocess.py
```

## 训练

在源代码目录，运行`train.py`，开始训练。

```sh
python train.py \
--model unet \
--epochs 30 \
--batch-size 64 \
--scale 0.5 \
-w 1e-4 \
-epc 5 \
-ls dice+ce \
-o adam \
--amp
```

如果CUDA内存不足，请尝试减小批处理大小或缩小图像的比例。相反，如果GPU利用率过低，请尝试增加批处理大小或放大图像。 

## 预测

```sh
python predict.py \
--pth ../i-checkpoints/unet_checkpoint_epoch30.pth \
--input ../train_data/imgs/P01-0080.png \
--scale 0.5 \
--viz \
--no-save
```

## 利用测试集评估模型

```sh
python eval_test.py \
--pth ../i-checkpoints/unet_checkpoint_epoch30.pth \
--input ../test1_data/imgs/ \
--output ../test1_data/i-masks \
--scale 0.5
```