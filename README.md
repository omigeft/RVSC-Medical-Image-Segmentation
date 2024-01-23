## 准备

创建工作目录，并拉取代码。

```
mkdir ws
cd ws
mkdir src
cd src
git clone https://github.com/omigeft/RVSC-Medical-Image-Segmentation.git
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
--epochs 50 \
--batch-size 16 \
--scale 0.5 \
-o adam \
--amp
```

## 测试

```sh
python train.py \
--model ../i-checkpoints/checkpoint_epoch49.pth \
--input ../train_data/imgs/P01-0080.png \
--scale 0.5 \
--viz \
--no-save
```

## 利用测试集评估模型

```sh
python eval_test.py \
--model ../i-checkpoints/checkpoint_epoch50.pth \
--input ../test1_data/imgs/ \
--output ../test1_data/i-masks \
--scale 0.5
```

TODO: `eval_test.py`无法评估不同种类模型