[中文版README](README_zh.md)

## Preparation

Create a working directory and pull the code.

```
mkdir ws
cd ws
mkdir src
cd src
git clone https://github.com/omigeft/RVSC-Medical-Image-Segmentation.git
```

## Data Preprocess

Extract the dataset to the 'ws' directory and make the dataset file structure as follows:

```
/TrainingSet
/TestSet/
    /Test1Set
    /Test2Set
    /Test1SetContours
    /Test2SetContours
```

Enter the source code directory and run `data_preprocess.py`, automatically processes the RVSC dataset into a trainable format and performs data augmentation.

```sh
cd src
python data_preprocess.py
```

## Training

In the source code directory, run `train.py` to start training.

```sh
python train.py \
--model unet \
--epochs 50 \
--batch-size 16 \
--scale 0.5 \
-o adam \
--amp
```

## Predicting

```sh
python predict.py \
--pth ../i-checkpoints/unet_checkpoint_epoch2.pth \
--input ../train_data/imgs/P01-0080.png \
--scale 0.5 \
--viz \
--no-save
```

## Evaluating a model on test dataset

```sh
python eval_test.py \
--pth ../i-checkpoints/unet_checkpoint_epoch2.pth \
--input ../test1_data/imgs/ \
--output ../test1_data/i-masks \
--scale 0.5
```