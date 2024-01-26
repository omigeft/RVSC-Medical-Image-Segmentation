[中文版README](README_zh.md)

## Preparation

Create a working directory and pull the code.

```
mkdir ws
cd ws
git clone https://github.com/omigeft/RVSC-Medical-Image-Segmentation.git src
```

Tested on Ubuntu 22.04, RTX 4090 * 1, CUDA 12.1, Python=3.10, torch=2.1.2, torchvision=0.16.2. Other similar versions should also work.

To install other required packages, run:

```sh
pip install -r requirements.txt
```

To visualize the training process, you need to register an account on [Weights & Biases](https://wandb.ai/). Then run the following command and follow the instructions to log in.

```sh
wandb login
```

## Data Preprocess

Extract the dataset to the `ws` directory and make the dataset file structure as follows:

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
--batch-size 64 \
--scale 0.5 \
-w 1e-4 \
-epc 5 \
-ls dice+ce \
-o adam \
--amp
```

If CUDA runs out of memory, try reducing the batch size or scaling down the images. Conversely, if the GPU utilization is too low, try increasing the batch size or scaling up the images.

## Predicting

```sh
python predict.py \
--pth ../i-checkpoints/unet_checkpoint_epoch50.pth \
--input ../train_data/imgs/P01-0080.png \
--scale 0.5 \
--viz \
--no-save
```

## Evaluating a model on test dataset

```sh
python eval_test.py \
--pth ../i-checkpoints/unet_checkpoint_epoch50.pth \
--input ../test1_data/imgs/ \
--output ../test1_data/i-masks \
--scale 0.5
```