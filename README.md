# Flood water mapping segmentation pipeline


```Tensorflow.keras``` Implementation

## Introduction

The image segmentation model can be used to extract real-world objects from images, blur backgrounds, create self-driving automobiles and perform other image-processing tasks. This pipeline aims to train segmentation models in any segmentation task by modifying the dataset class based on the study.

## Models

In this repository, we implement UNET, U2NET, UNET++, VNET, DNCNN, and MOD-UNET using the `Keras-TensorFLow` framework. We also add `keras_unet_collection`(`kuc`) and `segmentation-models`(`sm`) library models which are also implemented using `Keras-TensorFLow`. The following models are available in this repository.

| Model | Name | Reference |
|:---------------|:----------------|:----------------|
| `dncnn`     | DNCNN         | [Zhang et al. (2017)](https://ieeexplore.ieee.org/document/7839189) |
| `unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `vnet`      | V-net (modified for 2-d inputs) | [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) |
| `unet++` | U-net++         | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `u2net`     | U^2-Net         | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
| `fapnet`     | FAPNET         | [Islam et al. (2022)](https://www.mdpi.com/1424-8220/22/21/8245) |
|  | [**keras_unet_collection**](https://github.com/yingkaisha/keras-unet-collection) |  |
| `kuc_r2unet`   | R2U-Net         | [Alom et al. (2018)](https://arxiv.org/abs/1802.06955) |
| `kuc_attunet`  | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |
| `kuc_restunet` | ResUnet-a       | [Diakogiannis et al. (2020)](https://doi.org/10.1016/j.isprsjprs.2020.01.013) |
| `kuc_unet3pp` | UNET 3+        | [Huang et al. (2020)](https://arxiv.org/abs/2004.08790) |
| `kuc_tensnet` | Trans-UNET       | [Chen et al. (2021)](https://arxiv.org/abs/2102.04306) |
| `kuc_swinnet` | Swin-UNET       | [Hu et al. (2021)](https://arxiv.org/abs/2105.05537) |
| `kuc_vnet`      | V-net (modified for 2-d inputs) | [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) |
| `kuc_unetpp` | U-net++         | [Zhou et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) |
| `kuc_u2net`     | U^2-Net         | [Qin et al. (2020)](https://arxiv.org/abs/2005.09007) |
|  | [**segmentation-models**](https://github.com/qubvel/segmentation_models) |  |
| `sm_unet`      | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| `sm_linknet`     | LINK-Net         | [Chaurasia et al. (2017)](https://arxiv.org/pdf/1707.03718.pdf) |
| `sm_fpn`     | FPS-Net         | [Xiao et al. (2021)](https://arxiv.org/pdf/2103.00738.pdf) |
| `sm_fpn`     | PSP-Net         | [Zhao et al. (2017)](https://arxiv.org/pdf/1612.01105.pdf) |

## Setup

First clone the GitHub repo in your local or server machine by following:
```
git clone https://github.com/MojammelHossain/flood_water_mapping_segmentation.git
```

Create a new environment and install dependencies from the `requirement.txt` file. Before starting training check the variable inside config.yaml i.e. `height`, `in_channels`, `dataset_dir`, `root_dir` etc. **Remember to change `read_img()` and `transform_data()` as per your dataset.**

## Experiments

After setup the required folders and package run one of the following experiments. There are four experiments based on the combination of parameters passing through `argparse` and `config.yaml`. The combination of each experiment is given below. 

When you run the following code based on different experiments, some new directories will be created;
1. csv_logger (save all evaluation results in CSV format)
2. logs (tensorboard logger)
3. model (save model checkpoint)
4. prediction (validation and test prediction png format)

* **Comprehensive Full Resolution (CFR)**: This experiment utilizes the dataset as it is.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr \
    --patchify False \
    --patch_size 512 \
    --weights False \
```

* **Comprehensive Full Resolution with Class Balance (CFR-CB)**: We balance the dataset biases in this experiment using passed class weight. 

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr_cb \
    --patchify False \
    --patch_size 512 \
    --weights True \
```

* **Patchify Half Resolution (PHR)**: In this experiment, we take all the patch images for each image.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr \
    --patchify True \
    --patch_size 256 \
    --weights False \ 
```

* **Patchify Half Resolution with Class Balance (PHR-CB)**: In this experiment, we take a threshold value (XX%) of desired class and remove the patch images for each image that are less than the threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --patchify True \
    --patch_size 256 \
    --weights False \
    --patch_class_balance True
```

## Testing

* **CFR and CFR-CB Experiment**

Run the following model for evaluating the trained model on the test dataset.

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --plot_single False \
    --index -1 \
    --patchify False \
    --patch_size 512 \
    --experiment cfr \
```

* **PHR and PHR-CB Experiment**

```
python train.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --plot_single False \
    --index -1 \
    --patchify True \
    --patch_size 256 \
    --experiment phr \
```

## Notes
I completely renovated this repository from its prior version while I was a machine learning engineer at Canada Syntax. The original repository is openly accessible [here](https://github.com/samiulengineer/flood_water_mapping_segmentation).
