# PEIPNet: Parametric Efficient Image-Inpainting Network with Depthwise and Pointwise Convolution
### [Paper](https://www.mdpi.com/1424-8220/23/19/8313) | [BibTex](#citation)
## Abstract
Research on image-inpainting tasks has mainly focused on enhancing performance by augmenting various stages and modules. However, this trend does not consider the increase in the number of model parameters and operational memory, which increases the burden on computational resources. To solve this problem, we propose a Parametric Efficient Image InPainting Network (PEIPNet) for efficient and effective image-inpainting. Unlike other state-of-the-art methods, the proposed model has a one-stage inpainting framework in which depthwise and pointwise convolutions are adopted to reduce the number of parameters and computational cost. To generate semantically appealing results, we selected three unique components: spatially-adaptive denormalization (SPADE), dense dilated convolution module (DDCM), and efficient self-attention (ESA). SPADE was adopted to conditionally normalize activations according to the mask in order to distinguish between damaged and undamaged regions. The DDCM was employed at every scale to overcome the gradient-vanishing obstacle and gradually fill in the pixels by capturing global information along the feature maps. The ESA was utilized to obtain clues from unmasked areas by extracting long-range information. In terms of efficiency, our model has the lowest operational memory compared with other state-of-the-art methods. Both qualitative and quantitative experiments demonstrate the generalized inpainting of our method on three public datasets: Paris StreetView, CelebA, and Places2.

## Inpainting Performance
- ### Paris StreetView
  ![image](https://github.com/JK-the-Ko/PEIPNet/assets/55126482/561c45e2-897d-417f-992d-5d78c5467ba7)
- ### CelebA
  ![image](https://github.com/JK-the-Ko/PEIPNet/assets/55126482/0a1a28f7-fffe-4676-b696-70b3ce221b0e)
- ### Places2
  ![image](https://github.com/JK-the-Ko/PEIPNet/assets/55126482/4313357b-bfa9-49ba-921f-bcb63be0204b)
  
## Prerequisites
- Python 3.8.10
- PyTorch>=1.12.1
- Torchvision>=0.13.1
- NVIDIA GPU + CUDA cuDNN

## Installation
- ### Clone this repo.
```
git clone https://github.com/JK-the-Ko/PEIPNet.git
cd PEIPNet/
```
- ### Install PyTorch and dependencies from http://pytorch.org
- ### Please install dependencies by
```
pip install -r requirements.txt
```

## Dataset
- Binary Mask dataset. The following dataset should be placed in ```dataset/mask``` folder.
- Paris StreetView dataset. The following dataset should be placed in ```dataset/paris-streetview``` folder.
- CelebA dataset. The following dataset should be placed in ```dataset/celeba``` folder.
- Places2 dataset. The following dataset should be placed in ```dataset/places2``` folder.

## Training
The following script is for training **Paris StreetView dataset**. We recommend using commands written in the scripts folder.
```
python train.py --dataType paris-streetview --gpuIds GPUIDS --batchSize BATCHSIZE --numWorkers NUMWORKERS
```
The following script is for training **CelebA dataset**. We recommend using commands written in the scripts folder.
```
python train.py --dataType celeba --gpuIds GPUIDS --batchSize BATCHSIZE --numWorkers NUMWORKERS
```
The following script is for training **Places2 dataset**. We recommend using commands written in the scripts folder.
```
python train.py --dataType places2 --gpuIds GPUIDS --batchSize BATCHSIZE --numWorkers NUMWORKERS
```

## Evaluation
The following script is for evaluating **Paris StreetView dataset**. We recommend using commands written in the scripts folder.
```
python test.py --dataType paris-streetview --gpuIds 0
```
The following script is for evaluating **CelebA dataset**. We recommend using commands written in the scripts folder.
```
python test.py --dataType celeba --gpuIds 0
```
The following script is for evaluating **Places2 dataset**. We recommend using commands written in the scripts folder.
```
python test.py --dataType places2 --gpuIds 0
```
The final result will be saved automatically in ```results/dataset-name``` folder.

## Pre-Trained Models
Pre-trained models are located in ```checkpoints/dataset-name/models``` folder.

## Citation
If you use **PEIPNet** in your work, please consider citing us as

```
@Article{s23198313,
AUTHOR = {Ko, Jaekyun and Choi, Wanuk and Lee, Sanghwan},
TITLE = {PEIPNet: Parametric Efficient Image-Inpainting Network with Depthwise and Pointwise Convolution},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {19},
ARTICLE-NUMBER = {8313},
URL = {https://www.mdpi.com/1424-8220/23/19/8313},
ISSN = {1424-8220},
DOI = {10.3390/s23198313}
}
```
