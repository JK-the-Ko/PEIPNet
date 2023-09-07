from os import makedirs
from os.path import join, exists

import random

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import lpips

import torch
from torch import nn


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fixSeed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class AverageMeter(object) :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) :
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


def saveNetwork(opt, model, numIter, latest=False, best=False) :
    # Get Save Directory
    path = join(opt.checkpointsDir, opt.name, opt.dataType, "models")
    mkdirs(path)
    if latest :
        torch.save(model.module.netG.state_dict(), f"{path}/iter-{numIter}-G.pth")
        torch.save(model.module.netG.state_dict(), f"{path}/latest-G.pth")
    elif best :
        torch.save(model.module.netG.state_dict(), f"{path}/best-G.pth")


def loadNetwork(network, networkType, saveType, opt) :
    # Get Path Directory
    saveFileName = f"{saveType}-{networkType}.pth"
    savePath = join(opt.checkpointsDir, opt.name, opt.dataType, "models", saveFileName)
    
    # Load Network
    weights = torch.load(savePath)
    network.load_state_dict(weights)
    
    return network


class LPIPS(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(LPIPS, self).__init__()
        
        # Create LPIPS Instance
        self.model = lpips.LPIPS(net="alex")
        
        # Assign Device
        if opt.gpuIds != "-1" :
            self.model = self.model.cuda()

    def forward(self, fakeImage, realImage) :
        # Compute LPIPS
        dist = self.model.forward(fakeImage, realImage)
    
        return dist.mean()


def computePSNR(fakeImage, realImage) :
    psnr = 0
    for i in range(len(fakeImage)) :
        psnr += (peak_signal_noise_ratio(tensorToImage(realImage[i]), 
                                         tensorToImage(fakeImage[i]))/len(fakeImage))

    return psnr


def computeSSIM(fakeImage, realImage) :
    ssim = 0
    for i in range(len(fakeImage)) :
        ssim += (structural_similarity(tensorToImage(realImage[i]), 
                                       tensorToImage(fakeImage[i]), 
                                       channel_axis=2, 
                                       full=True)[0]/len(fakeImage))

    return ssim


class imageSaver :
    def __init__(self, opt) :
        self.opt = opt

    def visualizeBatch(self, resultList) :
        imageList = []
        for i in range(len(resultList[0])) :
            subImageList =[]
            for subImage in resultList :
                subImageList.append(tensorToImage(subImage[i]))
            imageSet = np.hstack(subImageList)
            imageList.append(imageSet)
        return np.vstack(imageList)


def tensorToImage(tensor, affine=True) :
    if affine :
        imageTensor = (tensor+1)/2
    else :
        imageTensor = tensor
    imageTensor = torch.clamp(imageTensor*255, 0, 255)
    imageNumpy = np.transpose(imageTensor.detach().cpu().numpy(), (1, 2, 0)).astype("uint8")
    return imageNumpy


def saveImage(tensor, imageSavePath, name, affine=True) :
    image = tensorToImage(tensor[0,:,:,:], affine)
    cv2.imwrite(join(imageSavePath, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def mkdirs(path) :
    # Make Directory
    if not exists(path) :
        makedirs(path)