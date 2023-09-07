from os import listdir
from os.path import join

import random

from PIL import Image

from natsort import natsorted

from torch.utils.data import Dataset

from data.utils import generateMask, preprocessData


class Places2Dataset(Dataset) :
    def __init__(self, opt, forMetrics) :
        # Inheritance
        super(Places2Dataset, self).__init__()
        
        self.opt = opt
        self.kernelList = [3,4,5,6,7]
        self.forMetrics = forMetrics
        self.maskDataset, self.imageDataset = self.getPathList()

    def __getitem__(self, index):
        # Load Data
        image = Image.open(join(self.imageDataset[index][0], self.imageDataset[index][1])).convert("RGB")
        
        # Generate Mask
        ratio = 0
        while ratio < self.opt.minRatio or self.opt.maxRatio < ratio :
            mask = Image.open(join(self.maskDataset[1], random.choice(self.maskDataset[0]))).convert("L")
            ratio, mask = generateMask(self.opt, self.kernelList, mask)

        # Transform Data
        mask, image = self.transforms(mask, image)
        
        return {"mask":mask, "image":image, "name":self.imageDataset[index][1]}

    def __len__(self):
        return len(self.imageDataset)

    def getPathList(self) :
        # Set Mode
        mode = "valid" if self.forMetrics else "train"
        
        # Get Absolute Parent Path
        maskPath = join(self.opt.dataRoot, "mask", "train")
        imageClassPath = join(self.opt.dataRoot, self.opt.dataType, mode)
        
        # Create List Instance for Adding Dataset Path
        maskPathList = listdir(maskPath)
 
        # Create List Instance for Adding File Name
        maskNameList = [maskName for maskName in maskPathList if ".png" in maskName]
        imageDataList = []
        for className in listdir(imageClassPath) :
            for imageName in listdir(join(imageClassPath, className)) :
                if ".png" in imageName or ".jpg" in imageName or ".PNG" in imageName or ".JPG" in imageName :
                    imageDataList.append([join(imageClassPath, className), imageName])
        
        # Sort List Instance
        maskNameList = natsorted(maskNameList)
        imageDataList = natsorted(imageDataList)
        
        return (maskNameList, maskPath), imageDataList
    
    def transforms(self, mask, image) :
        return preprocessData(self.opt, mask, image, self.forMetrics, expandDim=False)