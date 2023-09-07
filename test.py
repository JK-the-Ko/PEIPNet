from os import listdir
from os.path import join

import random

from natsort import natsorted

from PIL import Image

from tqdm import tqdm

import config
from data.utils import preprocessData
from models import models
from utils import utils


def main() : 
    # Read Options
    opt = config.readArguments(train=False)
    opt.gpuIds = "0"
    
    # Get Absolute Path of Input Dataset
    maskProbPath = join(opt.dataRoot, "mask", "test")
    imagePath = join(opt.dataRoot, opt.dataType, "test")
    
    # Sort Dataset Names
    maskProbList = natsorted(listdir(maskProbPath))
    imageNameList = natsorted(listdir(imagePath))

    # Create Model Instance
    model = models.PEIPNet(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    for maskProb in maskProbList :
        print(f"[Mask Prob. {maskProb}]")
        maskPath = join(opt.dataRoot, "mask", "test", maskProb)
        maskNameList = natsorted(listdir(maskPath))
        
        # Get Save Directory
        maskSavePath = join("results", opt.name, opt.dataType, maskProb, "mask")
        maskedImageSavePath = join("results", opt.name, opt.dataType, maskProb, "masked-image")
        imageSavePath = join("results", opt.name, opt.dataType, maskProb, "image")

        # Create Save Directory
        utils.mkdirs(maskSavePath)
        utils.mkdirs(maskedImageSavePath)
        utils.mkdirs(imageSavePath)
        
        with tqdm(total=len(imageNameList)) as pBar :
            for imageName in imageNameList :
                # Load Dataset
                mask = Image.open(join(maskPath, random.choice(maskNameList))).convert("L")
                image = Image.open(join(imagePath, imageName)).convert("RGB")
                
                # Preprocess Dataset and Assign Device
                mask, image = preprocessData(opt, mask, image, True, True)
                mask, image = models.assignDevice(opt, mask, image)
                
                # Get Final Results
                maskedImage, outputImage = model(mask, image, mode="inference")
                
                # Save Results
                utils.saveImage(mask, maskSavePath, imageName.replace("jpg", "png"), False)
                utils.saveImage(maskedImage, maskedImageSavePath, imageName)
                utils.saveImage(outputImage, imageSavePath, imageName)
                
                # Update Status
                pBar.set_description("[Test] < Saving Result >")
                pBar.update()


if __name__ == "__main__" :
    main()