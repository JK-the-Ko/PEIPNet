import random

from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


def generateMask(opt, kernelList:list, mask:Image.Image) -> Image.Image :
    # Get Image Size
    mask = np.array(mask, dtype="uint8")
    height, width = mask.shape[0], mask.shape[1]
    
    # Normalize and Threshold
    mask = mask/255
    mask = cv2.threshold(mask, 0.6, 1, cv2.THRESH_BINARY_INV)[-1]
    
    # Apply Random Dilation
    kernelSize = random.choice(kernelList)
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    mask = cv2.dilate(mask, kernel)
    
    # Apply Random Translation
    dx, dy = random.randint(-opt.dx, opt.dx+1), random.randint(-opt.dy, opt.dy+1)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]]) 
    mask = cv2.warpAffine(mask, matrix, (width+dx, height+dy))
    
    # Apply Random Rotation
    angle = random.randint(-opt.angle, opt.angle+1)
    matrix = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
    mask = cv2.warpAffine(mask, matrix, (width, height))
    
    # Apply Random Cropping 
    startX, startY = random.randint(0, width-opt.imageSize), random.randint(0, height-opt.imageSize)
    mask = mask[startY:startY+opt.imageSize, startX:startX+opt.imageSize]
    
    # Threshold Final Mask
    mask = np.where(mask>0, 1, 0)
    
    # Compute Ratio
    ratio = mask.sum()/opt.imageSize**2
    
    # Convert into Pillow Image
    mask = (mask*255).astype("uint8")
    mask = Image.fromarray(mask)

    return ratio, mask


def preprocessData(opt, mask, image, forMetrics, expandDim) :
    height, width = opt.imageSize, opt.imageSize
        
    # Resize Mask
    inputWidth, inputHeight = mask.size
    if inputWidth != width or inputHeight != height :
        mask = TF.resize(mask, (height, width), InterpolationMode.NEAREST)

    # Resize Image
    inputWidth, inputHeight = image.size
    if inputWidth != width or inputHeight != height :
        image = TF.resize(image, (height, width), InterpolationMode.BICUBIC)
        
    # Apply Horizontal Flip
    if not forMetrics :
        if random.random() < 0.5 :
            image = TF.hflip(image)
    
    # Convert to PyTorch Tensor
    mask = TF.to_tensor(mask)
    image = TF.to_tensor(image)

    # Apply Normalization
    image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    if expandDim :
        return mask.unsqueeze(0), image.unsqueeze(0)
    else :
        return mask, image