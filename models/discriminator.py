import math

import torch
from torch import nn
import torch.nn.functional as F

from models.conv2d import spectralNormalization


class MultiscaleDiscriminator(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(MultiscaleDiscriminator, self).__init__()
        
        # Create Discriminator Instance
        for i in range(opt.numD) :
            subnetD = self.createDiscriminatorInstance(opt)
            self.add_module(f"discriminator_{i}", subnetD)

    def createDiscriminatorInstance(self, opt) :
        return NLayerDiscriminator(opt)

    def downsample(self, input) :
        return F.interpolate(input, scale_factor=0.5, mode="nearest")

    def forward(self, fakeImage, realImage) :
        output = []
        for _, D in self.named_children() :
            subOutput = D(fakeImage, realImage)
            output.append(subOutput)
            fakeImage, realImage = self.downsample(fakeImage), self.downsample(realImage)

        return output


class NLayerDiscriminator(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(NLayerDiscriminator, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        inputDim = opt.inputDim+1 
        channels = opt.channelsD
        kernelSize = 4
        padding = int((kernelSize-1)/2)
        
        # Create Spectral Normalization Instance
        spectralNorm = spectralNormalization(opt.noSpectralNormD)
        
        # Create Convolutional Layer and Activation Function Instance
        self.network = nn.ModuleList([])
        self.network.append(spectralNorm(nn.Conv2d(inputDim, channels, kernel_size=kernelSize, stride=2, padding=padding)))
        self.network.append(nn.LeakyReLU(0.2))
        for i in range(1, opt.numLayerD) :
            channelsPrev = channels
            channels = min(channelsPrev*2, 512)
            stride = 1 if i == opt.numLayerD-1 else 2
            self.network.append(spectralNorm(nn.Conv2d(channelsPrev, channels, kernel_size=kernelSize, stride=stride, padding=padding)))
            self.network.append(nn.LeakyReLU(0.2))
        self.network.append(nn.Conv2d(channels, 1, kernel_size=kernelSize, stride=1, padding=padding))

    def forward(self, fakeImage, realImage) :
        # Concatenate Input
        input = torch.cat([fakeImage, realImage], dim=0)
        output = [input]
        for layer in self.network :
            intermediateOutput = layer(output[-1])
            output.append(intermediateOutput)
        
        return output[1:]