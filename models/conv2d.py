import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class DConv2d(nn.Module) :
    def __init__(self, opt, inChannels, kernelSize, stride, padding, dilation=1) :
        # Inheritance
        super(DConv2d, self).__init__()
        
        # Create Spectral Normalization Instance
        spectralNorm = spectralNormalization(opt.noSpectralNormG)
        
        # Create Convolutional Layer Instance
        self.conv = spectralNorm(nn.Conv2d(inChannels, inChannels, 
                                           kernel_size=kernelSize, stride=stride, 
                                           padding=padding, dilation=dilation, groups=inChannels, bias=False))
        
    def forward(self, input) :
        output = self.conv(input)
        
        return output

    
class SConv2d(nn.Module) :
    def __init__(self, opt, inChannels, outChannels) :
        # Inheritance
        super(SConv2d, self).__init__()
        
        # Create Spectral Normalization Instance
        spectralNorm = spectralNormalization(opt.noSpectralNormG)
        
        # Create Convolutional Layer Instance
        self.conv = spectralNorm(nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False))
        
    def forward(self, input) :
        output = self.conv(input)
        
        return output


def spectralNormalization(noSpectralNorm) :
    if noSpectralNorm :
        return nn.Identity()
    else :
        return spectral_norm


def normalizationLayer(opt, channels) :
    if opt.normType == "instance" :
        return nn.InstanceNorm2d(channels, affine=False)
    elif opt.normType == "batch" :
        return nn.BatchNorm2d(channels, affine=False)
    elif opt.normType == "none" :
        return nn.Identity()
    else :
        # Error Handling
        raise NotImplementedError(f"{opt.normType} is not supported in this model")
    

class CNorm(nn.Module) :
    def __init__(self, opt, inChannels, hiddenChannels, kernelSize) :
        super(CNorm, self).__init__()
        
        # Create Normalization Layer
        self.norm = normalizationLayer(opt, inChannels)

        # Create Convolutional Layer Instance
        self.convShared = nn.Sequential(DConv2d(opt, 2, kernelSize, 1, kernelSize//2, 1),
                                        SConv2d(opt, 2, hiddenChannels))
        self.convGamma = nn.Sequential(DConv2d(opt, hiddenChannels, kernelSize, 1, kernelSize//2, 1),
                                       SConv2d(opt, hiddenChannels, inChannels))
        self.convBeta = nn.Sequential(DConv2d(opt, hiddenChannels, kernelSize, 1, kernelSize//2, 1),
                                      SConv2d(opt, hiddenChannels, inChannels))

    def forward(self, input, mask) :
        outputNorm = self.norm(input)
        
        mask = F.interpolate(mask, size=input.size()[2:], mode="nearest")
        
        outputShared = self.convShared(torch.cat([mask, 1-mask], dim=1))
        outputGamma = self.convGamma(outputShared)
        outputBeta = self.convBeta(outputShared)

        output = outputNorm*(1+outputGamma) + outputBeta

        return output


class Conv2dNorm(nn.Module) :
    def __init__(self, opt, inChannels, outChannels, kernelSize, stride, padding, dilation) :
        # Inheritance
        super(Conv2dNorm, self).__init__()
        
        if outChannels >= inChannels :
            self.conv0 = DConv2d(opt, inChannels, kernelSize, stride, padding, dilation)
            self.conv1 = SConv2d(opt, inChannels, outChannels)
            self.norm0 = CNorm(opt, inChannels, 16, kernelSize)
            self.norm1 = CNorm(opt, outChannels, 16, kernelSize)
        else :
            self.conv0 = SConv2d(opt, inChannels, outChannels)
            self.conv1 = DConv2d(opt, outChannels, kernelSize, stride, padding, dilation)
            self.norm0 = CNorm(opt, outChannels, 16, kernelSize)
            self.norm1 = CNorm(opt, outChannels, 16, kernelSize)

    def forward(self, input, mask) :
        output = F.leaky_relu(self.norm0(self.conv0(input), mask), 0.2)
        output = F.leaky_relu(self.norm1(self.conv1(output), mask), 0.2)
            
        return output