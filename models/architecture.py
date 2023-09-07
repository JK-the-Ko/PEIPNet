import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights

from models.conv2d import Conv2dNorm, SConv2d
from models.attention import ESA


class Encoder(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Encoder, self).__init__()
        
        # Initialize Variable
        inputDim = opt.inputDim
        channels = opt.channelsG
        
        # Create Convolutionalal Layer Instance
        self.EB0 = Conv2dNorm(opt, inputDim, channels, 3, 1, 1, 1)
        self.EB1 = Conv2dNorm(opt, channels, channels*2, 3, 2, 1, 1)
        self.EB2 = Conv2dNorm(opt, channels*2, channels*4, 3, 2, 1, 1)

        # Create DDCM Instance
        self.DDCM0 = DDCM(opt, channels, reduction=2)
        self.DDCM1 = DDCM(opt, channels*2, reduction=2)
        self.DDCM2 = DDCM(opt, channels*4, reduction=2)

    def forward(self, input, mask) :
        E0 = self.EB0(input, mask)
        E1 = self.EB1(E0, mask)
        E2 = self.EB2(E1, mask)

        return self.DDCM0(E0, mask), self.DDCM1(E1, mask), self.DDCM2(E2, mask)


class Decoder(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Decoder, self).__init__()
        
        # Initialize Variable
        inputDim = opt.channelsG*4
        outputDim = opt.inputDim
        channels = opt.channelsG
        
        # Create Upsampling Layer Instance
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Create Convolutionalal Layer Instance
        self.DB0 = Conv2dNorm(opt, inputDim+channels*2, channels*2, 3, 1, 1, 1)
        self.DB1 = Conv2dNorm(opt, channels*2+channels, channels, 3, 1, 1, 1)
        self.DB2 = SConv2d(opt, channels, outputDim)
        
        # Create Efficient Self-Attention Instance
        self.ESA0 = ESA(opt, inputDim+channels*2, channels*4, channels*4, 16)
        self.ESA1 = ESA(opt, channels*2+channels, channels*2, channels*2, 8)
        
    def forward(self, scList, mask) :
        D0 = self.DB0(self.ESA0(torch.cat([self.upsample(scList[-1]), scList[-2]], dim=1)), mask)
        D1 = self.DB1(self.ESA1(torch.cat([self.upsample(D0), scList[-3]], dim=1)), mask)
        D2 = torch.tanh(self.DB2(D1))
        
        return D2


class DDCM(nn.Module) :
    def __init__(self, opt, inChannels, reduction) :
        # Inheritance
        super(DDCM, self).__init__()
        
        # Initialize Variable
        midChannels = inChannels//reduction
        
        # Create Convolutionalal Layer Instance
        self.convIn = Conv2dNorm(opt, inChannels, inChannels, 3, 1, 1, 1)
        self.conv0 = Conv2dNorm(opt, midChannels, midChannels, 3, 1, 2, 2)
        self.conv1 = Conv2dNorm(opt, midChannels, midChannels, 3, 1, 4, 4)
        self.conv2 = Conv2dNorm(opt, midChannels, midChannels, 3, 1, 6, 6)
        self.conv3 = Conv2dNorm(opt, midChannels, midChannels, 3, 1, 8, 8)
        self.conv4 = Conv2dNorm(opt, midChannels, midChannels, 3, 1, 10, 10)
        self.convOut = Conv2dNorm(opt, inChannels, inChannels, 3, 1, 1, 1)

        # Create Bottleneck Layer Instance
        self.bnckIn = SConv2d(opt, inChannels, midChannels)
        self.bnck0 = SConv2d(opt, midChannels*2, midChannels)
        self.bnck1 = SConv2d(opt, midChannels*3, midChannels)
        self.bnck2 = SConv2d(opt, midChannels*4, midChannels)
        self.bnckOut = SConv2d(opt, midChannels*5, inChannels)

    def forward(self, input, mask) :
        output = self.bnckIn(self.convIn(input, mask))
        
        f0 = self.conv0(output, mask)
        f1 = self.conv1(f0, mask)
        f2 = self.conv2(self.bnck0(torch.cat([f0, f1], dim=1)), mask)
        f3 = self.conv3(self.bnck1(torch.cat([f0, f1, f2], dim=1)), mask)
        f4 = self.conv4(self.bnck2(torch.cat([f0, f1, f2, f3], dim=1)), mask)
        
        output = self.convOut(self.bnckOut(torch.cat([f0, f1, f2, f3, f4], dim=1)), mask)

        return output


class VGG19(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(VGG19, self).__init__()
        
        # Load Pretrained Vgg Network
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        
        # Create List Instance for Adding Layers
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # Add Layers
        for layer in range(2) :
            self.slice1.add_module(str(layer), model[layer])
        for layer in range(2, 7) :
            self.slice2.add_module(str(layer), model[layer])
        for layer in range(7, 12) :
            self.slice3.add_module(str(layer), model[layer])
        for layer in range(12, 21) :
            self.slice4.add_module(str(layer), model[layer])
        for layer in range(21, 30) :
            self.slice5.add_module(str(layer), model[layer])
        
        # Fix Gradient Flow
        for param in self.parameters() :
            param.requires_grad = False

    def forward(self, input) :
        hRelu1 = self.slice1(input)
        hRelu2 = self.slice2(hRelu1)
        hRelu3 = self.slice3(hRelu2)
        hRelu4 = self.slice4(hRelu3)
        hRelu5 = self.slice5(hRelu4)

        return [hRelu1, hRelu2, hRelu3, hRelu4, hRelu5]