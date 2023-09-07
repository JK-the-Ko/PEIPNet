import torch
from torch import nn
import torch.nn.functional as F

from models.conv2d import SConv2d


class ESA(nn.Module) :
    def __init__(self, opt, inChannels, keyChannels, valueChannels, headCount) :
        # Inheritance
        super(ESA, self).__init__()
        
        # Initialize Variables
        self.inChannels = inChannels
        self.keyChannels = keyChannels
        self.valueChannels = valueChannels
        self.headCount = headCount

        # Create Convolutional Layer Instance
        self.keys = SConv2d(opt, inChannels, keyChannels)
        self.queries = SConv2d(opt, inChannels, keyChannels)
        self.values = SConv2d(opt, inChannels, valueChannels)
        self.reProjection = SConv2d(opt, valueChannels, inChannels)

    def forward(self, input) :
        n, _, h, w = input.size()
        
        keys = self.keys(input).reshape((n, self.keyChannels, h*w))
        queries = self.queries(input).reshape(n, self.keyChannels, h*w)
        values = self.values(input).reshape((n, self.valueChannels, h*w))
        headKeyChannels = self.keyChannels//self.headCount
        headValueChannels = self.valueChannels//self.headCount
        
        attendedValues = []
        for i in range(self.headCount) :
            key = F.softmax(keys[:, i*headKeyChannels:(i+1)*headKeyChannels, :], dim=2)
            query = F.softmax(queries[:, i*headKeyChannels:(i+1)*headKeyChannels, :], dim=1)
            value = values[:, i*headValueChannels:(i+1)*headValueChannels, :]
            context = key @ value.transpose(1, 2)
            attendedValue = (context.transpose(1, 2) @ query).reshape(n, headValueChannels, h, w)
            attendedValues.append(attendedValue)

        aggregatedValues = torch.cat(attendedValues, dim=1)
        reProjectedValue = self.reProjection(aggregatedValues)
        attention = reProjectedValue + input

        return attention