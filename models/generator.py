from torch import nn

from models.architecture import Encoder, Decoder


class Generator(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Generator, self).__init__()
        
        # Create Layer Instance
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
    
    def forward(self, mask, image) :
        # Create Masked Image
        maskedImage = image*(1-mask) + mask

        # Feed-Forward
        E0, E1, E2 = self.encoder(maskedImage, mask)
        outputImage = self.decoder([E0, E1, E2], mask)
        
        # Post-Process
        outputImage = image*(1-mask) + outputImage*mask
        
        return maskedImage, outputImage