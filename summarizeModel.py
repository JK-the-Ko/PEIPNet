import torch

import config
from models import models


def main() : 
    # Read Options
    opt = config.readArguments(train=True)

    # Create Model Instance
    model = models.EIPNet(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Create Dummy Variable Instances
    dummyMask  = torch.randn((1, 1, opt.imageSize, opt.imageSize))
    dummyImage = torch.randn((1, opt.inputDim, opt.imageSize, opt.imageSize))
    
    # Assign Device
    if opt.gpuIds != "-1" :
        dummyMask = dummyMask.cuda()
        dummyImage = dummyImage.cuda()
    
    # Print Model Components
    print(model.module.netG)
    print(model.module.netD)
    
    # Summarize Generator Outputs
    outputG = model.module.netG(dummyMask, dummyImage)
    print("Feed-Forward Successful! (Gen.)")
    for subOutput in outputG :
        for subSubOutput in subOutput :
            print(f"Output Size (G) : {subSubOutput.size()}")
    
    # Summarize Discriminator Outputs
    outputD = model.module.netD(torch.cat([dummyMask, dummyImage], dim=1), torch.cat([dummyMask, dummyImage], dim=1))
    print("Feed-Forward Successful! (Dis.)")
    for subOutput in outputD :
        print(f"Output Size (D) : {subOutput[-1].size()}")


if __name__ == "__main__" :
    main()
