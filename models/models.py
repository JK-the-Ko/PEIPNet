import torch
from torch import nn
from torch.nn import init

from models.generator import Generator
from models.discriminator import MultiscaleDiscriminator
from models.loss import GANLoss, FeatureLoss, ReconstructionLoss, VGGLoss
from utils import utils


class PEIPNet(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(PEIPNet, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        self.OKBLUE, self.ENDC = utils.bcolors.OKBLUE, utils.bcolors.ENDC
        
        # Create Generator Instance
        self.netG = Generator(opt)
        
        # Create Discriminator Instance
        if opt.phase == "train" :
            self.netD = MultiscaleDiscriminator(opt)
            self.FloatTensor = torch.cuda.FloatTensor if self.useGPU() else torch.FloatTensor
        
        # Compute Number of Parameters
        self.computeNumParameter()
        
        # Weight Initialization
        utils.fixSeed(opt.seed)
        self.initializeNetwork()
        self.loadCheckpoints()
        
        # Create Loss Function Instance
        if opt.phase == "train" :
            self.criterionGAN = GANLoss(opt.GANMode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionVGG = VGGLoss()

    def forward(self, mask, realImage, mode) :
        # Feed-Forward
        if mode == "generator" :
            loss, maskedImage, outputImage = self.computeGeneratorLoss(mask, realImage)
            return loss, maskedImage, outputImage
        elif mode == "discriminator" :
            loss = self.computeDiscriminatorLoss(mask, realImage)
            return loss
        elif mode == "inference" :
            self.netG.eval()
            with torch.no_grad() :
                maskedImage, outputImage = self.generateFakeImage(mask, realImage)
            return maskedImage, outputImage
        else :
            raise NotImplementedError(f"{mode} is not supported")

    def loadCheckpoints(self) :
        # Inference Final Results
        if self.opt.phase == "test" :
            saveType = self.opt.saveType
            self.netG = utils.loadNetwork(self.netG, "G", saveType, self.opt)

    def computeNumParameter(self) :
        if self.opt.phase == "train" :
            networkList = [self.netG, self.netD]
        else :
            networkList = [self.netG]
        print(f"{self.OKBLUE}PEIPNet{self.ENDC}: Now Computing Model Parameters.")
        for network in networkList :
            numParameter = 0
            for _, module in network.named_modules() :
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.InstanceNorm2d) :
                    numParameter += sum([p.data.nelement() for p in module.parameters()])
            print(f"{self.OKBLUE}PEIPNet{self.ENDC}: {utils.bcolors.OKGREEN}[{network.__class__.__name__}]{self.ENDC} Total params : {numParameter:,}.")
        print(f"{self.OKBLUE}PEIPNet{self.ENDC}: Finished Computing Model Parameters.")

    def initializeNetwork(self) :
        if self.opt.phase=="train" :
            def init_weights(m, initType=self.opt.initType, gain=0.02) :
                className = m.__class__.__name__
                # Weight Initialization
                if hasattr(m, "weight") and className.find("Conv") != -1 :
                    if initType == "normal" :
                        init.normal_(m.weight.data, 0.0, gain)
                    elif initType == "xavier" :
                        init.xavier_normal_(m.weight.data, gain=gain)
                    elif initType == "xavier_uniform" :
                        init.xavier_uniform_(m.weight.data, gain=gain)
                    elif initType == "kaiming" :
                        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                    elif initType == "orthogonal" :
                        init.orthogonal_(m.weight.data, gain=gain)
                    elif initType == "none" :
                        m.reset_parameters()
                    else:
                        raise NotImplementedError(f"{initType} method is not supported")
                    if hasattr(m, "bias") and m.bias is not None :
                        init.constant_(m.bias.data, 0.0)

            # Create List Instance for Adding Network
            if self.opt.phase == "train" :
                networkList = [self.netG, self.netD]
            else :
                networkList = [self.netG]
        
            # Weight Initialization
            for network in networkList :
                network.apply(init_weights)

    def computeGeneratorLoss(self, mask, realImage) :
        # Create Dictionary Instance for Adding Loss
        lossG = {}
        
        # Get Inference Result
        maskedImage, outputImage = self.generateFakeImage(mask, realImage)
        predImageFake, predImageReal =  self.discriminate(mask, outputImage, realImage)
        
        # Compute GAN Loss
        lossG["Adv"] = self.criterionGAN(predImageFake, True, forDiscriminator=False)*self.opt.lambdaAdv
        
        # Compute Feature Loss
        lossG["Feature"] = FeatureLoss(predImageFake, predImageReal)*self.opt.lambdaFeature
        
        # Compute Reconstruction Loss
        lossG["Recon"] = ReconstructionLoss(mask, outputImage, realImage)*self.opt.lambdaReconstruction
        
        # Compute Perceptual Loss and Style Loss
        lossVGG = self.criterionVGG(outputImage, realImage)
        lossG["Percp"] = lossVGG[0]*self.opt.lambdaPerceptual
        lossG["Style"] = lossVGG[1]*self.opt.lambdaStyle
        
        return lossG, maskedImage, outputImage

    def computeDiscriminatorLoss(self, mask, realImage) :
        # Create Dictionary Instance for Adding Loss
        lossD = {}
        
        # Fix Generator Weights Gradient
        with torch.no_grad() :
            _, outputImage = self.generateFakeImage(mask, realImage)
            fakeImage = outputImage.detach()
            
        # Get Inference Result
        predImageFake, predImageReal = self.discriminate(mask, fakeImage, realImage)
        
        # Compute Loss
        lossD["Adv-Fake"] = self.criterionGAN(predImageFake, False, forDiscriminator=True)
        lossD["Adv-Real"] = self.criterionGAN(predImageReal, True, forDiscriminator=True)
        
        return lossD
            
    def generateFakeImage(self, mask, realImage) :
        # Get Inference Result
        maskedImage, outputImage = self.netG(mask, realImage)
        
        return maskedImage, outputImage
    
    def discriminate(self, mask, fakeImage, realImage) :
        # Concatenate Image Data Tensor 
        fakeImageConcat = torch.cat([mask, fakeImage], dim=1)
        realImageConcat = torch.cat([mask, realImage], dim=1)
        
        # Get Image Inference Result
        imageDOut = self.netD(fakeImageConcat, realImageConcat)
        predImageFake, predImageReal = self.dividePrediction(imageDOut)
        
        return predImageFake, predImageReal
    
    def dividePrediction(self, pred) :
        if isinstance(pred, list) :
            fake, real = [], []
            for subPred in pred :
                fake.append([tensor[:tensor.size(0)//2] for tensor in subPred])
                real.append([tensor[tensor.size(0)//2:] for tensor in subPred])
        else :
            fake = pred[:pred.size(0)//2]
            real = pred[pred.size(0)//2:]
            
        return fake, real
    
    def useGPU(self) :
        return self.opt.gpuIds != "-1"


def assignOnMultiGpus(opt, model) :
    if opt.gpuIds != "-1" :
        gpus = list(map(int, opt.gpuIds.split(",")))
        model = nn.DataParallel(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpuIds.split(",")) == 0 or opt.batchSize % len(opt.gpuIds.split(",")) == 0
    
    return model


def assignDevice(opt, mask, realImage) :
    # Assign Device
    if opt.gpuIds != "-1" :
        mask, realImage = mask.cuda(), realImage.cuda()
    
    return mask, realImage