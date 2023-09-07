import wandb

import torch
from torch import optim

from tqdm import tqdm

import config
from data import dataloaders
from models import models
from utils import utils


def main() : 
    # Read Options
    opt = config.readArguments(train=True)
    
    # Create DataLoader Instance
    trainDataLoader, testDataLoader = dataloaders.getDataLoaders(opt)

    # Create Model Instance
    model = models.PEIPNet(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Initialize Wandb Settings
    if not opt.noWandb :
        wandb.init(config=opt, project=opt.dataType)
        wandb.run.name = opt.name
        wandb.watch(model.module.netG)
        wandb.watch(model.module.netD)
        imageSaver = utils.imageSaver(opt)
    
    # Create Optimizer Instance
    optimizerG = optim.Adam(model.module.netG.parameters(), 
                            lr=opt.lrG, 
                            betas=(opt.beta1, opt.beta2))
    optimizerD = optim.Adam(model.module.netD.parameters(), 
                            lr=opt.lrD, 
                            betas=(opt.beta1, opt.beta2))
    
    # Create Scheduler Instance
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, 
                                                      T_max=opt.numIters, 
                                                      eta_min=opt.lrG*opt.decayRate)
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, 
                                                      T_max=opt.numIters, 
                                                      eta_min=opt.lrD*opt.decayRate)
    
    # Initialize Variables for Saving Weights
    bestLPIPS = torch.inf
    computeLPIPS = utils.LPIPS(opt)
    
    # Create AverageMeter Instance
    testLPIPS, testSSIM, testPSNR = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    
    # Start Training
    print("< Training Started >")
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    currentIter = 1    
    while currentIter < opt.numIters+1 :
        for data in trainDataLoader :
            if currentIter == opt.numIters+1 :
                break
            else :
                # Load Dataset and Assign Device
                mask, image = data["mask"], data["image"]
                mask, image = models.assignDevice(opt, mask, image)
                
                # Update Generator Weights
                optimizerG.zero_grad()
                lossG, maskedImage, outputImage = model(mask, image, mode="generator")
                trainAdvG, trainFeature = lossG["Adv"].mean().item(), lossG["Feature"].mean().item()
                trainRecon, trainPercp, trainStyle = lossG["Recon"].mean().item(), lossG["Percp"].mean().item(), lossG["Style"].mean().item()
                sum(lossG.values()).mean().backward()
                optimizerG.step()
                
                # Update Discriminator Weights
                optimizerD.zero_grad()
                lossD = model(mask, image, mode="discriminator")
                trainAdvRealD, trainAdvFakeD = lossD["Adv-Real"].mean().item(), lossD["Adv-Fake"].mean().item()
                sum(lossD.values()).mean().backward()
                optimizerD.step()
                
                # Compute Metric
                trainLPIPS, trainSSIM, trainPSNR = computeLPIPS(outputImage, image).item(), utils.computeSSIM(outputImage, image), utils.computePSNR(outputImage, image)
                
                # Show Training Status
                print(f"[Train] [{currentIter}/{opt.numIters}] < LPIPS:{trainLPIPS:.4f} | SSIM:{trainSSIM:.4f} | PSNR:{trainPSNR:.4f} > < Adv-G:{trainAdvG:.4f} | Adv-D-Real:{trainAdvRealD:.4f} | Adv-D-Fake:{trainAdvFakeD:.4f} | Feature:{trainFeature:.4f} | Recon.:{trainRecon:.4f} | Percp.:{trainPercp:.4f} | Style:{trainStyle:.4f} >")

                # Start Testing
                if currentIter % opt.saveIters == 0 :
                    print("< Test Started >")
                    
                    # Create TQDM Instance
                    testBar = tqdm(testDataLoader)
                    
                    # Rest AverageMeter Instance
                    testLPIPS.reset(), testSSIM.reset(), testPSNR.reset()
                    
                    if not opt.noWandb :
                        imageSet = []

                    for data in testBar :
                        # Load Dataset and Assign Device
                        mask, image = data["mask"], data["image"]
                        mask, image = models.assignDevice(opt, mask, image)
                        
                        # Get Final Results
                        maskedImage, outputImage = model(mask, image, mode="inference")
                        
                        if not opt.noWandb :
                            # Add Images
                            subImageSet = imageSaver.visualizeBatch([maskedImage, outputImage, image])
                            if len(imageSet) <= 108 :
                                imageSet.append(subImageSet)
                        
                        # Compute Metric
                        testLPIPS.update(computeLPIPS(outputImage, image).item())
                        testSSIM.update(utils.computeSSIM(outputImage, image))
                        testPSNR.update(utils.computePSNR(outputImage, image))
                        
                        # Show Testing Status
                        testBar.set_description(desc=f"[Valid] [{currentIter}/{opt.numIters}] < LPIPS:{testLPIPS.avg:.4f} | SSIM:{testSSIM.avg:.4f} | PSNR:{testPSNR.avg:.4f} >")
                    
                    if not opt.noWandb :
                        # Upload Images
                        inferenceResult = []
                        for i, subImageSet in enumerate(imageSet) :
                            inferenceResult.append(wandb.Image(subImageSet, caption=f"Batch-{i+1}"))
                    
                    # Save Weights
                    if testLPIPS.avg < bestLPIPS :
                        bestLPIPS = testLPIPS.avg
                        utils.saveNetwork(opt, model, currentIter, latest=False, best=True)
                    utils.saveNetwork(opt, model, currentIter, latest=True, best=False)
                    
                    if not opt.noWandb :
                        # Upload Metrics
                        wandb.log({"Test LPIPS":testLPIPS.avg,
                                   "Test SSIM":testSSIM.avg,
                                   "Test PSNR":testPSNR.avg,
                                   "Inference Result":inferenceResult})

                    print("< Test Ended >")
                
                # Update Learning Rate
                currentIter += 1
                schedulerG.step(), schedulerD.step()

    print("< Training Ended >")
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__" :
    main()