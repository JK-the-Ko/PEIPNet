import argparse

from utils import utils


def readArguments(train=True) :
    parser = argparse.ArgumentParser()
    parser = addAllArguments(parser, train)
    parser.add_argument("--phase", type=str, default="train")
    opt=parser.parse_args()
    opt.phase="train" if train else "test"
    utils.fixSeed(opt.seed)
    
    return opt


def addAllArguments(parser, train) :
    #[general options]
    parser.add_argument("--name", type=str, default="PEIPNet", help="name of the experiment. It decides where to store samples and models")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpuIds", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpointsDir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
    parser.add_argument("--dataRoot", type=str, default="./dataset", help="path to dataset root")
    parser.add_argument("--dataType", type=str, default="paris-streetview", help="this option indicates which dataset should be loaded")
    parser.add_argument("--threshold", type=float, default=0.6, help="threshold for binarization")
    parser.add_argument("--dx", type=int, default=25, help="x-axis length for mask translation")
    parser.add_argument("--dy", type=int, default=25, help="y-axis length for mask translation")
    parser.add_argument("--angle", type=int, default=15, help="angle for mask rotation")
    parser.add_argument("--minRatio", type=float, default=0.1, help="minimum ratio of mask")
    parser.add_argument("--maxRatio", type=float, default=0.5, help="maximum ratio of mask")
    parser.add_argument("--imageSize", type=int, default=256, help="image size for interpolation")
    parser.add_argument("--noFlip", action="store_true", help="if specified, do not flip the images for data argumentation")
    parser.add_argument("--numWorkers", type=int, default=10, help="num_workers argument for dataloader")

    # For generator
    parser.add_argument("--channelsG", type=int, default=48, help="# of generator filters in first conv layer in generator")
    parser.add_argument("--normType", type=str, default="batch", help="which norm to use in generator")
    parser.add_argument("--inputDim", type=int, default=3, help="dimension of the input data")
    parser.add_argument("--noSpectralNormG", action="store_true", help="if specified, do not use spectral normalization")

    if train :
        parser.add_argument("--noWandb", action="store_true", help="if specified, do not use wandb library")
        parser.add_argument("--initType", type=str, default="normal", help="selects weight initialization type")
        parser.add_argument("--numIters", type=int, default=100000, help="number of iterations to train")
        parser.add_argument("--saveIters", type=int, default=1000, help="number of iterations to save model")
        parser.add_argument("--beta1", type=float, default=0.9, help="momentum term of adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="momentum term of adam")
        parser.add_argument("--lrG", type=float, default=1e-3, help="G learning rate, default=0.0001")
        parser.add_argument("--lrD", type=float, default=4e-3, help="D learning rate, default=0.0004")
        parser.add_argument("--decayRate", type=float, default=1e-2, help="learning rate decay")
        parser.add_argument("--GANMode", type=str, default="hinge", help="{vanilla | ls | hinge}")
        parser.add_argument("--lambdaAdv", type=float, default=1, help="weight for adversarial loss")
        parser.add_argument("--lambdaReconstruction", type=float, default=1e3, help="weight for reconstruction loss")
        parser.add_argument("--lambdaFeature", type=float, default=1e2, help="weight for feature loss")
        parser.add_argument("--lambdaPerceptual", type=float, default=1e2, help="weight for perceptual loss")
        parser.add_argument("--lambdaStyle", type=float, default=1e2, help="weight for style loss")
        
        # For discriminator
        parser.add_argument("--channelsD", type=int, default=64, help="# of discriminator filters in first conv layer in discriminator")
        parser.add_argument("--numLayerD", type=int, default=4, help="# of conv layers in discriminator")
        parser.add_argument("--noSpectralNormD", action="store_true", help="if specified, do not use spectral normalization")
        parser.add_argument("--numD", type=int, default=2, help="# of discriminator for multi-scale feature extraction")

    else:
        parser.add_argument("--saveType", type=str, default="best", help="save type for loading model")

    return parser