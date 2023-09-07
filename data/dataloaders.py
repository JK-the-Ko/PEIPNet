import importlib
from torch.utils.data import DataLoader


def selectDatasetName(mode) :
    # Select Dataset Name
    if mode == "paris-streetview" :
        return "ParisStreetviewDataset"
    elif mode == "celeba" :
        return "CelebADataset"
    elif mode == "places2" :
        return "Places2Dataset"
    else :
        raise NotImplementedError(f"No such dataset as {mode}")


def getDataLoaders(opt) :
    # Select Dataset Name
    datasetName = selectDatasetName(opt.dataType)
    
    # Import Python Code
    fileName = importlib.import_module(f"data.{datasetName}")
    
    # Create Dataset Instance
    trainDataset = fileName.__dict__[datasetName](opt, forMetrics=False)
    testDataset = fileName.__dict__[datasetName](opt, forMetrics=True)
    
    # Train PyTorch DataLoader Instance
    trainDataLoader = DataLoader(trainDataset, 
                                 batch_size=opt.batchSize, 
                                 shuffle=True, 
                                 drop_last=True, 
                                 num_workers=opt.numWorkers)
    testDataLoader = DataLoader(testDataset, 
                                batch_size=opt.batchSize, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers=opt.numWorkers)
    
    return trainDataLoader, testDataLoader