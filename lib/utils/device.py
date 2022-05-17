import torch

def device(Printer=[], Mode=1):

    if Mode == 0:
        CUDA_AVAI = False
        DataParallel = False
        Device = torch.device('cpu')
        
    elif Mode == 1:
        CUDA_AVAI = torch.cuda.is_available()
        DataParallel = torch.cuda.device_count() > 1
        Device = torch.device('cuda:0' if CUDA_AVAI else 'cpu')
        if CUDA_AVAI == False:
            Printer.append("No cuda is available")
        
    return CUDA_AVAI, DataParallel, Device
