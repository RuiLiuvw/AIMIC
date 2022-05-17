import os
import re
import csv
import random
import numpy as np
from typing import Any
from pathlib import Path

import torch
from torch import optim

# from device import CUDA_AVAI

TextColors = {
    'logs': '\033[34m',  # 033 is the escape code and 34 is the color code
    'info': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m',
    'bold': '\033[1m',
    'end_color': '\033[0m',
    'light_red': '\033[36m'
}


def colorText(in_text: str) -> str:
    return TextColors['light_red'] + in_text + TextColors['end_color']


def seedSetting(RPMode, Seed=999):
    # Set random seed for reproducibility
    if RPMode:
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", Seed)
        np.random.seed(Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
    return


def pair(Res):
    return Res if isinstance(Res, tuple) else (Res, Res)

    
def getSubdirectories(Dir):
    return [SubDir for SubDir in os.listdir(Dir)
            if os.path.isdir(os.path.join(Dir, SubDir))]


def expFolderCreator(ModelName,SetName,ExpType,DestPath='',Mode=0):
    # Count the number of exsited experiments
    DestPath = DestPath + '/'
    Path(DestPath + ExpType).mkdir(parents=True, exist_ok=True)
    
    ExpList = getSubdirectories(DestPath + ExpType)
    if len(ExpList) == 0:
        ExpCount = 1
    else:
        MaxNum = 0
        for idx in range(len(ExpList)):
            temp = int(re.findall('\d+', ExpList[idx])[0]) + 1
            if MaxNum < temp:
                MaxNum = temp
        ExpCount = MaxNum if Mode == 0 else MaxNum - 1
    
    DestExpPath = DestPath + '%s/exp' % (ExpType) + str(ExpCount) + '_' + ModelName + '_' + SetName + '/'
    Path(DestExpPath).mkdir(parents=True, exist_ok=True)
    Path(DestExpPath + '/trained model').mkdir(parents=True, exist_ok=True)
    
    return DestExpPath, ExpCount


def writeCsv(DestPath, FieldName, FileData, NewFieldNames=[], DictMode=False):
    Flag = 0 if os.path.isfile(DestPath) else 1
    
    with open(DestPath, 'a', encoding='UTF8', newline='') as f:
        if DictMode:
            writer = csv.DictWriter(f, fieldnames=FieldName)
            if Flag == 1:
                writer.writeheader()
            writer.writerows(FileData) # write data
        else:
            writer = csv.writer(f)
            if Flag == 1:
                if NewFieldNames != []:
                    _ = [FieldName.append(FiledName) for FiledName in NewFieldNames]
                writer.writerow(FieldName) # write the header
            writer.writerow(FileData) # write data


def wightFrozen(Model, WeightFrize, PreTrained=2):
    
    ModelDict = Model.state_dict()
    if WeightFrize != 0:
        for idx, (name, param) in enumerate(Model.named_parameters()):
            if WeightFrize == 1: 
                # if 'features' in name:
                Judger = 'fc' not in name
                if PreTrained == 2:
                    Judger = Judger and idx < len(ModelDict.keys()) - 7
                    
                if Judger:
                    param.requires_grad = False
                else:
                    print(name, param.requires_grad)
                    
            elif WeightFrize == 2: 
                param.requires_grad = False
                
            else:
                print(name, param.requires_grad)
    return Model


def loadModelWeight(CUDA_AVAI, Model, WeightFrize, PreTrainedWeight, PreTrained=2, DropLast=False):
    if CUDA_AVAI:
        PretrainedDict = torch.load(PreTrainedWeight)
    else:
        PretrainedDict = torch.load(PreTrainedWeight, map_location=torch.device('cpu'))

    load_PretrainDict = {k: v for k, v in PretrainedDict.items() if Model.state_dict()[k].numel() == v.numel()}
    print(Model.load_state_dict(load_PretrainDict, strict=False))
     
    # ModelDict = Model.state_dict()
    # if PreTrained == 1:
    #     # Get weight if pretrained weight has the same dict
    #     PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and 'classifier' not in k}
    #     # PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and 'classifier' not in k and 'fc' not in k}
    # elif PreTrained == 2:
    #     # Get weight if pretrained weight has the partly same structure but diffetnent keys
    #     # initialize keys and values to keep the original order
    #     OldDictKeys = list(PretrainedDict.keys())
    #     OldValues = list(PretrainedDict.values())
    #     NewDictKeys = list(ModelDict.keys())
    #     NewValues = list(ModelDict.values())
    #
    #     LenFlag = len(PretrainedDict) > len(ModelDict)
    #     MaxLen = max(len(PretrainedDict), len(ModelDict))
    #
    #     idxNew, idxOld = 0, 0
    #     for _ in range(MaxLen):
    #         OldKey = OldDictKeys[idxOld]
    #         OldVal = OldValues[idxOld]
    #         NewKey = NewDictKeys[idxNew]
    #         NewVal = NewValues[idxNew]
    #
    #         if 'classifier' in OldKey or 'classifier' in NewKey:
    #             break
    #
    #         if OldVal.shape == NewVal.shape:
    #             PretrainedDict[NewKey] = PretrainedDict.pop(OldKey)
    #         elif LenFlag:
    #             idxNew -= 1
    #             idxOld -= 1
    #             PretrainedDict.pop(OldKey)
    #         else:
    #             idxOld -= 1
    #         idxNew += 1
    #         idxOld += 1
    #
    #         if DropLast:
    #             if LenFlag and idxOld == len(OldDictKeys) - 2:
    #                 break
    #             elif idxNew == len(NewDictKeys) - 2:
    #                 break
    #
    # ModelDict.update(PretrainedDict)
    # print(Model.load_state_dict(ModelDict, strict=False))

    print('Knowledge transfer with model weight:', PreTrainedWeight)
    
    return Model


def optimizerChoice(NetParam, lr, Choice='Adam', **kwargs: Any):
    
    # OptimChoices = ['Adam', 'AdamW', 'Adamax', 'SparseAdam', 'SGD', 'ASGD',
    #                'RMSprop', 'Rprop', 'LBFGS', 'Adadelta', 'Adagrad']
    
    CallDict = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adamax': optim.Adamax,
    'sparseadam': optim.SparseAdam,
    'sgd': optim.SGD,
    'asgd': optim.ASGD,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'lbfgs': optim.LBFGS,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    }
    
    Optimizer = CallDict[Choice](NetParam, lr=lr, **kwargs)
        
    return Optimizer