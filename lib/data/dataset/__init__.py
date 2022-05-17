import os
from glob import glob

import numpy as np
from PIL import Image
# import imgaug.augmenters as iaa
from sklearn.model_selection import KFold
import sklearn.utils._typedefs

from ...utils import pair
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

    
class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        label = np.longlong(self.labels[index])  # Convert the data type of label to long int type
        img = Image.open(img_path)

        # Change Image channels
        if img.mode == "RGBA":
            r, g, b, _ = img.split()
            img = Image.merge("RGB", (r, g, b))
        if img.mode != "RGB":
            img = img.convert("RGB")
        # img = img.convert("RGB")
        img_data = self.transforms(img)
        return img_data, label

    def __len__(self):
        return len(self.img_paths)
    
    
class Transforms(transforms.Compose):
    def __init__(self, ResizeRes = 224):
        # Set up transforms
        ImgHeight, ImgWidth = pair(ResizeRes)
        self.TF = {
            "train": transforms.Compose([
                # transforms.RandomRotation(10),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((ImgHeight, ImgWidth)),
                transforms.ToTensor(), # divided by 255. This is how it is forces the network to be between 0 and 1.
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                transforms.Normalize((0.5, 0.5, 0.5), [0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(ResizeRes),
                transforms.Resize((ImgHeight, ImgWidth)),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                transforms.Normalize((0.5, 0.5, 0.5), [0.5, 0.5, 0.5])
            ]),
        }
    
    def retrunItem(self):
        return self.TF


def getImgPath(DatasetPath, NumSplit, Mode=1, Shuffle=True):
    # Put images into train set or test set
    if Mode == 1:
        '''
        root/split1/dog_1.png
        root/split1/dog_2.png
        root/split2/cat_1.png
        root/split2/cat_2.png
        '''
        Trainset, Testset = [], []
        for i in range(1, NumSplit + 1):
            Testset.append(glob(DatasetPath + '/' + 'split{}'.format(i) + '/*'))
            
            TrainImgs = []
            for j in range(1, NumSplit + 1):
                if j != i:
                    TrainImgs.extend(glob(DatasetPath + '/' + 'split{}'.format(j) + '/*'))
            Trainset.append(TrainImgs)
                
    elif Mode == 2:
        '''
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
        '''
        Trainset, Testset = [[] for i in range(NumSplit)], [[] for j in range(NumSplit)]
        ClassNames = os.listdir(DatasetPath)
        Kf = KFold(n_splits=NumSplit, shuffle=Shuffle)
        
        for ClassName in ClassNames:
            ImagePath = glob(DatasetPath + '/' + ClassName + '/*')
            IndexList = range(0, len(ImagePath))

            Kf.get_n_splits(IndexList)
            
            for idx, (TrainIndexes, TestIdexes) in enumerate(Kf.split(IndexList)):
                [Trainset[idx].append(ImagePath[i]) for i in TrainIndexes]
                [Testset[idx].append(ImagePath[j]) for j in TestIdexes]
            
    return Trainset, Testset


def myDataLoader(DatasetPath, LabelList, BatchSize, ResizeRes, SetType="train", NumWorker=0, 
                 Shuffle=True, DropLast=True):
    # Create labels
    Labels = []
    for ImgPath in DatasetPath:
        for j, spec in enumerate(LabelList):
            if spec in ImgPath:
                Labels.append(j)
                break

    MyDataset = Mydataset(DatasetPath, Labels, Transforms(ResizeRes).TF[SetType])

    MyDataLoader = DataLoader(MyDataset, batch_size=BatchSize, shuffle=Shuffle, 
                              drop_last=DropLast, num_workers=NumWorker)
    
    return MyDataLoader