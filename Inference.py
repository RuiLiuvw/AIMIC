# -*- coding:utf-8 -*-
"""
Author：LIU Rui
Date：2022/01/12
"""
import sys
import os
import json
import glob
from torchvision import transforms
from lib.model import getModel
from collections import OrderedDict
import torch
import torchvision
from lib.utils import pair
from PIL import Image

from PyQt5.QtCore import pyqtSignal, QObject

import numpy as np
import pandas as pd
from math import ceil,sqrt

import imageio
# import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

class Prediction(QObject):
    Pred_result = pyqtSignal(list)

    def __init__(self):
        super(Prediction, self).__init__()

    def doPrediction(self, device_infer, modelname_infer,weightpath_infer,imgpath_infer,classinfo_infer):
        self.imgpath_infer = imgpath_infer
        self.img_paths = glob.glob(self.imgpath_infer + '/*')
        if "inception" in modelname_infer:
            ImgHeight, ImgWidth = pair(299)
        else:
            ImgHeight, ImgWidth = pair(224)
        data_transform = transforms.Compose(
            [
                transforms.Resize((ImgHeight, ImgWidth)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        try:
            class_file = open(classinfo_infer,'r')
            class_indict = json.load(class_file)
            self.clsInfo = ''
        except:
            self.clsInfo = 'Please make sure the content and format of the class information file is in an appropriate way !'
        else:
            num_classes = len(class_indict)

            model = getModel(modelname_infer, num_classes)

            model.to(device_infer)

            state_dict = torch.load(weightpath_infer, map_location=torch.device(device_infer))
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v

            # model.load_state_dict(new_state_dict)
            try:
                model.load_state_dict(new_state_dict)
                self.matchInfo = ''
            except:
                self.matchInfo = 'Please make sure you have selected the right model and loaded the corresponding weight!'
            else:
                self.Filelist = []
                self.PredictedClasslist = []
                self.Probabilitylist = []
                for img_path in self.img_paths:
                    img = Image.open(img_path)
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    model.eval()
                    with torch.no_grad():
                        output0 = torch.squeeze(model(img.to(device_infer))).cpu()
                        output0 = torch.softmax(output0, dim=0)
                        labels_pred = torch.argmax(output0).numpy()
                        predictedclass = class_indict[str(labels_pred)]
                        probability = output0[labels_pred].numpy()
                        print("File: {}   Predicted Class: {}   Probability: {:.3}".format(img_path,predictedclass,probability))
                        self.Pred_result.emit([img_path,class_indict[str(labels_pred)],output0[labels_pred].numpy()])
                        self.Filelist.append(img_path)
                        self.PredictedClasslist.append(predictedclass)
                        self.Probabilitylist.append(probability)

    def writeInferResult(self, outputFile_infer):
        OutputExcel = {'Files': self.Filelist, 'PredClass': self.PredictedClasslist, 'Probability': self.Probabilitylist}
        Output = pd.DataFrame(OutputExcel)
        OutputFieldNames = ['Files', 'PredClass', 'Probability']
        Output.to_csv(outputFile_infer, columns=OutputFieldNames,encoding='utf-8')

    # Visulize the Inference results
    def visualizeInfer(self,numimgs_fig):
        self.numimgs_fig = numimgs_fig  # number of images in each figure
        self.img_names = os.listdir(self.imgpath_infer)
        j = 0
        global Pos
        Pos = 0
        for i, img_path in enumerate(self.img_paths):
            Img = imageio.imread(img_path)
            Pos += 1
            if i % self.numimgs_fig == 0:
                plt.figure("Inference" + '_{}'.format(j))
                plt.suptitle('The Inference Results' + ' {}'.format(j), size=12)
                j += 1
                Pos = 0
            column = ceil(sqrt(self.numimgs_fig))
            row = ceil(self.numimgs_fig/column)
            plt.subplot(row, column, Pos + 1)
            plt.imshow(Img)
            plt.axis('off')
            plt.title('Img: {}\nPred: {}\nProb: {:.3}'.format(self.img_names[i],self.PredictedClasslist[i],self.Probabilitylist[i]), loc='left', size=8)

            plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    weightpath_infer = r'H:\Medical Image Classification\Experiment Results_after data balance_00001\1_PBC_dataset_normal_DIB_data_split\mobilenetv2\weights/1_PBC_dataset_normal_DIB_data_split_mobilenetv2_run0.pth'
    imgpath_infer = r'F:\AAAAA Medical Image Classification\prediction\image'
    classinfo_infer = r'F:\AAAAA Medical Image Classification\prediction/class_indices.json'
    outputPath_infer = r'F:\AAAAA Medical Image Classification\AIMedicalWorld\Inference output\Inferenceresult.csv'
    prediction = Prediction()
    prediction.doPrediction(device_infer='cuda:0',modelname_infer='mobilenet_v2',weightpath_infer=weightpath_infer,imgpath_infer=imgpath_infer,
                            classinfo_infer=classinfo_infer)
    prediction.visualizeInfer(8)
    prediction.writeInferResult(outputFile_infer=outputFile_infer)