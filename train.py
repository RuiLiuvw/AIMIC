## Import module
 # path manager
from pathlib import Path
 # data processing
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
 # torch module
import torch
import torchvision.models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
 # my module
from lib.utils.cnnUtils import CnnTrain
from lib.model import getModel
from lib.data import getImgPath, myDataLoader
from lib.utils import seedSetting, expFolderCreator, writeCsv, loadModelWeight, optimizerChoice
# pyqt module
from PyQt5.QtCore import pyqtSignal, QObject

class Train(QObject):
    # Create signal
    CurrentEpoch = pyqtSignal(int)
    EachEpoch = pyqtSignal(int)

    CurrentSplit = pyqtSignal(int)
    NumTrainValData = pyqtSignal(list)
    MeticsList = pyqtSignal(list)
    EpochLog = pyqtSignal(list)
    TimeCost = pyqtSignal(float)
    def __init__(self) -> None:
        super(Train, self).__init__()
        
        self.CnnTrain = CnnTrain()
        self.CnnTrain.CurrentIter.connect(self.setCurrentIter)

    def doRun(self, 
                CUDA_AVAI, DataParallel, Device,
                ModelName, MyOptimizer,
                Epochs, BatchSize, ResizeRes, StopStation, NumSplit,
                LrRate,
                DatasetPath, PreTrainedWeight, DestPath,
                SetName, Shuffle, DropLast, NumWorker,
                ClassNames,
                SampleMode, LrDecay=False, PreTrained=0,
                WeightFrize=0, LossMode=0, RepeatedValidNum=0,
                Seed = 999
                ):
        
        Tick0 = time.perf_counter()
        
        seedSetting(RPMode=1, Seed=Seed)
        
        # Path init
        DestExpPath, ExpCount = expFolderCreator(ModelName,SetName,ExpType='training_log',DestPath=DestPath)
        
        ExpLogPath = DestPath + '/training_log/log.csv' # No need to change
        InputLogPath = DestExpPath + 'input_param_' + ModelName + '_' + SetName + '.csv'
        ModelSavePath = DestExpPath + 'trained model'  # Save weight
        TrainMetricsPath = DestExpPath + '/' + 'metrics_' + ModelName + '_' + SetName + '.csv'  # Save metrics
        TrainRecordPathSingle = DestExpPath + '/training record of the metrics/'
        TrainRecordPath = TrainRecordPathSingle+ 'record_' + ModelName + '_' + SetName + '_'  # Save indicators during training
        
        # save input params
        self.Logfield = ['exp', 'date', 'Model', 'Optimizer', 'Dataset',
                    'LrDecay', 'SampleMode', 'PreTrained', 'WeightFrize',
                    'NumberofSplit', 'RepeatedValidNum', 'LrRate',
                    'Epochs', 'BatchSize', 'StopStation', 'ResizeResolution'] # Define header
        self.LogInfo = [ExpCount, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ModelName, MyOptimizer, SetName, 
                LrDecay, SampleMode, PreTrained, WeightFrize,
                NumSplit, RepeatedValidNum, LrRate,
                Epochs, BatchSize, StopStation, ResizeRes]
        writeCsv(InputLogPath, self.Logfield, self.LogInfo)
 
        Trainset, Testset = getImgPath(DatasetPath, NumSplit, Mode=2)
       
        NumClasses = len(ClassNames)
        
        for i in range(NumSplit):
            self.CurrentSplit.emit(0) # init each val bar
            self.CurrentEpoch.emit(0) # init each epoch bar
            
            Model = getModel(
                             ModelName,
                             NumClasses,
                             # InitWeights=True,
                             # ResizeRes=ResizeRes
                             )
            if PreTrained:
                Model = loadModelWeight(CUDA_AVAI, Model, WeightFrize, PreTrainedWeight, PreTrained)

            if DataParallel:
                Model = torch.nn.DataParallel(Model)
                # Model = DDP(Model) # Optimizer, grad, and backward should also be changed
            Model.to(Device)

            Optimizer = optimizerChoice(Model.parameters(), lr=LrRate, Choice=MyOptimizer)
            Scheduler = StepLR(Optimizer, step_size=20, gamma=0.9)
            # Scheduler = ReduceLROnPlateau(Optimizer, mode='min', factor=0.05, patience=5, min_lr=0.0001)
            # Scheduler = MultiStepLR(Optimizer, milestones=[10, 30, 60, 100], gamma=0.05)
            
            TrainImgs, TestImgs = Trainset[i], Testset[i]
            # TrainImgs, TestImgs = Trainset, Testset  # 修改

            TrainDL = myDataLoader(TrainImgs, ClassNames, BatchSize, ResizeRes, "train", NumWorker, Shuffle, DropLast)
            TestDL = myDataLoader(TestImgs, ClassNames, BatchSize, ResizeRes, "test", NumWorker, Shuffle, DropLast)
            
            self.LenTrainDL = len(TrainDL)
            self.LenTestDL = len(TestDL)
            self.NumTrainValData.emit([len(TrainImgs), len(TestImgs)])
            
            Temp = []
            [Temp.append([]) for _ in range(8)]
            self.TrainLossList, self.TrainAccuList, self.TestLossList, self.TestAccuList, \
                self.AvgRecallList, self.AvgPrecisionList, self.AvgF1ScoreList, self.LrRateList = tuple(Temp)

            # Start tranining
            Temp = 0
            IndicatorType = 'accuracy'
            BestStopIndicator = 10000 if IndicatorType == 'loss' else 0 # Best metric indicator
            self.CurrentSplit.emit(i + 1)
            
            for Epoch in range(Epochs):
                self.CurrentEpoch.emit(Epoch + 1)

                Tick1 = time.perf_counter()

                self.CnnTrain.doRun(CUDA_AVAI, Device, NumClasses, Optimizer, Model, ModelName, TrainDL, TestDL, LossMode)
                
                AvgRecall = np.mean(self.CnnTrain.Recall)
                AvgPrecision = np.mean(self.CnnTrain.Precision)
                AvgF1Score = np.mean(self.CnnTrain.F1Score)
                
                self.MeticsList.emit([self.CnnTrain.TrainLoss, self.CnnTrain.TrainAcc, self.CnnTrain.TestLoss, self.CnnTrain.TestAcc, AvgRecall, AvgPrecision, AvgF1Score])
                
                # Save weight, if test accuracy remains unchanged in continous N times, we stop the training progress
                # StopIndicator = self.CnnTrain.TestLoss
                StopIndicator = self.CnnTrain.TestLoss if IndicatorType == 'loss' else self.CnnTrain.TestAcc
                if (StopIndicator < BestStopIndicator and IndicatorType == 'loss') or \
                (StopIndicator > BestStopIndicator and IndicatorType != 'loss'):
                        BestStopIndicator = StopIndicator
                        # BestModelWT = copy.deepcopy(Model.state_dict())
                        torch.save(Model.state_dict(), ModelSavePath + '/' + ModelName + '_' + SetName + '_run{}.pth'.format(i + 1))
                        Temp = 0
                else:
                    Temp += 1
                    if Temp == StopStation:
                        break
                    
                self.TrainLossList.append(self.CnnTrain.TrainLoss)
                self.TrainAccuList.append(self.CnnTrain.TrainAcc)
                self.TestLossList.append(self.CnnTrain.TestLoss)
                self.TestAccuList.append(self.CnnTrain.TestAcc)
                self.AvgRecallList.append(AvgRecall)
                self.AvgPrecisionList.append(AvgPrecision)
                self.AvgF1ScoreList.append(AvgF1Score)
                
                if LrDecay:
                    Scheduler.step()
                    self.LrRateList.append(Scheduler.get_last_lr())
                else:
                    self.LrRateList.append(LrRate)
                            
                self.EpochLog.emit([time.perf_counter() - Tick1, Scheduler.get_last_lr()])
                # self.CurrentEpoch.emit(Epoch + 1)
                
            self.Epoch = Epoch
            
            ## Writing results
            Path(TrainRecordPathSingle).mkdir(parents=True, exist_ok=True)
            self.writeAvgMetrics(TrainRecordPath, BatchSize, StopStation, ResizeRes, i)
            self.writeBestMetrics(TrainMetricsPath, IndicatorType, i)
            
            if i == NumSplit - RepeatedValidNum - 1 or SampleMode == 2:
                self.writeAvgBestMetrics(TrainMetricsPath)
                break

        TimeCost = time.perf_counter() - Tick0
        self.TimeCost.emit(TimeCost)
        
        self.writeLogFile(ExpLogPath, TimeCost)
        
    def writeAvgMetrics(self, TrainRecordPath, BatchSize, StopStation, ResizeRes, i):
        # Write avrage training metrics record
        OutputExcel = {'LrRate': self.LrRateList, 'BatchSize': BatchSize,
                    'StopStation': StopStation, 'ResizeResolution': ResizeRes, 
                    'TrainLoss': self.TrainLossList, 'TrainAccuracy': self.TrainAccuList, 'ValLoss': self.TestLossList,
                        'ValAccuracy': self.TestAccuList, 'AvgRecall': self.AvgRecallList,
                        'AvgPrecision': self.AvgPrecisionList, 'AvgF1Score': self.AvgF1ScoreList}
        Output = pd.DataFrame(OutputExcel)

        OutputFieldNames = ['LrRate', 'BatchSize', 'StopStation', 'ResizeResolution', 
                    'TrainLoss', 'TrainAccuracy', 'ValLoss', 'ValAccuracy',
                    'AvgRecall', 'AvgPrecision', 'AvgF1Score']
        Output.to_csv(TrainRecordPath + 'run_{}.csv'.format(i + 1), columns=OutputFieldNames, encoding='utf-8')
    
    def writeBestMetrics(self, TrainMetricsPath, IndicatorType, i):
        # export best results
        if IndicatorType == 'loss':
            idx = self.TestLossList.index(min(self.TestLossList))
        else:
            idx = self.TestAccuList.index(max(self.TestAccuList))
        
        BestAccuracy = self.TestAccuList[idx]
        BestRecall = self.AvgRecallList[idx]
        BestPrecision = self.AvgPrecisionList[idx]
        BestF1Score = self.AvgF1ScoreList[idx]
        
        DfBest = ['run_{}'.format(i + 1), BestAccuracy, BestRecall, BestPrecision, BestF1Score]
        self.MetricsFieldNames = ['K-Fold', 'BestAccuracy', 'BestRecall', 'BestPrecision', 'BestF1Score']    
        writeCsv(TrainMetricsPath, self.MetricsFieldNames, DfBest)
        
    def writeAvgBestMetrics(self, TrainMetricsPath):
        # get mean values of metrics in k-fold validation
        MetricsReader = csv.reader(open(TrainMetricsPath, 'r'))
        BestMetrics = []
        for Row in MetricsReader:
            BestMetrics.append(Row)
        BestMetrics.pop(0) # remove the header/title in the first row
        RowNum = len(BestMetrics)
        ColNum = len(BestMetrics[0])
        
        Values = np.zeros((ColNum - 1, 1))
        for i in range(ColNum - 1):
            for j in range(RowNum):
                Values[i] += float(BestMetrics[j][i + 1])
        Values /= RowNum

        self.AvgBestMetric = ['Average']
        self.AvgBestMetric.extend(list(Values.flatten()))
        writeCsv(TrainMetricsPath, self.MetricsFieldNames, self.AvgBestMetric)

    def writeLogFile(self, ExpLogPath, TimeCost):
        # Write input and output param in log file
        self.LogInfo.extend([TimeCost, self.Epoch])
        self.LogInfo.extend(self.AvgBestMetric[1:])
        self.LogInfo[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        NewFieldNames = ['TimeCost', 'StopEpoch', 'Accuracy', 'Recall', 'Precision', 'F1Score']
        writeCsv(ExpLogPath, self.Logfield, self.LogInfo, NewFieldNames)

    def setCurrentIter(self, CurrentIter):
        self.EachEpoch.emit(round((CurrentIter + 1) / self.LenTrainDL * 100))

## Test
if __name__ == "__main__":
    import os
    from lib.utils.device import device
    
    CUDA_AVAI, DataParallel, Device = device(Mode=1)
    ModelName = 'alexnet'
    SetName = 'MyDataset'
    MyOptimizer = 'adam'
    Epochs = 2
    BatchSize = 4
    ResizeRes = 224
    StopStation = 50
    NumSplit = 3
    LrRate = 0.00001
    Seed = 999
    DatasetPath = './dataset/sample_by_name'
    TrainedModelPath = ''
    DestPath = './exp/'
    Shuffle = True
    DropLast = True
    NumWorker = 0
    ClassNames = os.listdir(DatasetPath)
    SampleMode = 0
   # BNorm = True
    LrDecay = False
    PreTrained = 0
    WeightFrize = 0
    LossMode = 0
    
    
    ClsTrain = Train()
    ClsTrain.doRun(CUDA_AVAI, DataParallel, Device,
              ModelName, MyOptimizer,
              Epochs, BatchSize, ResizeRes, StopStation, NumSplit,
              LrRate, Seed,
              DatasetPath, TrainedModelPath, DestPath,
              SetName, Shuffle, DropLast, NumWorker,
              ClassNames,
              SampleMode, LrDecay, PreTrained,
              WeightFrize, LossMode)