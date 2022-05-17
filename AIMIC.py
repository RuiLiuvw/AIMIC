import os
from glob import glob
import time

import matplotlib.pyplot as plt

# from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication
# from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import sklearn.utils._typedefs

from train import Train
from Inference import Prediction
from lib.data import drawSamplebyClass
from lib.utils.device import device
from lib.utils.guiUtils import Worker
from lib.utils.defaultValues import (ModelList, OptimList,
                                     DefaultValue, DefaultPath, DefaultName,DefaultPath_infer,DefaultValue_infer)


# load gui desginer window from .ui file
UI, _ = loadUiType('AIMedicalWorld.ui', resource_suffix='')


class MainApp(QMainWindow, UI):
    def __init__(self):
        super(MainApp, self).__init__()
        # Init parameters
        self.Flag1 = 0
        self.Flag2 = 0
        self.Flag3 = 0
        self.Flag4 = 0
        self.Flag5 = 0
        self.Flag1_infer = 0
        self.Flag2_infer = 0
        self.Flag3_infer = 0
        self.Flag4_infer = 0

        self.cwd = os.getcwd()  # Get the current path

        self.setupUi(self) # Initialize user interface
        self.setWindow()
        self.setWidgets()
        self.setDefaultValues()
        self.initOutput()

        self.TrainProcess = Train()
        self.InferenceProcess = Prediction()

        self.signalSlots()

        # Fix window size
        self.setFixedSize(self.width(), self.height())

    ## Set window properties
    def setWindow(self):
        Statusbar = self.statusBar()
        Statusbar.showMessage('Ready')

    def setWidgets(self):
        pass

    ## Set default values
    def setDefaultValues(self):
        # Input params
        [self.Epochs, self.BatchSize, self.ResizeRes, self.StopStation,
            self.NumSplit, self.LrRate,self.NumWorker] = DefaultValue
        self.Model, self.Optimizer, self.SetName = DefaultName
        self.CUDA_AVAI, self.DataParallel, self.Device = device(Mode=1)
        self.DatasetPath, self.WeightPath, self.DestPath = DefaultPath

        self.LiditLoadDataset.setText(self.DatasetPath)
        self.LiditLoadWeight.setText(self.WeightPath)
        self.LiditSetOutputPath.setText(self.DestPath)

        self.SboxEpochs.setValue(self.Epochs)
        self.SboxStopStation.setValue(self.StopStation)
        self.SboxBatchSize.setValue(self.BatchSize)
        self.SboxResizeRes.setValue(self.ResizeRes)
        self.SboxNumSplit.setValue(self.NumSplit)
        self.dSBoxLrRate.setValue(self.LrRate)
        # self.LiditLrRate.setText(self.LrRate)
        self.LiditSetName.setText(self.SetName)
        # self.LiditSeed.setText(str(self.Seed))
        # self.SboxNumWorker.setValue(self.NumWorker)

        # Input params for inference
        self.Model_infer = DefaultName[0]
        self.CUDA_AVAI_infer, self.DataParallel_infer, self.Device_infer = device(Mode=1)
        [self.numimgs_fig_infer] = DefaultValue_infer
        self.ImagePath_infer,self.WeightPath_infer = DefaultPath_infer

        self.LiditLoadWeight_infer.setText(self.WeightPath_infer)
        self.LiditLoadImage_infer.setText(self.ImagePath_infer)
        self.LiditImageNumber_infer.setText('0')
        self.Liditnumimgs_fig_infer.setText(str(self.numimgs_fig_infer))

    def initOutput(self):
        # Output Params
        self.Loss = []
        self.Accuracy = []
        self.TestLoss = []
        self.TestAcc = []

    ## Connect signals and slots
    def signalSlots(self):
        # self.BtnStartTrain.clicked.connect(self.training)
        # Set drop down list
        self.CboxDevice.currentTextChanged.connect(self.pickDevice)
        self.CboxModel.currentTextChanged.connect(self.pickModel)
        self.CboxModel2.currentTextChanged.connect(self.pickModel2)
        self.CboxOptim.currentTextChanged.connect(self.pickOptim)
        self.CboxOptim2.currentTextChanged.connect(self.pickOptim2)

        self.CboxDevice_infer.currentTextChanged.connect(self.pickDevice_infer)
        self.CboxModel_infer.currentTextChanged.connect(self.pickModel_infer)
        self.CboxModel2_infer.currentTextChanged.connect(self.pickModel2_infer)

        # Set input parameters
        self.SboxEpochs.valueChanged.connect(self.getEpochs)
        self.SboxStopStation.valueChanged.connect(self.getStopStation)
        self.SboxBatchSize.valueChanged.connect(self.getBatchSize)
        self.SboxResizeRes.valueChanged.connect(self.getResizeRes)
        self.SboxNumSplit.valueChanged.connect(self.getNumSplit)
        self.dSBoxLrRate.valueChanged.connect(self.getLrRatebyLidit)
        # self.LiditLrRate.textChanged.connect(self.getLrRatebyLidit)
        # self.DialLrRate.valueChanged.connect(self.getLrRatebyDial)
        self.LiditSetName.textChanged.connect(self.getSetName)
        # self.LiditSeed.textChanged.connect(self.getSeed)
        # self.SboxNumWorker.valueChanged.connect(self.getNumWorker)
        # Load path
        self.BtnLoadData.clicked.connect(self.loadDataset)
        self.BtnLoadWeight.clicked.connect(self.loadModel)
        self.BtnOutputPath.clicked.connect(self.loadOutPath)

        # Function link
        self.BtnViewInfo.clicked.connect(self.viewInfo)
        self.BtnViewSampleImgs.clicked.connect(self.viewSampleImgs)
        self.BtnViTrain.clicked.connect(self.setPlotFlag)
        # Start and stop train
        self.BtnStartTrain.clicked.connect(self.startTrain)
        self.BtnStopTrain.clicked.connect(self.stopTrain)
        # set signal slot
        self.TrainProcess.CurrentEpoch.connect(self.setCurrentEpoch)
        self.TrainProcess.EachEpoch.connect(self.setEachEpoch)
        self.TrainProcess.CurrentSplit.connect(self.printCurrentSplit)
        self.TrainProcess.NumTrainValData.connect(self.printNumTrainValData)
        self.TrainProcess.MeticsList.connect(self.printMeticsList)
        self.TrainProcess.EpochLog.connect(self.setTrainProcess)
        self.TrainProcess.TimeCost.connect(self.printTimeCost)

        # set input parameters for inference
        self.Liditnumimgs_fig_infer.textChanged.connect(self.getnumimgs_fig_infer)

        # Load path for inference
        self.BtnLoadImage_infer.clicked.connect(self.loadImage_infer)
        self.BtnLoadWeight_infer.clicked.connect(self.loadModel_infer)
        self.BtnLoadClassinfo_infer.clicked.connect(self.loadClassinfo_infer)
        self.BtnLoadOutputPath_infer.clicked.connect(self.loadOutPath_infer)

        # Function link for inference
        self.BtnViInfer.clicked.connect(self.visual_infer)

        # Start inference
        self.BtnStartInfer.clicked.connect(self.startInfer)

        # Set signal slot for inference
        self.InferenceProcess.Pred_result.connect(self.printInferresult)

    ## Set drop down list    
    def pickDevice(self, Device):
        Mode = 0 if Device == 'CPU' else 1
        self.CUDA_AVAI, self.DataParallel, self.Device = device(self.TeditTrainLog, Mode)
        self.TeditTrainLog.append("Using {} device".format(self.Device))

    def pickDevice_infer(self, Device_infer):
        Mode = 0 if Device_infer == 'CPU' else 1
        self.CUDA_AVAI_infer, self.DataParallel_infer, self.Device_infer = device(self.TeditInferLog, Mode)
        self.TeditInferLog.append("Using {} device".format(self.Device_infer))

    def pickModel(self, Model):
        self.Model = Model
        if Model in ModelList:
            self.CboxModel2.clear()
            self.CboxModel2.addItems(ModelList[Model])

    def pickModel2(self, Model):
        self.Model = Model

    def pickModel_infer(self, Model_infer):
        self.Model_infer = Model_infer
        if Model_infer in ModelList:
            self.CboxModel2_infer.clear()
            self.CboxModel2_infer.addItems(ModelList[Model_infer])

    def pickModel2_infer(self, Model_infer):
        self.Model_infer = Model_infer

    def pickOptim(self, Optim):
        self.Optimizer = Optim
        if Optim in OptimList:
            self.CboxOptim2.clear()
            self.CboxOptim2.addItems(OptimList[Optim])

    def pickOptim2(self, Optim):
        self.Optimizer = Optim

    ## Set input parameters
    def getEpochs(self, Epochs):
        self.Epochs = int(Epochs)

    def getBatchSize(self, BatchSize):
        self.BatchSize = int(BatchSize)

    def getResizeRes(self, ResizeRes):
        self.ResizeRes= int(ResizeRes)

    def getStopStation(self, StopStation):
        self.StopStation = int(StopStation)

    def getNumSplit(self, NumSplit):
        self.NumSplit = int(NumSplit)

    def getLrRatebyLidit(self, LrRate):
        self.LrRate = float(LrRate)

    # def getLrRatebyDial(self, LrRate):
    #     self.LrRate = float(LrRate / 100000.0)
    #     self.LiditLrRate.setText(str(self.LrRate))

    def getSetName(self, SetName):
        self.SetName = str(SetName)

    # def getSeed(self, Seed):
    #     self.Seed = int(Seed)

    # def getNumWorker(self, NumWorker):
    #     self.NumWorker = int(NumWorker)


    def getnumimgs_fig_infer(self,numimgs_fig_infer):
        self.numimgs_fig_infer = int(numimgs_fig_infer)

    ## Load path
    def loadDataset(self):
        DatasetPath = str(QFileDialog.getExistingDirectory(None, "Select dataset directory"))
        self.DatasetPath = DatasetPath
        self.LiditLoadDataset.setText(DatasetPath)
        self.Flag4 = 1

    def loadImage_infer(self):
        ImagePath_infer = str(QFileDialog.getExistingDirectory(None, "Select image directory"))
        self.ImagePath_infer = ImagePath_infer
        self.LiditLoadImage_infer.setText(self.ImagePath_infer)
        self.Number = len(glob(self.ImagePath_infer+'/*'))
        self.LiditImageNumber_infer.setText(str(self.Number))
        self.Flag2_infer = 1


    def loadModel(self):
        if self.CKboxPretrainMode.isChecked():
            WeightPath, filetype = QFileDialog.getOpenFileName(None, "Select pretrained weight")
            self.WeightPath = WeightPath
            self.LiditLoadWeight.setText(WeightPath)
            self.Flag5 = 1
        else:
            self.TeditTrainLog.append('Please activate transfer learning mode first !')

    def loadModel_infer(self):
        WeightPath_infer, filetype = QFileDialog.getOpenFileName(None, "Select trained model")
        self.WeightPath_infer = WeightPath_infer
        self.LiditLoadWeight_infer.setText(WeightPath_infer)
        self.Flag3_infer = 1

    def loadClassinfo_infer(self):
        Classinfo_infer, filetype = QFileDialog.getOpenFileName(None, "Select class information file")
        self.Classinfo_infer = Classinfo_infer
        self.LiditLoadClassinfo_infer.setText(Classinfo_infer)
        self.Flag4_infer = 1

    def loadOutPath(self):
        DestPath = str(QFileDialog.getExistingDirectory(None, "Select output directory"))
        self.DestPath = DestPath
        self.LiditSetOutputPath.setText(DestPath)

    def loadOutPath_infer(self):
        if self.Flag1_infer == 1:
            outputFile_infer, filetype = QFileDialog.getSaveFileName(None, "Save the results", self.cwd,'*.csv')
            if outputFile_infer == '':
                self.TeditInferLog.append('Cancel selection!')
                return   # if no return, the pragram will get stuck
            self.outputFile_infer = outputFile_infer
            self.LiditSetOutputPath_infer.setText(outputFile_infer)
            self.InferenceProcess.writeInferResult(outputFile_infer=self.outputFile_infer)
            self.TeditInferLog.append('The inferrence result has been saved as :\n{}'.format(self.outputFile_infer))
        elif self.Flag1_infer == 0:
            self.TeditInferLog.append('Please do the inference first !!!')

    ## Add function
     # View dataset information
    def viewInfo(self):
        if self.Flag4 == 0 or len(self.DatasetPath) == 0:
            self.TeditDatasetInfo.append('No datasets have been uploaded yet !')
        elif self.Flag4 == 1 and len(self.DatasetPath) != 0:
            self.TeditDatasetInfo.clear()

            ClassNames = os.listdir(self.DatasetPath)
            NumClass = len(ClassNames)
            TotalNumImgs = len(glob(self.DatasetPath + '/*' + '/*'))
            self.TeditDatasetInfo.append('Dataset has %d class and %d images'
                                         %(NumClass, TotalNumImgs))

            # NumImgEachClass = [[]] * NumClass
            for idx, ClassName in enumerate(ClassNames):
                ImagePath = glob(self.DatasetPath + '/' + ClassName + '/*')
                # NumImgEachClass[idx] = len(ImagePath)
                NumImgEachClass0 = len(ImagePath)
                self.TeditDatasetInfo.append('Class %s has %d images' %(ClassName, NumImgEachClass0))

     # View sample data images
    def viewSampleImgs(self):
        if self.Flag4 == 1 and len(self.DatasetPath) != 0:
            drawSamplebyClass(self.DatasetPath, os.listdir(self.DatasetPath), Mode=2)
        elif self.Flag4 == 0 or len(self.DatasetPath) == 0:
            self.TeditDatasetInfo.append('No datasets have been uploaded yet !')

    # Plot metrics
    def initAxis(self):
        # Set axis labels
        self.Ax[0].set_title('Loss', fontsize=14)
        self.Ax[0].set_xlabel('Epochs', fontsize=14)
        self.Ax[0].set_ylabel('Values', fontsize=14)
        self.Ax[0].tick_params(labelsize=14)
        # self.Ax[0].tick_params(axis='y', rotation=90)

        self.Ax[1].set_title('Accuracy', fontsize=14)
        self.Ax[1].set_xlabel('Epochs', fontsize=14)
        self.Ax[1].set_ylabel('Values', fontsize=14)
        self.Ax[1].tick_params(labelsize=14)
        # self.Ax[1].tick_params(axis='y', rotation=90)

    def setFlag2(self, _):
        self.Flag2 = 0
        self.Flag3 = 0

    def setPlotFlag(self):

        if self.Flag1 == 1:
            if self.Flag2 == 0:
                # self.Flag2 = 1
                self.Fig, self.Ax = plt.subplots(ncols=2, figsize=(12,5))
                self.Fig.suptitle("The Metrics of Training", fontsize=16)
                self.initAxis()
                self.Ax[0].axis([0, self.Epochs, 0, 1])
                self.Ax[1].axis([0, self.Epochs, 0, 1])

                plt.show()
                self.Flag2 = 1
                self.Fig.canvas.mpl_connect('close_event', self.setFlag2)

        else:
            self.TeditTrainLog.append('Please start training first !')

    def plotMetrics(self):
        if self.Flag2 == 1:

            x = range(1, self.CurrentEpoch + 1) # one epoch delay to get signal

            self.Ax[0].plot(x, self.Loss, 'r', marker="o", label='Train Loss')
            self.Ax[0].plot(x, self.TestLoss, 'b', marker="o", label='Test Loss')
            self.Ax[0].autoscale()

            self.Ax[1].plot(x, self.Accuracy, 'g', marker="o", label='Train Accuracy')
            self.Ax[1].plot(x, self.TestAcc, 'y', marker="o", label='Test Accuracy')
            self.Ax[1].autoscale()

            if self.Flag3 == 0:
                self.Ax[0].legend()
                self.Ax[1].legend()

            plt.draw()

            self.Flag3 = 1

    ## Train and stop
    def doTrain(self):
        if self.Flag4 == 1 and len(self.DatasetPath) != 0:
            if self.CKboxPretrainMode.isChecked() and self.Flag5 == 0:
                self.TeditTrainLog.append('Please load pretrained model for transfer learning !')
            else:
                self.TeditTrainLog.append('=====================Start training !=====================')
                self.BtnStartTrain.setText('Running')
                self.Flag1 = 1

                ClassNames = os.listdir(self.DatasetPath)
                PreTrained = self.CKboxPretrainMode.isChecked()
                # BNorm = self.CKboxBatchNorm.isChecked()
                Shuffle = self.CKboxShuffle.isChecked()
                DropLast = self.CKboxDropLast.isChecked()

                self.TeditTrainLog.append('The number of classes: %d' %(len(ClassNames)))
                self.TeditTrainLog.append("Using {} device".format(self.Device))

                self.TrainProcess.doRun(self.CUDA_AVAI, self.DataParallel, self.Device,
                                self.Model, self.Optimizer,
                                self.Epochs, self.BatchSize, self.ResizeRes, self.StopStation, self.NumSplit,
                                self.LrRate,
                                self.DatasetPath, self.WeightPath, self.DestPath,
                                self.SetName, Shuffle, DropLast, self.NumWorker,
                                ClassNames,
                                0, False, PreTrained,
                                0, 0, 0, 999)
                self.BtnStartTrain.setText('Start')
                self.Flag1 = 0
                self.TeditTrainLog.append('=====================Complete training !=====================')
        elif self.Flag4 == 0 or len(self.DatasetPath) == 0:
            self.TeditTrainLog.append('Please load the dataset for training!')

    def doInference(self):
        start = time.perf_counter()
        if self.Flag2_infer == self.Flag3_infer == self.Flag4_infer == 1:
            self.TeditInferLog.append('=====================Start Inference !=====================')
            self.BtnStartInfer.setText('Inferring')
            self.InferenceProcess.doPrediction(
                                                   device_infer=self.Device_infer, modelname_infer=self.Model_infer,
                                                   weightpath_infer=self.WeightPath_infer,
                                                   imgpath_infer=self.ImagePath_infer,
                                                   classinfo_infer=self.Classinfo_infer
                                                 )
            self.TeditInferLog.append(self.InferenceProcess.clsInfo)
            self.TeditInferLog.append(self.InferenceProcess.matchInfo)
            time.sleep(0.01)
            self.BtnStartInfer.setText('Infer')
            self.TeditInferLog.append('=====================Complete Inference !=====================')
            end = time.perf_counter()
            time_cost_infer = end - start
            self.TeditInferLog.append('Time cost: {}'.format(time_cost_infer) + 's')
            self.LiditInfertime.setText(str(time_cost_infer)[:6] + 's')
            self.LiditInfertime_perimage.setText(str(time_cost_infer/self.Number)[:6] + 's')
            self.Flag1_infer = 1
        else:
            self.TeditInferLog.append('Please make sure you have loaded the model, class information and images correctly !')

    def doStop(self):
        self.TeditTrainLog.append('Stop training!')
        self.BtnStartTrain.setText('Start')
        self.ThreadTrain.terminate()
        self.Flag1 = 0

    def startTrain(self):
        # Value change
        if self.Flag1 == 0:
            self.ThreadTrain = Worker(self.doTrain)
            self.ThreadTrain.start()

    def startInfer(self):
        self.ThradInfer = Worker(self.doInference)
        self.ThradInfer.start()

    def stopTrain(self):
        if self.Flag1 == 1:
            self.ThreadStop = Worker(self.doStop)
            self.ThreadStop.start()

    ## Set feedback value
    def setCurrentEpoch(self, CurrentEpoch):
        self.CurrentEpoch = CurrentEpoch

    ## Set bar value
    def setTrainProcess(self, EpochLog):
        self.TeditTrainLog.append('Time cost: %.4f' % (EpochLog[0]) + '  Learning rate: %.9f' % (EpochLog[1][0]))
        self.BarTrainProcess.setValue(round((self.CurrentEpoch) / self.Epochs * 100))

    def setEachEpoch(self, EachEpoch):
        self.BarEachEpoch.setValue(EachEpoch)

    ## Print training process information
    def printCurrentSplit(self, CurrentSplit):
        self.CurrentSplit = CurrentSplit
        if CurrentSplit != 0:
            self.TeditTrainLog.append('***************run{}*****************'.format(CurrentSplit))

        self.initOutput()
        if self.Flag2 == 1:
            self.Ax[0].clear()
            self.Ax[1].clear()
            self.initAxis()
            self.Flag3 = 0

    def printNumTrainValData(self, NumTrainValData):
        self.TeditTrainLog.append('The number of training data is: %d' %(NumTrainValData[0]))
        self.TeditTrainLog.append('The number of testing data is: %d' %(NumTrainValData[1]))

    def printMeticsList(self, MeticsList):
        self.TeditTrainLog.append('Epoch: [%d/%d]\tCrossValid: [%d/%d]' % (self.CurrentEpoch, self.Epochs, self.CurrentSplit, self.NumSplit))
        self.TeditTrainLog.append(
            'TrainLoss: %.4f  TrainAccuracy: %.4f  ValLoss: %.4f  ValAccuracy: %.4f  AvgRecall: %.4f  AvgPrecision: %.4f  AvgF1Score: %.4f'
            % (MeticsList[0], MeticsList[1], MeticsList[2], MeticsList[3], MeticsList[4], MeticsList[5], MeticsList[6]))

        self.Loss.extend([MeticsList[0]])
        self.Accuracy.extend([MeticsList[1]])
        self.TestLoss.extend([MeticsList[2]])
        self.TestAcc.extend([MeticsList[3]])

        self.plotMetrics()


    def printTimeCost(self, TimeCost):
        self.TeditTrainLog.append('Training time: %.4f' % (TimeCost))

    ## Close event
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def printInferresult(self,result):
        self.TeditInferLog.append("File: {}   Predicted Class: {}   Probability: {:.3}\n".format(result[0],result[1],result[2]))

    def visual_infer(self):
        if self.Flag1_infer == 0:
            self.TeditInferLog.append('Please do the inference first !!!')
        elif self.Flag1_infer == 1:
            self.InferenceProcess.visualizeInfer(self.numimgs_fig_infer)

if __name__ == '__main__':

    app = QApplication([])
    Window = MainApp()
    Window.show()
    app.exec_()
