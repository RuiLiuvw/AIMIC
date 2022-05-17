import numpy as np

import torch
from torch import nn

from PyQt5.QtCore import pyqtSignal, QObject


class CnnTrain(QObject):
    CurrentIter = pyqtSignal(int)

    def __init__(self) -> None:
        super(CnnTrain, self).__init__()

    def doRun(self, CUDA_AVAI, Device, NumClasses, Optim, Model, ModelName, TrainDL, TestDL, LossMode):
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0
        if LossMode == 0:
            lossFun = nn.CrossEntropyLoss()
        # elif LossMode == 2:
        #     lossFun = nn.MultiLabelMarginLoss()

        PositivesGT, PositivesPre, TruePositives = np.zeros((3, NumClasses), dtype=int)
        Recall, Precision, F1Score = np.zeros((3, NumClasses), dtype=float)

        Temp = 0
        Model.train()  # Normalization is different in trainning and evaluation
        for x, y in TrainDL:  # iterate x, y in dataloader (one batch data)
            x, y = x.to(Device), y.to(Device)

            Optim.zero_grad()  # Initialize gradient, preventing accumulation

            if LossMode == 0:
                if 'googlenet' in ModelName:
                    YPred, aux_logits2, aux_logits1 = Model(x)
                    loss0 = lossFun(YPred, y)
                    loss1 = lossFun(aux_logits1, y)
                    loss2 = lossFun(aux_logits2, y)
                    loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                elif 'inception' in ModelName:
                    YPred, aux_logits1 = Model(x)
                    loss0 = lossFun(YPred, y)
                    loss1 = lossFun(aux_logits1, y)
                    loss = loss0 + loss1 * 0.3
                else:
                    YPred = Model(x)  # prediction
                    loss = lossFun(YPred, y)

            loss.backward()  # backpropagation
            Optim.step()  # optimize model's weight
            with torch.no_grad():
                YPred = torch.argmax(YPred, dim=1)
                NumCorrect += (YPred == y).sum().item()
                NumTotal += y.size(0)
                RunningLoss += loss.item()

                Temp += 1
                self.CurrentIter.emit(Temp)

        TrainLoss = RunningLoss / NumTotal
        TrainAcc = NumCorrect / NumTotal

        val_NumCorrect = 0
        val_NumTotal = 0
        val_RunningLoss = 0

        Model.eval()
        with torch.no_grad():
            for x, y in TestDL:
                x, y = x.to(Device), y.to(Device)

                if LossMode == 0:
                    YPred = Model(x)  # Evaluation
                    loss = lossFun(YPred, y)
                elif LossMode == 1:
                    if CUDA_AVAI:
                        labels = torch.cuda.LongTensor(y)
                        labels = torch.eye(NumClasses).cuda().index_select(dim=0, index=labels)  # one-hot vectors
                    else:
                        labels = torch.LongTensor(y)
                        labels = torch.eye(NumClasses).index_select(dim=0, index=labels)  # one-hot vectors

                    classes, reconstructions, YPred = Model(x)
                    loss = lossFun(x, labels, classes, reconstructions)

                YPred = torch.argmax(YPred, dim=1)
                val_NumCorrect += (YPred == y).sum().item()
                BatchSize = y.size(
                    0)  # It could be unequal to the batchsize you use, if dataloader without dropping last
                val_NumTotal += BatchSize
                val_RunningLoss += loss.item()

                # Get true/false and positive/negative samples
                for i in range(BatchSize):
                    Flag1 = 0
                    Flag2 = 0
                    for k in range(NumClasses):
                        if y[i].item() == k and Flag1 == 0:
                            PositivesGT[k] += 1
                            if YPred[i] == y[i]:
                                TruePositives[k] += 1
                                # TrueNegatives += 1
                                # TrueNegatives[k] -= 1
                                Flag1 = 1
                        if YPred[i].item() == k and Flag2 == 0:
                            PositivesPre[k] += 1
                            Flag2 = 1
                        if Flag1 == 1 and Flag2 == 1:  # 提前跳出循环，提高效率
                            break

        TestLoss = val_RunningLoss / val_NumTotal
        TestAcc = val_NumCorrect / val_NumTotal

        # Compute metrics: Recall, Precision, F-score
        for k in range(NumClasses):
            if PositivesGT[k] != 0:
                Recall[k] = TruePositives[k] / PositivesGT[k]
            if PositivesPre[k] != 0:
                Precision[k] = TruePositives[k] / PositivesPre[k]
            if (Recall[k] + Precision[k]) != 0:
                F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])

        self.TrainLoss = TrainLoss
        self.TrainAcc = TrainAcc
        self.TestLoss = TestLoss
        self.TestAcc = TestAcc
        self.Recall = Recall
        self.Precision = Precision
        self.F1Score = F1Score
