from typing import Any, Optional

import torch.nn as nn

from . import registerClsModels
from .baseConv2d import initWeight


# input size 224 x 224 x 3
@registerClsModels("mobilenet")
class Mobilenet(nn.Module):
    def __init__(
        self, 
        NumClasses: int = 1000,
        InitWeights=False,
        **kwargs: Any,
        ):
        super(Mobilenet, self).__init__()
        self.nclass = NumClasses

        def conv_bn(inplanes, planes, stride):
            return nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inplanes, planes, stride):
            return nn.Sequential(
                # depthwise convolution
                nn.Conv2d(inplanes, inplanes, 3, stride, 1, groups=inplanes, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                
                # pointwise convolution
                nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.nclass)
        
        if InitWeights:
            self.apply(initWeight)


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
