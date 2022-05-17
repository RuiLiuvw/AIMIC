import os
import importlib
from typing import Any
from ...utils import colorText

ClsModelRegistry = {}


def registerClsModels(name):
    def registerModelClass(cls):
        if name in ClsModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        ClsModelRegistry[name] = cls
        return ClsModelRegistry[name]
    return registerModelClass


def buildClassificationModel(ModelName, NumClasses, **kwargs: Any):
    model = None
    if ModelName in ClsModelRegistry:
        model = ClsModelRegistry[ModelName](num_classes=NumClasses, **kwargs)
    else:
        SupportedModels = list(ClsModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, MName in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colorText(MName))
            
    # for key in ClsModelRegistry.keys():
    #     print(key)
    return model


# Automatically import the models
ModelsDir = os.path.dirname(__file__)
for file in os.listdir(ModelsDir):
    path = os.path.join(ModelsDir, file)
    if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
    ):
        ModelName = file[: file.find(".py")] if file.endswith(".py") else file
        Module = importlib.import_module("lib.model.classification." + ModelName)