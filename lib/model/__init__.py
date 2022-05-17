from typing import Any
from .classification import buildClassificationModel


SUPPORTED_TASKS = ["segmentation", "classification", "detection"]

def getModel(
            ModelName,
            NumberofClasses,
            # InitWeights=False,
            DatasetCategory='classification',
            **kwargs: Any
            ):
    
    model = None
    if ModelName == 'inception':
        ModelName = 'inception3'
    if DatasetCategory == "classification":
        model = buildClassificationModel(ModelName, NumberofClasses, **kwargs)
    # elif dataset_category == "segmentation":
    #     model = build_segmentation_model(opts=opts)
    # elif dataset_category == "detection":
    #     model = build_detection_model(opts=opts)
    else:
        task_str = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(DatasetCategory)
        for i, task_name in enumerate(SUPPORTED_TASKS):
            task_str += "\n\t {}: {}".format(i, task_name)
    return model