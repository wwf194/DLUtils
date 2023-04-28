

import DLUtils
import DLUtils.task.image as image
from .image import ImageClassificationTask
from .image import MNIST, CIFAR10
from .image import ImageNet1k
DatasetMap = DLUtils.IterableKeyToElement({
    ("MNIST", "mnist"): MNIST,
    ("CIFAR", "cifar", "cifar10"): CIFAR10,
    ("ImageClassification"): ImageClassificationTask
})

def Dataset(Name, *Args, **Dict):
    if Name in DatasetMap:
        return DatasetMap[Name](*Args, **Dict)
    else:
        raise Exception()

def _Task(Name, *Args, **Dict):
    if Name in DatasetMap:
        return DatasetMap[Name](*Args, **Dict)
    else:
        raise Exception()

class Task:
    def __init__(self, Type=None):
        if Type is not None:
            self.SetType(Type)
    def SetType(Type):
        return DatasetMap[Type]()
    def ImageClassification(self):
        return ImageClassificationTask()