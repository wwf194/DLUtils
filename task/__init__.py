

import DLUtils
import DLUtils.task.image as image
# from .image import ImageClassTask
# from .image import MNIST, CIFAR10
# from .image import ImageNet1k
# DatasetMap = DLUtils.IterableKeyToElement({
#     ("MNIST", "mnist"): MNIST,
#     ("CIFAR", "cifar", "cifar10"): CIFAR10,
#     ("ImageClassification"): ImageClassTask
# })

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from DLUtils.task.image.mnist import MNISTHandler

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
        return ImageClassTask()

class TaskHandler(DLUtils.Module):
    pass

def GetMNISTHandler() -> "MNISTHandler":
    from .image.mnist import MNISTHandler
    return MNISTHandler()

def GetTaskHandler(Name):
    if Name in ["mnist", "MNIST"]:
        from .image.mnist import MNISTHandler
        Handler = MNISTHandler()
        return Handler
    else:
        raise Exception()
GetDatasetHandler = GetTaskHandler