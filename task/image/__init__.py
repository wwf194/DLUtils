import DLUtils

class ImageClassTask(DLUtils.Module):
    def __init__(self, Type=None, *List, **Dict):
        super().__init__(*List, **Dict)
        if Type is not None:
            self.SetType(Type)
    # def MNIST(self):
    #     self.__class__ = MNIST
    #     self._INIT()
    #     assert isinstance(self, MNIST)
    #     return self
    def SetType(self, Type):
        if Type in ["MNIST", "mnist"]:
            return self.MNIST()
        elif Type in ["CIFAR10", "cifar10"]:
            self.__class__ = CIFAR10
            self._INIT()
            assert isinstance(self, CIFAR10)         
        elif Type in ["MSCOCO", "MS COCO"]:
            self.__class__ = MSCOCO
            self._INIT()
            assert isinstance(self, MSCOCO)
        elif Type in ["ImageNet", "ImageNet-1k"]:
            self.__class__ = ImageNet1k
        else:
            raise Exception()
        return self
    def PreprocessData(self, Items:DLUtils.param):
        for Item in Items:
            Type = Item.Type
            if Type in ["Norm"]:
                Item.Subtype = Item.setdefault("SubType", "MinMax2FixedValue")
                
            else:
                raise Exception()
        return self
    def RandomValidationSample(self):
        raise Exception()
    def GetSubDataSet(self, Name):
        """
        An image dataset typically consist of separate sets of data, for train and validate.
        Train set can be further separated into train and test set.
        """
        if Name in ["train", "Train"]:
            raise NotImplementedError()
        elif Name in ["test", "Test"]:
            raise NotImplementedError()
        elif Name in ["validate", "validation", "Validate", "Validation"]:
            raise NotImplementedError()
        else:
            raise Exception()
        return
    GetDataGroup = GetSubSet = GetSubDataSet
def MinMax2FixedValue(BatchSize):
    return
class Flow():
    def __init__(self):
        pass
    def SetDataPath(self, DataPath):
        self.DataPath = DataPath
    def Get():
        return 

ModuleList = [
    "CIFAR10",
    "MNIST",
    "MSCOCO",
]

def BuildModuleIfIsLegalType(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type

    if IsLegalModuleType(Type):
        if Type in ["cifar10", "CIFAR10"]:
            return cifar10.DataManagerForEpochBatchTrain()
        elif Type in ["MNIST", "mnist"]:
            return 
        else:
            raise Exception(Type)
    else:
        return None

def IsLegalModuleType(Type):
    return Type in ModuleList

def DataSetType2InputOutputOutput(Type):
    if Type in ["CIFAR10", "cifar10"]:
        return DLUtils.dataset.cifar10.DatasetConfig
    elif Type in ["MNIST", "mnist"]:
        return DLUtils.dataset.mnist.DatasetConfig
    else:
        raise Exception(Type)

ModuleList = set(ModuleList)

import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()

def CalculateBatchNum(BatchSize, SampleNum):
    BatchNum = SampleNum // BatchSize
    if SampleNum % BatchSize > 0:
        BatchNum += 1
    return BatchNum

# config = DLUtils.file.JsonFile2PyObj(
#     DLUtils.file.GetFileDir(__file__) + "config.jsonc"
# )

def GetDataPath(Name):
    if Name in ["CIFAR10", "cifar10"]:
        Name = "cifar10"
    elif Name in ["MNIST", "mnist"]:
        Name = "mnist"
    else:
        raise Exception(Name)

    assert hasattr(config, Name)
    return getattr(config, Name).path

GetDataSetPath = GetDatasetPath = GetDataPath

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import DLUtils.task.image.mnist as mnist
    import DLUtils.task.image.cifar10 as cifar10
    import DLUtils.task.image.imagenet as imagenet
    import DLUtils.task.image.mscoco as mscoco

def __getattr__(Name):
    if Name in ["mnist"]:
        import DLUtils.task.image.mnist as _mnist
        global mnist
        mnist = _mnist
        from .mnist import MNIST as _MNIST
        global MNIST
        MNIST = _MNIST
        return mnist
    elif Name in ["cifar10"]:
        import DLUtils.task.image.cifar10 as _cifar10
        global cifar10
        cifar10 = _cifar10
        from .cifar10 import CIFAR10 as _CIFAR10
        global CIFAR10
        CIFAR10 = _CIFAR10
        return cifar10
    elif Name in ["imagenet"]:
        import DLUtils.task.image.imagenet as _imagenet
        global imagenet
        imagenet = _imagenet
        from .imagenet import ImageNet1k as _ImageNet1k
        global ImageNet1k
        ImageNet1k = _ImageNet1k
        return imagenet
    elif Name in ["mscoco"]:
        import DLUtils.task.image.mscoco as _mscoco
        global mscoco
        mscoco = _mscoco
        from .mscoco import MSCOCO as _MSCOCO
        global MSCOCO
        MSCOCO = _MSCOCO
        return mscoco
    else:
        raise Exception(Name)