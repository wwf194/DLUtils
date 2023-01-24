import DLUtils

class ImageClassificationTask(DLUtils.module.AbstractModule):
    def __init__(self, Type=None):
        super().__init__()
        if Type is not None:
            self.SetType(Type)
    def SetType(self, Type):
        if Type in ["MNIST", "mnist"]:
            self.__class__ = MNIST
            self._INIT()
            assert isinstance(self, MNIST)
        elif Type in ["CIFAR10", "cifar10"]:
            self.__class__ = CIFAR10
            self._INIT()
            assert isinstance(self, CIFAR10)         
        elif Type in ["MSCOCO", "MS COCO"]:
            self.__class__ = MSCOCO
            self._INIT()
            assert isinstance(self, MSCOCO)
        else:
            raise Exception()
        return self
    def NewFlow(Name):
        return
    def PreprocessData(self, Items:DLUtils.param):
        for Item in Items:
            Type = Item.Type
            if Type in ["Norm"]:
                Item.Subtype = Item.setdefault("SubType", "MinMax2FixedValue")
                
            else:
                raise Exception()
        return self

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

import torch
import DLUtils

def CalculateBatchNum(BatchSize, SampleNum):
    BatchNum = SampleNum // BatchSize
    if SampleNum % BatchSize > 0:
        BatchNum += 1
    return BatchNum

# config = DLUtils.file.JsonFile2PyObj(
#     DLUtils.file.GetFileDir(__file__) + "config.jsonc"
# )

def GetDatasetPath(Name):
    if Name in ["CIFAR10", "cifar10"]:
        Name = "cifar10"
    elif Name in ["MNIST", "mnist"]:
        Name = "mnist"
    else:
        raise Exception(Name)

    assert hasattr(config, Name)
    return getattr(config, Name).path

import DLUtils.task.image.classification.mnist as mnist
import DLUtils.task.image.classification.cifar10 as cifar10

from .mnist import MNIST
from .cifar10 import CIFAR10