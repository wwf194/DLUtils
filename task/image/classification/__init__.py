import DLUtils
import DLUtils.task.image.classification.cifar10 as cifar10
import DLUtils.task.image.classification.mnist as mnist

class ImageClassificationTask:
    def __init__(self, Type=None):
        self.Param = DLUtils.Param()
        if Type is not None:
            self.SetType(Type)
    def SetType(self, Type):
        if Type in ["MNIST", "mnist"]:
            self.__class__ = MNIST
            self._INIT()
        else:
            raise Exception()
        return self
class MNIST(ImageClassificationTask):
    def __init__(self):
        super().__init__()
        self._INIT()
    def _INIT(self):
        Param = self.Param
        Param._CLASS = "DLUtils.task.image.classification.MNIST"
        return self
    def SetDataPath(self, DataPath, CheckIntegrity=True):
        DLUtils.file.EnsureDir(DataPath)
        Param = self.Param
        Param.DataPath = DataPath
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "mnist-folder-structure.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=False)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config)
        return self
    def SetParam():
        return self
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

config = DLUtils.file.JsonFile2PyObj(
    DLUtils.file.GetFileDir(__file__) + "config.jsonc"
)

def GetDatasetPath(Name):
    if Name in ["CIFAR10", "cifar10"]:
        Name = "cifar10"
    elif Name in ["MNIST", "mnist"]:
        Name = "mnist"
    else:
        raise Exception(Name)

    assert hasattr(config, Name)
    return getattr(config, Name).path



