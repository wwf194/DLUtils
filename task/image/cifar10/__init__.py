import sys
import numpy as np
import torch
import DLUtils
from .. import ImageClassificationTask

LabelContent = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

class CIFAR10(ImageClassificationTask):
    def __init__(self):
        super().__init__()
        self._INIT()
    def _INIT(self):
        Param = self.Param
        self.DataLoaderList = set()
        return self
    def SetDataPath(self, DataPath, CheckIntegrity=True):
        Param = self.Param
        Param.DataPath = DataPath
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "cifar10-config.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=True, SplitKeyException=["Folder"])
        DataPath = ExtractDataSetFile(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config.Folder)
        self.DataPath = DataPath
        self.Config = Config
        self.IsDataPathBind = True
        return self
    def PreprocessData(self, Preprocess, BatchSize=1000):
        assert self.IsDataPathBind
        Data = LoadDataSet(self.DataPath)
        return self
    def TestData(self, BatchSize=64):
        return self.DataLoader("Test", BatchSize)
    def TrainData(self, BatchSize=64):
        return self.DataLoader("Train", BatchSize)
    def DataLoader(self, Type, BatchSize=64):
        if not hasattr(self, "Data"):
            self.Data = LoadDataSet(self.DataPath)
        Data = self.Data
        if Type in ["Train"]:
            Data = Data.Train
            BatchNum=self.TrainBatchNum(BatchSize)
        else:
            Data = Data.Test
            BatchNum=self.TestBatchNum(BatchSize)
        _DataLoader = DataLoader(
            DataFetcher(Data.Image, Data.Label),
            BatchSize=BatchSize, BatchNum=BatchNum
        )
        self.DataLoaderList.add(_DataLoader)
        if hasattr(self, "Device"):
            _DataLoader.SetDevice(self.Device)
        return _DataLoader
    def TrainBatchNum(self, BatchSize):
        Param = self.Param
        Config = self.Config
        BatchNum = Config.Train.Num // BatchSize
        if Config.Train.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def TestBatchNum(self, BatchSize):
        Param = self.Param
        Config = self.Config
        BatchNum = Config.Test.Num // BatchSize
        if Config.Test.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def SetDevice(self, Device, IsRoot=True):
        for Data in self.DataLoaderList:
            Data.SetDevice(Device)
        self.Device = Device
        return self

from six.moves import cPickle as pickle
import numpy as np
import os
import platform

#读取文件
def LoadPickle(f):
    version = platform.python_version_tuple() # 取python版本号
    if version[0] == '2':
        return pickle.load(f) # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def LoadDataSetFile(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = LoadPickle(f)   # dict类型
        Image = datadict['data']        # X, ndarray, 像素值
        Label = datadict['labels']      # Y, list, 标签, 分类
    
    Image = Image.reshape(10000, 3, 32, 32).astype(np.float32)
    # Image = Image.transpose(0,2,3,1)
    Label = np.array(Label).astype(np.uint8)
    return Image, Label

def ExtractImage(Data, IndexList=None, PlotNum=10, SavePath="./", LabelContent=None):
    if IndexList is None:
        IndexList = DLUtils.MultipleRandomIntInRange(0, 60000, PlotNum)
    for Index in IndexList:
        if Index >= 50000:
            Type = "Test "
            Image = Data.Test.Image[Index - 50000]
            Label = Data.Test.Label[Index - 50000]
        else:
            Type = "Train"
            Image = Data.Train.Image[Index]
            Label = Data.Train.Label[Index]

        #Image = Image.reshape(32, 32, 3) # np.float32
        Image = Image.transpose(1, 2, 0)
        Image = Image / 256.0
        DLUtils.plot.NpArray2ImageFile(
            Image, SavePath + "{0} {1:0>5} Class {2} {3}.png".format(Type, Index, Label, LabelContent[Label])
        )

def LoadDataSet(FolderPath, PlotExampleImage=True):
    """ load all of cifar """
    ImageList = [] # list
    LabelList = []

    # TrainData
    for Index in range(1,6):
        FilePath = FolderPath + f'data_batch_{Index}'
        Image, Label = LoadDataSetFile(FilePath)
        ImageList.append(Image) # 在list尾部添加对象X, x = [..., [X]]
        LabelList.append(Label)    
        ImageTrain = np.concatenate(ImageList) # [ndarray, ndarray] 合并为一个ndarray
        LabelTrain = np.concatenate(LabelList)

    # TestData
    IamgeTest, LabelTest = LoadDataSetFile(FolderPath + 'test_batch')
    
    Data = DLUtils.param({
        "Train":{
            # np.ndarray dtype=np.int64 all values are non-negative integers.
            "Image": ImageTrain,
            "Label": LabelTrain
        },
        "Test":{
            "Image": IamgeTest,
            "Label": LabelTest
        }
    })

    if PlotExampleImage:
        ExampleDir = FolderPath + "example/"
        DLUtils.file.EnsureDir(ExampleDir)
        if DLUtils.file.ExistsDir(ExampleDir) \
            and len(DLUtils.file.ListAllFiles(ExampleDir)) < 30:
                ExtractImage(Data=Data, SavePath=ExampleDir, LabelContent=LabelContent)
    return Data

def ExtractDataSetFile(Path, MoveCompressedFile2ExtractFolder=True, IsFile=True):
    Path = DLUtils.file.ToAbsPath(Path)
    if DLUtils.file.FileExists(Path): # extract from mnist zip file
        # typical file name: cifar-10-python.tar.gz
        CompressFilePath = Path
        FolderPath = DLUtils.file.FolderPathOfFile(Path) # extract to same parent folder
        if DLUtils.file.IsGzFile(Path): 
            assert Path.endswith(".gz")
            ExtractFilePath = DLUtils.file.ExtractGzFile(CompressFilePath)
            Path = ExtractFilePath
        if DLUtils.file.IsZipFile(Path): 
            assert FileName.endswith(".zip")
            ExtractFolderPath = FolderPath + "cifar10/"
            Path = DLUtils.file.ExtractZipFile(Path, ExtractFolderPath)
        if DLUtils.file.IsTarFile(Path):
            FilePath = Path
            assert FilePath.endswith(".tar")
            ExtractFolderPath = DLUtils.file.FolderPathOfFile(FilePath) + "cifar10/"
            ExtractFolderPath = DLUtils.file.ExtractTarFile(FilePath, ExtractFolderPath)
            if not DLUtils.file.IsSameFile(FilePath, CompressFilePath):
                DLUtils.file.DeleteFile(FilePath)
            DLUtils.file.MoveFolder(ExtractFolderPath + "cifar-10-batches-py/", ExtractFolderPath)
            FolderPath = ExtractFolderPath
        if MoveCompressedFile2ExtractFolder:
            DLUtils.file.MoveFile(CompressFilePath, ExtractFolderPath)
    elif DLUtils.file.FolderExists(Path) and not IsFile:
        FolderPath = Path
    else:
        if Path.endswith(".gz"):
            # .zip file already extracted and put into extracted folder.
            FileName = DLUtils.FileNameFromPath(Path)
            FolderPath = DLUtils.FolderPathOfFile(Path) + "cifar10/"
            assert DLUtils.FolderExists(FolderPath)
        else:
            raise Exception()
    return FolderPath

def DataSetFolderConfig(DataSetFolderPath, SaveDir=None):
    FolderConfigFilePath = DLUtils.file.FolderPathOfFile(__file__) + "cifar10-folder-config.jsonc"
    if DLUtils.file.FileExists(FolderConfigFilePath):
        ConfigFolder = DLUtils.file.JsonFile2Param(FolderConfigFilePath)
    else:
        ConfigFolder = DLUtils.file.FolderConfig(DataSetFolderPath)
    if SaveDir is not None:
        ConfigFolder.ToJsonFile(SaveDir + "cifar10-folder-config.jsonc")
    return ConfigFolder

def DataSetConfig(DataSetFolderPath, SaveDir=None):
    DataSetFolderPath = DLUtils.file.StandardizePath(DataSetFolderPath)
    Data = LoadDataSet(DataSetFolderPath)
    Config = DLUtils.Param({
        "Test.Num": 10000,
        "Train.Num": 50000,
        "Image.Shape": [3, 32, 32],
        "Class.Num": 10,
        "Image.Value.Range": [0, 255],
        "Image.Value.Type": "uint8",
        "Label.Content": LabelContent
    })
    Config.Absorb(DataSetStat(Data))
    Config.Folder = DataSetFolderConfig(DataSetFolderPath, SaveDir)
    if SaveDir is not None:
        Config.ToJsonFile(SaveDir + "cifar10-config.jsonc")
    return Config

def DataSetStat(Data):
    Stat = DLUtils.Param()
    ColorChannelStat = Stat.Image.ColorChannel
    ColorChannelStat.Train.Mean = np.nanmean(Data.Train.Image, axis=(0, 1, 2)) # (ImageNum, 32, 32, 3)
    ColorChannelStat.Train.Std = np.nanstd(Data.Train.Image, axis=(0, 1, 2)) # (ImageNum, 32, 32, 3)
    ColorChannelStat.Test.Mean = np.nanmean(Data.Test.Image, axis=(0, 1, 2)) # (ImageNum, 32, 32, 3)
    ColorChannelStat.Test.Std = np.nanstd(Data.Test.Image, axis=(0, 1, 2)) # (ImageNum, 32, 32, 3)
    return Stat

class DataFetcher(torch.utils.data.Dataset, DLUtils.module.AbstractModule):
    def __init__(self, Image, Label, BatchSize=None):
        DLUtils.module.AbstractModule.__init__(self)
        torch.utils.data.Dataset.__init__(self)
        self.Image = Image
        self.Label = Label
        self.BatchSize = BatchSize
    def SetDevice(self, Device, IsRoot=True):
        self.Device = Device
        self.Log(f"change device to {Device}")
        return self
    def __getitem__(self, Index): # get train data with Index.
        Image = self.Image[Index]
        Label = self.Label[Index]
        return Image, Label
    def __len__(self):
        return self.Image.shape[0]

class DataLoader(torch.utils.data.DataLoader, DLUtils.module.AbstractModule):
    def __init__(self, DataFetcher=None, BatchSize=None, BatchNum=None):
        self.DataFetcher = DataFetcher
        DLUtils.module.AbstractModule.__init__(self)
        torch.utils.data.DataLoader.__init__(
            self,
            dataset=DataFetcher,
            batch_size=BatchSize,
            # num_workers=2 # Setting num_workers > 1 might severely slow down speed.
        )
        self.BatchSize = BatchSize
        self._BatchNum = BatchNum
        if hasattr(DataFetcher, "Device"):
            self.Device = DataFetcher.Device
        self.Reset()
    def BeforeEpoch(self, Dict):
        self.Reset()
    def AfterEpoch(self, Dict):
        self.Reset()
    def GetNextBatch(self, BatchIndex):
        Image, Label = next(self.Iter)
        # return Image.to(self.Device), Label.to(self.Device)
        Image, Label
    Get = GetNextBatch
    def SetDevice(self, Device, IsRoot=True):
        self.DataFetcher.SetDevice(Device)
        self.Device = Device
        return self
    def BatchNum(self):
        return self._BatchNum
    def Reset(self):
        self.Iter = iter(self)
        return self
    def Init(self, IsSuper=False, IsRoot=True):  
        return super().Init(IsSuper=True, IsRoot=IsRoot)