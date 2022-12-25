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
        Param._CLASS = "DLUtils.task.image.classification.CIFAR10"
        Param.Train.Num = 60000
        Param.Test.Num = 10000
        self.DataLoaderList = set()
        return self
    def SetDataPath(self, DataPath, CheckIntegrity=True):
        DLUtils.file.EnsureDir(DataPath)
        Param = self.Param
        Param.DataPath = DataPath
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "cifar10-config.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=False)
        DataPath = ExtractDataSetFile(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config.Folder)
        self.DataPath = DataPath
        self.IsDataPathBind = True
        return self
    def SetParam(self):
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
        BatchNum = Param.Train.Num // BatchSize
        if Param.Train.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def TestBatchNum(self, BatchSize):
        Param = self.Param
        BatchNum = Param.Test.Num // BatchSize
        if Param.Test.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def SetDevice(self, Device, IsRoot=True):
        for Data in self.DataLoaderList:
            Data.SetDevice(Device)
        self.Device = Device
        return self


def LoadOriginalFiles(Dir):
    Files = DLUtils.file.ListFiles(Dir)
    FileMD5s = DatasetConfig.Original.Files.MD5.ToDict()
    FileNames = DatasetConfig.Original.Files.Train + DatasetConfig.Original.Files.Test
    Dict = {}
    for File in Files:
        if File in FileNames:
            assert DLUtils.File2MD5(Dir + File) == FileMD5s[File]
            DataDict = DLUtils.file.LoadBinaryFilePickle(Dir + File)
            keys, values = DLUtils.Unzip(DataDict.items()) # items() cause logic error if altering dict in items() for-loop.
            for key, value in zip(keys, values):
                if isinstance(key, bytes):
                    DataDict[DLUtils.Bytes2Str(key)] = value # Keys in original dict are bytes. Turn them to string for convenience.
                    DataDict.pop(key)
            Dict[File] = DataDict
    assert len(Dict) != DatasetConfig.Original.Files.Num
    return Dict

def OriginalFiles2DataFile(LoadDir, SaveDir):
    OriginalDict = LoadOriginalFiles(LoadDir)
    DataDict = DLUtils.EmptyPyObj()
    DataDict.Train = ProcessOriginalDataDict(OriginalDict, FileNameList=DatasetConfig.Original.Files.Train)
    DataDict.Test = ProcessOriginalDataDict(OriginalDict, FileNameList=DatasetConfig.Original.Files.Test)
    DLUtils.json.PyObj2DataFile(DataDict, SaveDir)

def ProcessOriginalDataDict(Dict, FileNameList):
    Labels = []
    Images = []
    FileNames = []
    for File in FileNameList:
        Data = Dict[File]
        # Keys: batch_label, labels, data, filenames
        # Pixel values are integers with range [0, 255], so using datatype np.uint8.
        # Saving as np.float32 will take ~ x10 disk memory as original files.
        _Images = DLUtils.ToNpArray(Data["data"], DataType=np.uint8)
        _Images = _Images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Images.append(_Images)
        _Labels = DLUtils.ToNpArray(Data["labels"], DataType=np.uint8)
        Labels.append(_Labels)
        FileNames += Data["filenames"]
    Labels = np.concatenate(Labels, axis=0)
    Images = np.concatenate(Images, axis=0)
    DataObj = DLUtils.PyObj({
        "Labels": Labels,
        "Images": Images,
        "FileNames": FileNames
    })
    ImageNum = Images.shape[0]
    return DataObj

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
    
    Image = Image.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
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
        if len(Image.shape) == 1:
            Image = Image.reshape(32, 32, 3) # np.float32
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

def ExtractDataSetFile(Path, MoveCompressedFile2ExtractFolder=True):
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
    elif DLUtils.file.FolderExists(Path):
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
        "Image.Shape": [32, 32, 3],
        "Class.Num": 10,
        "Image.Value.Range": [0, 255],
        "Image.Value.Type": "uint8",
        "Label.Content": LabelContent
    })
    Config.Absorb(DataSetStat(Data))
    Config.Folder = DataSetFolderConfig(DataSetFolderPath)
    if SaveDir is not None:
        Config.ToJsonFile(SaveDir + "cifar10-config.jsonc")
    return Config

def DataSetStat(Data):
    Stat = DLUtils.Param()
    ColorChannelStat = Stat.Image.ColorChannel
    ColorChannelStat.Train.Mean = np.nanmean(Data.Train.Image, axis=(0, 1, 2)) # [ImageNum, 32, 32, 3]
    ColorChannelStat.Train.Std = np.nanmean(Data.Train.Image, axis=(0, 1, 2)) # [ImageNum, 32, 32, 3]
    ColorChannelStat.Test.Mean = np.nanmean(Data.Test.Image, axis=(0, 1, 2)) # [ImageNum, 32, 32, 3]
    ColorChannelStat.Test.Std = np.nanmean(Data.Test.Image, axis=(0, 1, 2)) # [ImageNum, 32, 32, 3]
    return Stat