import sys
import numpy as np
import torch
import DLUtils
from .. import ImageClassificationTask

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
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "cifar10-folder-config.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=False)
        DataPath = ExtractDataSetFile(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config)
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

class DataManagerForEpochBatchTrain(DLUtils.module.AbstractModule):
    def __init__(self):
        #DLUtils.transform.InitForNonModel(self, param, **kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        cache = self.cache
        param = self.param
        cache.flows = DLUtils.EmptyPyObj()
        # self.CreateFlowRandom("DefaultTest", "Test")
        # self.CreateFlowRandom("DefaultTrain", "Train")
        self.PrepareData()
        return self
    def PrepareData(self):
        param = self.param
        cache = self.cache
        UseCachedDataTransform = False
        if HasAttrs(param, "Data.Transform.Md5"):
            Md5s = DLUtils.file.ListFilesAndCalculateMd5("./cache/", Md5InKeys=True)
            if param.Data.Transform.Md5 in Md5s.keys():
                FileName = Md5s[param.Data.Transform.Md5]
                cache.Data = DLUtils.DataFile2PyObj("./cache/" + FileName)
                UseCachedDataTransform = True
        if not UseCachedDataTransform:
            self.LoadData(Dir="Auto")
            self.ApplyTransformOnData()
    def GetInputOutputShape(self):
        cache = self.cache
        InputShape = cache.Data.Train.Images[0].shape
        if len(InputShape) == 1:
            InputShape = InputShape[0]
        return InputShape, 10 # InputShape, OutputShape
    def ApplyTransformOnData(self, TransformParam="Auto", Type=["Train", "Test"], Save=True):
        DLUtils.Log("Applying transformation on dataset images...")
        param = self.param
        cache = self.cache
        if TransformParam in ["Auto"]:
            TransformParam = param.Data.Transform
        assert hasattr(cache, "Data")
        for _Type in Type:
            Data = getattr(cache.Data, _Type)
            Images = GetAttrs(Data.Images)
            for Transform in TransformParam.Methods:
                if Transform.Type in ["ToGivenDataType"]:
                    Images = DLUtils.ToGivenDataTypeNp(Images, DataType=Transform.DataType)
                elif Transform.Type in ["Color2Gray", "ColorImage2GrayImage"]:
                    Images = DLUtils.plot.ColorImage2GrayImage(Images, ColorAxis=3)
                elif Transform.Type in ["Norm2Mean0Std1"]:
                    EnsureAttrs(Transform, "axis", None)
                    Images = DLUtils.math.Norm2Mean0Std1Np(Images, axis=tuple(GetAttrs(Transform.axis)))
                elif Transform.Type in ["Flatten"]:
                    # Plot example images before Flatten, which is usually the last step.
                    DLUtils.plot.PlotExampleImage(Images, SaveDir=DLUtils.GetMainSaveDir() + "Dataset/", SaveName="CIFAR10-%s"%_Type)
                    Shape = Images.shape
                    Images = Images.reshape(Shape[0], -1)
                else:
                    raise Exception(Transform.Type)
            SetAttrs(Data, "Images", value=Images)        
        DLUtils.Log("Applied transformation on dataset images.")
        if Save:
            SavePath = DLUtils.RenameFileIfExists("./" + "cache/" + "Cifar10-Transformed-Cached.data")
            DLUtils.PyObj2DataFile(
                cache.Data,
                SavePath
            )
            Md5 = DLUtils.File2MD5(SavePath)
            DLUtils.Log("Saved transformed data. Md5:%s"%Md5)
            SetAttrs(param, "Data.Transform.Md5")
    def Labels2ClassNames(self, Labels):
        ClassNames = []
        for Label in Labels:
            ClassNames.append()
    def Label2ClassName(self, Label):
        return
    def NotifyEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def LoadData(self, Dir="Auto"):
        cache = self.cache
        if Dir in ["Auto", "auto"]:
            Dir = DLUtils.dataset.GetDatasetPath("CIFAR10")
        DataFile = Dir + "CIFAR10-Data"
        cache.Data = DLUtils.json.DataFile2PyObj(DataFile)
        return
    def EstimateBatchNum(self, BatchSize, Type="Train"):
        cache = self.cache
        Data = getattr(cache.Data, Type)
        return DLUtils.dataset.CalculateBatchNum(BatchSize, Data.Images.Num)
    def HasFlow(self, Name):
        return hasattr(self.cache.flows, Name)
    def GetFlow(self, Name):
        return getattr(self.cache.flows, Name)
    def CreateFlow(self, Name, BatchParam, Type="Train", IsRandom=False):
        cache = self.cache
        # if self.HasFlow(Name):
        #     DLUtils.AddWarning("Overwriting existing flow: %s"%Name)
        #self.ClearFlow(Type=Type)
        #flow = cache.flows.SetAttr(Name, DLUtils.EmptyPyObj())
        flow = DLUtils.EmptyPyObj()
        flow.Name = Name
        flow.IndexCurrent = 0
        flow.BatchSize = BatchParam.Batch.Size
        Data = getattr(cache.Data, Type)
        flow.BatchNumMax = DLUtils.dataset.CalculateBatchNum(flow.BatchSize, Data.Images.Num)
        flow.IndexMax = Data.Images.Num
        flow.Data = Data
        flow.Images = GetAttrs(flow.Data.Images)
        flow.Labels = GetAttrs(flow.Data.Labels)
        if hasattr(BatchParam, "Batch.Num"): # Limited Num of Batches
            flow.BatchNum = BatchParam.Batch.Num
        else: # All
            flow.BatchNum = flow.BatchNumMax
        flow.BatchIndex = -1
        if IsRandom:
            flow.IsRandom = True
            flow.RandomBatchOrder = DLUtils.RandomOrder(range(flow.BatchNum))
            flow.RandomBatchIndex = 0
        else:
            flow.IsRandom = False
        self.ResetFlow(flow)
        return flow
    def CreateFlowRandom(self, BatchParam, Name, Type):
        return self.CreateFlow(BatchParam, Name, Type, IsRandom=True)
    #def ClearFlow(self, Type="Train"):
    def ClearFlow(self, Name):
        cache = self.cache
        if hasattr(cache.flows, Name):
            delattr(cache.flows, Name)
        else:
            DLUtils.AddWarning("No such flow: %s"%Name)
    # def GetBatch(self, Name):
    #     self.GetBatch(self, self.GetFlow(Name))
    def GetBatch(self, flow):
        flow.BatchIndex += 1
        assert flow.BatchIndex < flow.BatchNum
        if flow.IsRandom:
            return self.GetBatchRandomFromFlow(flow)
        assert flow.IndexCurrent <= flow.IndexMax
        IndexStart = flow.IndexCurrent
        IndexEnd = min(IndexStart + flow.BatchSize, flow.IndexMax)
        DataBatch = self.GetBatchFromIndex(flow, IndexStart, IndexEnd)
        flow.IndexCurrent = IndexEnd
        if flow.IndexCurrent >= flow.IndexMax:
            flow.IsEnd = True
        return DataBatch
    def GetData(self, Type):
        return getattr(self.cache.Data, Type)
    def GetBatchFromIndex(self, Data, IndexStart, IndexEnd):
        DataBatch = {
            "Input": DLUtils.NpArray2Tensor(
                    Data.Images[IndexStart:IndexEnd]
                ).to(self.GetTensorLocation()),
            "Output": DLUtils.NpArray2Tensor(
                    Data.Labels[IndexStart:IndexEnd],
                    DataType=torch.long # CrossEntropyLoss requires label to be LongTensor.
                ).to(self.GetTensorLocation()),
        }
        return DataBatch
    def GetBatchRandom(self, BatchParam, Type, Seed=None):
        BatchSize = BatchParam.Batch.Size
        Data = self.GetData(self, Type)
        if Seed is not None:
            IndexStart = Seed % (Data.Images.Num - BatchSize + 1)
        else:
            IndexStart = DLUtils.RandomIntInRange(0, Data.Images.Num - BatchSize)
        IndexEnd = IndexStart + BatchSize
        assert IndexEnd < Data.Images.Num
        return self.GetBatchFromIndex(self, Data, IndexStart, IndexEnd)
    def GetBatchRandomFromFlow(self, Name):
        flow = self.GetFlow(Name)
        assert flow.IsRandom
        IndexStart = flow.RandomBatchOrder[flow.RandomBatchIndex] * flow.BatchSize
        IndexEnd = min(IndexStart + flow.BatchSize, flow.IndexMax)
        DataBatch = self.GetBatchFromIndex(flow.Images, IndexStart, IndexEnd)
        flow.RandomBatchIndex += 1
        if flow.RandomBatchIndex > flow.IndexMax:
            flow.RandomBatchIndex = 0
        return DataBatch
    def ResetFlowFromName(self, Name):
        #flow = self.GetFlow(Name)
        self.ResetFlow(Name)
    def ResetFlow(self, flow):
        flow.IndexCurrent = 0
        flow.BatchIndex = -1
        flow.IsEnd = False
    def GetBatchNum(self, Name="Train"):
        cache = self.cache
        flow = getattr(cache.flows, Name)
        return flow.BatchNum

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
    Label = np.array(Label)
    return Image, Label

def ExtractImage(Data, IndexList=None, PlotNum=10, SavePath="./"):
    if IndexList is None:
        IndexList = DLUtils.MultipleRandomIntInRange(0, 70000, PlotNum)
    for Index in IndexList:
        if Index >= 50000:
            Type = "Test"
            Image = Data.Test.Image[Index - 50000]
            Label = Data.Test.Label[Index - 50000]
        else:
            Type = "Train"
            Image = Data.Train.Image[Index]
            Label = Data.Train.Label[Index]
        if len(Image.shape) == 1:
            Image = Image.reshape(32, 32, 3)
        DLUtils.plot.NpArray2ImageFile(
            Image, SavePath + f"{Type} No.{Index} Class:{Label}.png" 
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
        ExampleDir = FolderPath + "./example"
        DLUtils.file.EnsureDir(ExampleDir)
        if DLUtils.file.ExistsDir(ExampleDir) \
            and len(DLUtils.file.ListAllFiles(ExampleDir)) < 10:
                ExtractImage(Data=Data, SavePath=ExampleDir)
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

def DataSetConfig(DataSetFolderPath, SaveDir):
    Config = DLUtils.file.FolderConfig(DataSetFolderPath)
    Config.ToJsonFile(SaveDir + "cifar10-folder-config.jsonc")