import torch
import DLUtils
from DLUtils.module import AbstractModule
import numpy as np
import struct
from .. import ImageClassificationTask

class MNIST(ImageClassificationTask):
    def __init__(self):
        super().__init__()
        self._INIT()
    def _INIT(self):
        Param = self.Param
        Param._CLASS = "DLUtils.task.image.classification.MNIST"
        Param.Train.Num = 50000
        Param.Validation.Num = 10000
        self.DataLoaderList = set()
        return self
    def SetDataPath(self, DataPath, CheckIntegrity=True):
        DataPath = DLUtils.file.ToAbsPath(DataPath)
        DLUtils.file.EnsureDir(DataPath)
        Param = self.Param
        Param.DataPath = DataPath
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "mnist-folder-config.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=False)
        DataPath = ExtractDataSet(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config)
        self.DataPath = DataPath
        self.IsDataPathBind = True
        self.Config = Config
        return self
    def SetParam(self):
        return self
    # def PreprocessData(self, Preprocess, BatchSize=1000):
    #     assert self.IsDataPathBind
    #     Data = LoadDataSet(self.DataPath)
    #     return self
    def ValidationData(self, BatchSize=64):
        return self.DataLoader("Validation", BatchSize)
    def TrainData(self, BatchSize=64):
        return self.DataLoader("Train", BatchSize)
    # generate a dataloader
    def GetDataLoader(self, Type, BatchSize=64):
        if not hasattr(self, "Data"):
            self.Data = LoadDataSet(self.DataPath)
        Data = self.Data
        if Type in ["Train"]:
            Data = Data.Train
            BatchNum=self.TrainBatchNum(BatchSize)
        else:
            Data = Data.Validation
            BatchNum=self.ValidationBatchNum(BatchSize)
        DataLoader = DataLoader(
            DataFetcher=DataFetcher(Data.Image, Data.Label),
            BatchSize=BatchSize, BatchNum=BatchNum
        )
        self.DataLoaderList.add(DataLoader)
        # device setting
        if hasattr(self, "Device"):
            DataLoader.SetDevice(self.Device)
        return DataLoader
    def TrainBatchNum(self, BatchSize):
        Param = self.Param
        BatchNum = Param.Train.Num // BatchSize
        if Param.Train.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def ValidationBatchNum(self, BatchSize):
        Param = self.Param
        BatchNum = Param.Validation.Num // BatchSize
        if Param.Validation.Num // BatchSize > 0:
            BatchNum += 1
        return BatchNum
    def SetDevice(self, Device, IsRoot=True):
        for Data in self.DataLoaderList:
            Data.SetDevice(Device)
        self.Device = Device
        return self
    def AllTrainData(self):
        return DLUtils.param(self.Data.Train)

def LoadImage(file_name):
    # 在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它.
    # file object = open(file_name [, access_mode][, buffering])
    # file_name是包含您要访问的文件名的字符串值.
    # access_mode指定该文件已被打开，即读，写，追加等方式.
    # 0表示不使用缓冲，1表示在访问一个文件时进行缓冲.
    # 这里rb表示只能以二进制读取的方式打开一个文件.
    binfile = open(file_name, 'rb') 
    # 从一个打开的文件读取数据
    buffers = binfile.read()
    # 读取image文件前4个整型数字
    magic, num, rows, cols = struct.unpack_from('>IIII',buffers, 0)
    # 整个images数据大小为60000*28*28
    bits = num * rows * cols
    # 读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    # 关闭文件
    binfile.close()
    # 转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    
    # np.int64 -> np.uint8
    images = images.astype(np.uint8).reshape(-1, 28, 28)

    return images

def LoadLabel(file_name):
    ##   打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据    
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic,num = struct.unpack_from('>II', buffers, 0) 
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])

    # np.int64 -> np.uint8
    labels = labels.astype(np.uint8)
    return labels

def ExtractImage(Data, IndexList=None, PlotNum=10, SavePath="./"):
    if IndexList is None:
        IndexList = DLUtils.MultipleRandomIntInRange(0, 70000, PlotNum)
    for Index in IndexList:
        if Index >= 60000:
            Type = "Validation"
            Image = Data.Validation.Image[Index - 60000]
            Label = Data.Validation.Label[Index - 60000]
        else:
            Type = "Train"
            Image = Data.Train.Image[Index]
            Label = Data.Train.Label[Index]
        if len(Image.shape) == 1:
            Image = Image.reshape(28, 28)
        DLUtils.plot.NpArray2ImageFile(
            Image, SavePath + f"{Type} No.{Index} Class:{Label}.png" 
        )
    
class DataFetcher(torch.utils.data.Dataset, AbstractModule):
    def __init__(self, Image, Label, BatchSize=None):
        AbstractModule.__init__(self)
        torch.utils.data.Dataset.__init__(self)
        self.Image = Image
        self.Label = Label
        self.BatchSize = BatchSize
    def SetDevice(self, Device, IsRoot=True):
        self.Device = Device
        self.Log(f"change device to {Device}")
        return self
    def __getitem__(self, Index):
        #Image = torch.from_numpy(self.Image[Index]).to(self.Device)
        #Label = torch.from_numpy(self.Label[Index]).to(self.Device)
        Image = self.Image[Index]
        Label = self.Label[Index]
        return Image, Label
    def __len__(self):
        return self.Image.shape[0]

class DataLoader(DLUtils.train.DataLoaderForEpochBatchTrain):
    # def __init__(self, DataFetcher, BatchSize, BatchNum):
    #     self.DataFetcher = DataFetcher
    #     AbstractModule.__init__(self)
    #     torch.utils.data.DataLoader.__init__(
    #         self, 
    #         dataset=DataFetcher, 
    #         batch_size=BatchSize,
    #         # num_workers=2 # Setting num_workers > 1 might severely slow down speed.
    #     )
    #     self.BatchSize = BatchSize
    #     self._BatchNum = BatchNum
    #     if hasattr(DataFetcher, "Device"):
    #         self.Device = DataFetcher.Device
    #     self.Reset()
    pass
    
def ExtractDataSet(Path, MoveCompressedFile2ExtractFolder=True):
    Path = DLUtils.file.ToAbsPath(Path)

    if DLUtils.file.FileExists(Path): # extract from mnist zip file
        # typical file name: mnist.zip
        CompressFilePath = Path
        FolderPath = DLUtils.file.FolderPathOfFile(Path) # extract to same parent folder
        ExtractFolderPath = FolderPath + "mnist/"
        if DLUtils.file.IsZipFile(Path):
            Path = DLUtils.file.ExtractZipFile(Path, ExtractFolderPath)
        elif DLUtils.file.IsTarFile(Path):
            Path = DLUtils.file.ExtractTarFile(Path, ExtractFolderPath)
        else:
            raise Exception()
        FolderPath = ExtractFolderPath
        if MoveCompressedFile2ExtractFolder:
            DLUtils.file.MoveFile(CompressFilePath, ExtractFolderPath)
    else:
        Sig = True
        if DLUtils.file.FolderExists(Path):
            if DLUtils.file.IsEmptyDir(Path):
                DLUtils.file.RemoveDir(Path)
                Sig = True
            else:
                FolderPath = Path
                Sig = False
        if Sig:
            if Path.endswith(".zip"):
                # .zip file already extracted and put into extracted folder.
                FileName = DLUtils.FileNameFromPath(Path)
                FolderPath = DLUtils.FolderPathOfFile(Path) + "mnist/"
                assert DLUtils.FolderExists(FolderPath)
            else:
                raise Exception()

    for FileName in DLUtils.ListAllFiles(FolderPath):
        # if files in folder is gz file.
        # uncompress them to files with same name, without gz suffix.
        if DLUtils.file.IsGzFile(FolderPath + FileName): 
            assert FileName.endswith(".gz")
            DLUtils.file.ExtractGzFile(FolderPath + FileName)

    return FolderPath

def LoadDataSet(FolderPath, PlotExampleImage=True):
    TrainImagePath = FolderPath + 'train-images-idx3-ubyte'
    TrainLabelPath = FolderPath + 'train-labels-idx1-ubyte'
    ValidationImagePath =  FolderPath + 't10k-images-idx3-ubyte'
    ValidationLabelPath =  FolderPath + 't10k-labels-idx1-ubyte'

    Data = DLUtils.param({
        "Train":{
            # np.ndarray dtype=np.int64 all values are non-negative integers.
            "Image": LoadImage(TrainImagePath), 
            "Label": LoadLabel(TrainLabelPath)
        },
        "Validation":{
            "Image": LoadImage(ValidationImagePath),
            "Label": LoadLabel(ValidationLabelPath)
        }
    })

    if PlotExampleImage:
        ExampleDir = FolderPath + "./example"
        DLUtils.file.EnsureDir(ExampleDir)
        if DLUtils.file.ExistsDir(ExampleDir) \
            and len(DLUtils.file.ListAllFiles(ExampleDir)) < 10:
                ExtractImage(Data=Data, SavePath=ExampleDir)
    return Data