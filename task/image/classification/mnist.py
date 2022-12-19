import torch
import DLUtils

# def ProcessMNIST(dataset_dir, augment=True, batch_size=64):    
#     transform = transforms.Compose(
#     [transforms.ToTensor()])
#     trainset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=True, download=False)
#     testset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=False, download=False)
#     trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     return trainloader, testloader
from . import ImageClassificationTask

class MNIST(ImageClassificationTask):
    def __init__(self):
        super().__init__()
        self._INIT()
    def _INIT(self):
        Param = self.Param
        Param._CLASS = "DLUtils.task.image.classification.MNIST"
        Param.Train.Num = 60000
        Param.Test.Num = 10000
        return self
    def SetDataPath(self, DataPath, CheckIntegrity=True):
        DLUtils.file.EnsureDir(DataPath)
        Param = self.Param
        Param.DataPath = DataPath
        ConfigFile = DLUtils.file.ParentFolderPath(__file__) + "mnist-folder-config.jsonc"
        Config = DLUtils.file.JsonFile2Param(ConfigFile, SplitKeyByDot=False)
        Data, DataPath = ExtractMNIST(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config)
        self.DataPath = DataPath
        self.IsDataPathBind = True
        return self
    def SetParam(self):
        return self
    def PreprocessData(self, Preprocess, BatchSize=1000):
        assert self.IsDataPathBind
        Data = LoadMNIST(self.DataPath)
        return self
    def TestData(self, BatchSize=64):
        if not hasattr(self, "Data"):
            self.Data = LoadMNIST(self.DataPath)
        Data = self.Data
        return DataLoader(
            DataFetcher(Data.Test.Image, Data.Test.Label).SetLog(self),
            BatchSize=BatchSize
        )
    def TrainData(self, BatchSize=64):
        if not hasattr(self, "Data"):
            self.Data = LoadMNIST(self.DataPath)
        Data = self.Data
        return DataLoader(
            DataFetcher(Data.Train.Image, Data.Train.Label).SetLog(self),
            BatchSize=BatchSize
        )
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

import numpy as np
import struct
def LoadImage(file_name):
    ##   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。##
    ##   file object = open(file_name [, access_mode][, buffering])          ##
    ##   file_name是包含您要访问的文件名的字符串值。                         ##
    ##   access_mode指定该文件已被打开，即读，写，追加等方式。               ##
    ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
    ##   这里rb表示只能以二进制读取的方式打开一个文件                        ##
    binfile = open(file_name, 'rb') 
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取image文件前4个整型数字
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    ##   整个images数据大小为60000*28*28
    bits = num * rows * cols
    ##   读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    ##   关闭文件
    binfile.close()
    ##   转换为[60000,784]型数组
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

def ExtractMNIST(Path):
    Path = DLUtils.file.ToAbsPath(Path)
    if DLUtils.file.IsFile(Path):
        FolderPath = DLUtils.file.FolderPathOfFile(Path)
        if DLUtils.file.IsZipFile(Path):
            Path = DLUtils.file.ZipFile2Folder(Path, FolderPath + "mnist/")
            FolderPath = FolderPath + "mnist/"
        elif DLUtils.file.IsTarFile(Path):
            Path = DLUtils.file.ExtractTarFile(Path, FolderPath + "mnist/")
            FolderPath = FolderPath + "mnist/"
        else:
            raise Exception()

    for FileName in DLUtils.ListAllFiles(FolderPath):
        if DLUtils.file.IsGzFile(FolderPath + FileName):
            assert FileName.endswith(".gz")
            DLUtils.file.ExtractGzFile(FolderPath + FileName)

    Data = LoadMNIST(FolderPath)

    ExampleDir = FolderPath + "ImageExample/"
    if DLUtils.file.ExistsDir(ExampleDir) \
        and len(DLUtils.file.ListAllFiles(ExampleDir)) >= 10:
            ExtractImage(MNISTData=Data, SavePath=ExampleDir)
    return Data, FolderPath

def LoadMNIST(FolderPath):
    TrainImagePath = FolderPath + 'train-images-idx3-ubyte'
    TrainLabelPath = FolderPath + 'train-labels-idx1-ubyte'
    TestImagePath = FolderPath + 't10k-images-idx3-ubyte'
    TestLabelPath = FolderPath + 't10k-labels-idx1-ubyte'

    return DLUtils.param({
        "Train":{
            # np.ndarray dtype=np.int64 all values are non-negative integers.
            "Image": LoadImage(TrainImagePath), 
            "Label": LoadLabel(TrainLabelPath)
        },
        "Test":{
            "Image": LoadImage(TestImagePath),
            "Label": LoadLabel(TestLabelPath)
        }
    })

def ExtractImage(MNISTData, IndexList=None, PlotNum=10, SavePath="./"):
    if IndexList is None:
        IndexList = DLUtils.MultipleRandomIntInRange(0, 70000, PlotNum)
    for Index in IndexList:
        if Index >= 60000:
            Type = "Test"
            Image = MNISTData.Test.Image[Index - 60000]
            Label = MNISTData.Test.Label[Index - 60000]
        else:
            Type = "Train"
            Image = MNISTData.Train.Image[Index]
            Label = MNISTData.Train.Label[Index]
        if len(Image.shape) == 1:
            Image = Image.reshape(28, 28)
        DLUtils.plot.NpArray2ImageFile(
            Image, SavePath + f"{Type} No.{Index} Class:{Label}.png" 
        )
    
class DataFetcher(torch.utils.data.Dataset):
    def __init__(self, Image, Label, BatchSize=None):
        self.Image = Image
        self.Label = Label
        self.BatchSize = BatchSize
    def SetDevice(self, Device):
        self.Device = Device
        self.AddLog(f"change device to {Device}")
        return self
    def __getitem__(self, Index):
        #Image = torch.from_numpy(self.Image[Index]).to(self.Device)
        #Label = torch.from_numpy(self.Label[Index]).to(self.Device)
        Image = self.Image[Index]
        Label = self.Label[Index]
        return Image, Label
    def __len__(self):
        return self.Image.shape[0]
    def SetLog(self, Log):
        self.Log = Log
        return self
    def AddLog(self, Content):
        self.Log.AddLog(Content)
        return self

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, DataFetcher, BatchSize):
        self.DataFetcher = DataFetcher
        super().__init__(
            dataset=DataFetcher, 
            batch_size=BatchSize, 
            # num_workers=2 # Setting num_workers > 1 might severely slow down speed.
        )
    def Get(self, BatchIndex):
        Image, Label = next(iter(self))
        return Image.to(self.Device), Label.to(self.Device)
    def SetDevice(self, Device):
        self.DataFetcher.SetDevice(Device)
        self.Device = Device
        return self
    
