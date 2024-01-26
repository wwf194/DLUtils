import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
import DLUtils
from DLUtils.module import AbstractModule
import struct
from .. import ImageClassTask

def CalculateStatistics(DataPath="~/Data/mnist", SavePath=None):
    Dataset = MNISTHandler().SetDataPath(DataPath).Init()
    Stat = {
        "Train": DLUtils.math.NpArrayStat(
            DLUtils.ToNpArray(Dataset.Data.Train.Image) / 255.0
        ),
        "Test": DLUtils.math.NpArrayStat(
            DLUtils.ToNpArray(Dataset.Data.Test.Image) / 255.0
        ),
    }
    
    if SavePath is None:
        SavePath = DLUtils.CurrentDirPath(__file__) + "mnist-statistics.jsonc"
    DLUtils.file.JsonDict2JsonFile(Stat, SavePath)

from ... import TaskHandler
class MNISTHandler(TaskHandler):
    ClassNum = 10
    def __init__(self):
        """
        mnist Dataset contains 50000 train images and 10000 test images.
        each image contains a hand-written digit in [0, 1...9]
        each image has size 28x28 pixels.
        each pixle has 1 color channel.
        """
        super().__init__()
    def AfterLoadOrInit(self):
        Param = self.Param
        Param.TrainSet.Num = 50000
        Param.TestSet.Num = 10000
        return super().AfterLoadOrInit()
    def BeforeToFileOrClear(self):
        delattr(self, "Data")
        return super().BeforeToFileOrClear()
    def GetSubDatasetName(self):
        return ["Train", "Test"]
    def GetTrainSet(self):
        Param = self.Param
        return MNISTDataset(
            Type="Train",
            DataNum=Param.TrainSet.Num,
            Parent=self
        )
    def GetTestSet(self):
        Param = self.Param
        return MNISTDataset(
            Type="Test",
            DataNum=Param.TestSet.Num,
            Parent=self
        )
    def GetTrainImage(self, Index):
        # assert self.IsBuild()
        return self.TrainImage[Index]
    def GetTrainLabel(self, Index):
        # assert self.IsBuild()
        return self.TrainLabel[Index]
    def GetTestImage(self, Index):
        # assert self.IsBuild()
        return self.TestImage[Index]
    def GetTestLabel(self, Index):
        # assert self.IsBuild()
        return self.TestLabel[Index]
    def Build(self, *List, **Dict):
        Param = self.Param

        DatasetDirPath = None
        assert Param.hasattr("DataDirPathOrigin") or Param.hasattr("ZipFilePathOrigin")

        # unzip file if needed
        if Param.hasattr("DataDirPathOrigin"):
            DataDirPathOrigin = Param.DataDirPathOrigin
            DatasetDirPathSuspect = DLUtils.StandardizeDirPath(DataDirPathOrigin)
            if self.CheckIntegrityOfDatasetDir(DatasetDirPathSuspect):
                DatasetDirPath = DatasetDirPathSuspect
            
        if DatasetDirPath is None and Param.hasattr("ZipFilePathOrigin"):
            ZipFilePathOrigin = Param.ZipFilePathOrigin
            ZipFilePath = DLUtils.StandardizeFilePath(ZipFilePathOrigin)
            ZipFileDirPath = DLUtils.DirPathOfFile(ZipFilePath)
            DatasetDirPathSuspect = ZipFileDirPath + "mnist/"
            
            # check whether dataset files have already been extracted to a folder
            if DLUtils.ExistsDir(DatasetDirPathSuspect):
                if self.CheckIntegrityOfDatasetDir(DatasetDirPathSuspect):
                    DatasetDirPath = DatasetDirPathSuspect

            # extract dataset files from .zip file to to a folder
            
            if DatasetDirPath is None:
                assert DLUtils.ExistsFile(ZipFilePath)
                DatasetDirPath = self.ExtractDatasetFromZipFile(ZipFilePath)
        
        assert DatasetDirPath is not None
        assert self.CheckIntegrityOfDatasetDir(DatasetDirPath)
        
        # load mnist data from files
        self.Dataset = self.LoadDataset(DatasetDirPath)
        if self.IsVerbose():
            print("Loaded mnist dataset from (%s)"%(DatasetDirPath))
        
        self.TrainImage = self.Dataset.Train.Image
        self.TrainLabel = self.Dataset.Train.Label
        self.TestImage = self.Dataset.Test.Image
        self.TestLabel = self.Dataset.Test.Label

        return super().Build(*List, **Dict)
    def SetDataFilePath(self, ZipFilePath):
        """
        provide path of .zip file of mnist dataset
        """
        self.Param.ZipFilePathOrigin = ZipFilePath
        return self
    SetZipFilePath = SetDataFilePath
    def SetDataDirPath(self, DataDirPath, CheckIntegrity=True):
        Param = self.Param
        Param.DataDirPathOrigin = DataDirPath
        return self
        ConfigFilePath = DLUtils.file.ParentFolderPath(__file__) + "mnist-folder-config.jsonc"
        DLUtils.CheckFileExists(ConfigFilePath)
        Config = DLUtils.file.JsonFile2Param(ConfigFilePath, SplitKeyByDot=False)
        DataPath = ExtractDataset(DataPath)
        if CheckIntegrity:
            assert DLUtils.file.CheckIntegrity(DataPath, Config)
        self.DataPath = DataPath
        self.IsDataPathBind = True
        self.Config = Config
        return self
    def TestData(self, BatchSize=64):
        return self.DataLoader("Test", BatchSize)
    def TrainData(self, BatchSize=64):
        return self.DataLoader("Train", BatchSize)

    # get a dataloader
    def GetDataLoader(self, Type, BatchSize=64):
        if not hasattr(self, "Data"):
            self.Data = LoadDataset(self.DataPath)
        Data = self.Data
        if Type in ["Train"]:
            Data = Data.Train
            BatchNum=self.TrainBatchNum(BatchSize)
        else:
            Data = Data.Test
            BatchNum=self.TestBatchNum(BatchSize)
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
    def AllTrainData(self):
        return DLUtils.param(self.Data.Train)
    # def ExtractDataset(self, DataDirPath, MoveCompressedFile2ExtractFolder=True):
    #     DataDirPath = DLUtils.file.ToAbsPath(DataDirPath)

    #     if DLUtils.file.FileExists(Path): # extract from mnist zip file
    #         # typical file name: mnist.zip
    #         CompressFilePath = Path
    #         FolderPath = DLUtils.file.FolderPathOfFile(Path) # extract to same parent folder
    #         ExtractFolderPath = FolderPath + "mnist/"
    #         if DLUtils.file.IsZipFile(Path):
    #             Path = DLUtils.file.ExtractZipFile(Path, ExtractFolderPath)
    #         elif DLUtils.file.IsTarFile(Path):
    #             Path = DLUtils.file.ExtractTarFile(Path, ExtractFolderPath)
    #         else:
    #             raise Exception()
    #         FolderPath = ExtractFolderPath
    #         if MoveCompressedFile2ExtractFolder:
    #             DLUtils.file.MoveFile(CompressFilePath, ExtractFolderPath)
    #     else:
    #         Sig = True
    #         if DLUtils.file.FolderExists(Path):
    #             if DLUtils.file.IsEmptyDir(Path):
    #                 DLUtils.file.RemoveDir(Path)
    #                 Sig = True
    #             else:
    #                 FolderPath = Path
    #                 Sig = False
    #         if Sig:
    #             if Path.endswith(".zip"):
    #                 # .zip file already extracted and put into extracted folder.
    #                 FileName = DLUtils.FileNameFromPath(Path)
    #                 FolderPath = DLUtils.FolderPathOfFile(Path) + "mnist/"
    #                 assert DLUtils.FolderExists(FolderPath)
    #             else:
    #                 raise Exception()
    #     FolderPath = DLUtils.StandardizeFolderPath(FolderPath)
    #     for FileName in DLUtils.ListAllFiles(FolderPath):
    #         # if files in folder is gz file.
    #         # uncompress them to files with same name, without gz suffix.
    #         if DLUtils.file.IsGzFile(FolderPath + FileName): 
    #             assert FileName.endswith(".gz")
    #             DLUtils.file.ExtractGzFile(FolderPath + FileName)
    #     return FolderPath
    def LoadImage(self, FilePath):
        FileBinary = open(FilePath, 'rb') 
        Buf = FileBinary.read()
        # read 4 integers from Buf
        Magic, ImageNum, RowNum, ColNum = struct.unpack_from('>IIII', Buf, 0)
        BitNum = ImageNum * RowNum * ColNum
        # read image data from Buf
        ImageList = struct.unpack_from('>' + str(BitNum) + 'B', Buf, struct.calcsize('>IIII'))
        FileBinary.close()
        ImageList = np.reshape(ImageList, [ImageNum, RowNum * ColNum]) # [60000, 784]
        # np.int64 -> np.uint8
        ImageList = ImageList.astype(np.uint8).reshape(-1, 28, 28)
        return ImageList
    def LoadLabel(self, FilePath):
        FileBinary = open(FilePath, 'rb')    
        Buf = FileBinary.read()
        # read 2 integer from Buf
        Magic, DataNum = struct.unpack_from('>II', Buf, 0) 
        # read label data from Buf
        LabelList = struct.unpack_from('>' + str(DataNum) + "B", Buf, struct.calcsize('>II'))
        FileBinary.close()
        LabelList = np.reshape(LabelList, [DataNum])
        # np.int64 -> np.uint8
        LabelList = LabelList.astype(np.uint8)
        return LabelList
    def ExtractDatasetFromZipFile(self, ZipFilePath):
        ZipFileDirPath = DLUtils.DirPathOfFile(ZipFilePath)
        DatasetDirPathTarget = ZipFileDirPath + "mnist/"
        DLUtils.file.ExtractZipFile(ZipFilePath, DatasetDirPathTarget)
        
        # check all 4 required .gz files have been extracted
        TrainImageGzFilePath = DatasetDirPathTarget + 'train-images-idx3-ubyte.gz'
        TrainLabelGzFilePath = DatasetDirPathTarget + 'train-labels-idx1-ubyte.gz'
        TestImageGzFilePath = DatasetDirPathTarget + 't10k-images-idx3-ubyte.gz'
        TestLabelGzFilePath = DatasetDirPathTarget + 't10k-labels-idx1-ubyte.gz'
        DLUtils.CheckAllFilesExist(
            TrainImageGzFilePath, TrainLabelGzFilePath,
            TestImageGzFilePath, TestLabelGzFilePath
        )

        # extract all 4 .gz files
        TrainImageFilePath = DatasetDirPathTarget + 'train-images-idx3-ubyte'
        TrainLabelFilePath = DatasetDirPathTarget + 'train-labels-idx1-ubyte'
        TestImageFilePath = DatasetDirPathTarget + 't10k-images-idx3-ubyte'
        TestLabelFilePath = DatasetDirPathTarget + 't10k-labels-idx1-ubyte'
        DLUtils.file.ExtractGzFile(TrainImageGzFilePath, TrainImageFilePath)
        DLUtils.file.ExtractGzFile(TrainLabelGzFilePath, TrainLabelFilePath)
        DLUtils.file.ExtractGzFile(TestImageGzFilePath, TestImageFilePath)
        DLUtils.file.ExtractGzFile(TestLabelGzFilePath, TestLabelFilePath)
        
        # clean .gz files
        DLUtils.file.RemoveFiles(
            TrainImageGzFilePath, TrainLabelGzFilePath,
            TestImageGzFilePath, TestLabelGzFilePath
        )
        return DatasetDirPathTarget
    def CheckIntegrityOfDatasetDir(self, DataDirPath):
        DataDirPath = DLUtils.StandardizeDirPath(DataDirPath)
        if not DLUtils.DirExists(DataDirPath):
            return False
        return DLUtils.AllFilesExist(
            TrainImageFilePath = DataDirPath + 'train-images-idx3-ubyte',
            TrainLabelFilePath = DataDirPath + 'train-labels-idx1-ubyte',
            TestImageFilePath = DataDirPath + 't10k-images-idx3-ubyte',
            TestLabelFilePath = DataDirPath + 't10k-labels-idx1-ubyte'
        )
    def CheckIntegrityOfExtractedDir(self, DataDirPath):
        DataDirPath = DLUtils.CheckDirExists(DataDirPath)
        return DLUtils.CheckAllFilesExist(
            TrainImageFilePath = DataDirPath + 'train-images-idx3-ubyte.gz',
            TrainLabelFilePath = DataDirPath + 'train-labels-idx1-ubyte.gz',
            TestImageFilePath = DataDirPath + 't10k-images-idx3-ubyte.gz',
            TestLabelFilePath = DataDirPath + 't10k-labels-idx1-ubyte.gz'
        )
    def LoadDataset(self, DatasetDirPath, PlotExampleImage=False):
        DatasetDirPath = DLUtils.CheckDirExists(DatasetDirPath)
        self.CheckIntegrityOfDatasetDir(DatasetDirPath)

        TrainImageFilePath = DatasetDirPath + 'train-images-idx3-ubyte'
        TrainLabelFilePath = DatasetDirPath + 'train-labels-idx1-ubyte'
        TestImageFilePath = DatasetDirPath + 't10k-images-idx3-ubyte'
        TestLabelFilePath = DatasetDirPath + 't10k-labels-idx1-ubyte'

        Dataset = DLUtils.param({
            "Train":{
                # np.ndarray dtype=np.int8 all values are non-negative integers.
                "Image": self.LoadImage(TrainImageFilePath), 
                "Label": self.LoadLabel(TrainLabelFilePath)
            },
            "Test":{
                "Image": self.LoadImage(TestImageFilePath),
                "Label": self.LoadLabel(TestLabelFilePath)
            }
        })

        if PlotExampleImage:
            PlotDirPath = DatasetDirPath + "example/"
            if DLUtils.file.ExistsDir(PlotDirPath) \
                and len(DLUtils.file.ListAllFiles(PlotDirPath)) < 10:
                self.PlotExampleImage(Dataset=Dataset, PlotNum=10, SaveDirPath=PlotDirPath)
        return Dataset
    def PlotExampleImage(self, Dataset=None, IndexList=None, PlotNum=10, SaveDirPath="./"):
        """
        Index [0, 600000) is train data.
        Index [60000, 70000) is test data.
        """
        if Dataset is None:
            assert self.HasBuild()
            Dataset = self.Dataset
        SaveDirPath = DLUtils.EnsureDir(SaveDirPath)
        if IndexList is None:
            IndexList = DLUtils.math.MultiRandomIntInRange(PlotNum, 0, 70000)
        for Index in IndexList:
            if Index >= 60000:
                Type = "Test"
                Image = Dataset.Test.Image[Index - 60000]
                Label = Dataset.Test.Label[Index - 60000]
            else:
                Type = "Train"
                Image = Dataset.Train.Image[Index]
                Label = Dataset.Train.Label[Index]
            
            # Image: (28, 28), uint8.
            DLUtils.plot.NpArrayUInt8ToImageFile(
                Image, SaveDirPath + f"{Type}-No{Index}-Label{Label}.png" 
            )

class MNISTDataset:
    """
    Pytorch framework requires torch.utils.data.Dataset to implement __len__ and __item__ method.
        __len__ tells how many data samples are in Dataset
        __item__ allows retrieving i-th sample by Dataset[i]
    """
    def __init__(self, Type, DataNum, Parent=None):
        self.Type = Type
        self.DataNum = DataNum
        if Parent is not None:
            self.SetParent(Parent)
    def SetParent(self, Parent: MNISTHandler):
        self.Parent = Parent
        if self.Type in ["train", "Train"]:
            self.GetImage = lambda Index:self.Parent.GetTrainImage(Index)
            self.GetLabel = lambda Index:self.Parent.GetTrainLabel(Index)
        elif self.Type in ["test", "Test"]:
            self.GetImage = lambda Index:self.Parent.GetTestImage(Index)
            self.GetLabel = lambda Index:self.Parent.GetTestLabel(Index)
        else:
            raise Exception()
        return self
    def __len__(self):
        return self.DataNum
    def __getitem__(self, Index):
        # assert Index < self.DataNum
        return self.GetImage(Index), self.GetLabel(Index)

# class DataFetcher(torch.utils.data.Dataset, AbstractModule):
#     def __init__(self, Image, Label, BatchSize=None):
#         AbstractModule.__init__(self)
#         torch.utils.data.Dataset.__init__(self)
#         self.Image = Image
#         self.Label = Label
#         self.BatchSize = BatchSize
#     def SetDevice(self, Device, IsRoot=True):
#         self.Device = Device
#         self.Log(f"change device to {Device}")
#         return self
#     def __getitem__(self, Index):
#         # Image = torch.from_numpy(seexamplelf.Image[Index]).to(self.Device)
#         # Label = torch.from_numpy(self.Label[Index]).to(self.Device)
#         Image = self.Image[Index]
#         Label = self.Label[Index]
#         return Image, Label
#     def __len__(self):
#         return self.Image.shape[0]

# class DataLoader(DLUtils.train.DataLoaderForEpochBatchTrain):
#     def __init__(self, DataFetcher, BatchSize, BatchNum):
#         self.DataFetcher = DataFetcher
#         AbstractModule.__init__(self)
#         torch.utils.data.DataLoader.__init__(
#             self, 
#             Dataset=DataFetcher, 
#             batch_size=BatchSize,
#             # num_workers=2 # Setting num_workers > 1 might severely slow down speed.
#         )
#         self.BatchSize = BatchSize
#         self._BatchNum = BatchNum
#         if hasattr(DataFetcher, "Device"):
#             self.Device = DataFetcher.Device
#         self.Reset()
#     pass