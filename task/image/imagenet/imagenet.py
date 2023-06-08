import DLUtils
import os
import torch
from PIL import Image as Im
import json

# adapted from
# https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
class ImageNet1k(DLUtils.AbstractModule):
    ParamMap = DLUtils.ExpandIterableKey({
        ("DataPath", "DataSetPath"): "Data.Path",
    })
    def __init__(self, Transform=None, **Dict):
        super().__init__(**Dict)
        self.Transform = Transform
    def TrainData(self, **Dict):
        return self.DataLoader("Train",  **Dict)
    def ValidationData(self, **Dict):
        return self.DataLoader("Validation", **Dict)
    def DataLoader(self, Type, **Dict):
        if Type in ["Train"]:
            InList = self.Train.InList
            OutList = self.Train.OutList
        elif Type in ["Validation"]:
            InList = self.Validation.InList
            OutList = self.Validation.OutList
        else:
            raise Exception()
        _DataLoader = DataLoader(
            DataFetcher=DataFetcher(
                InList=InList,
                OutList=OutList,
                Transform=self.Transform,
            ),
            **Dict
        ).Init()
        self.DataLoaderList.add(_DataLoader)
        if hasattr(self, "Device"):
            _DataLoader.SetDevice(self.Device)
        return _DataLoader
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # dataset folder path setting
        assert Param.hasattr("Data.Path")
        Param.Data.Path = DLUtils.file.StandardizePath(Param.Data.Path)
        self.DataPath = Param.Data.Path
        assert self.DataPath.endswith("/")
        
        self.InList = [] # input data sample list
        self.OutList = []
        
        # self.Mode = Param.setdefault("Mode", "Validation")
    
        ClassIndex2ClassCodeNameFilePath = self.DataPath + "class-code-2-class-index.dat"
        
        if not DLUtils.ExistsFile(ClassIndex2ClassCodeNameFilePath):
            self.ClassCode2ClassIndex = {}
            self.ClassIndex2ClassName = {}
            # with open(os.path.join(self.DataPath, "imagenet-2012-1k-class-index.json"), "rb") as f:
            #     ClassCode2ClassIndexJson = json.load(f)
            ClassIndex2ClassCodeJson = DLUtils.JsonFile2Dict(
                # "ClassIndex": ["FolderNameInTrainFolder", "ClassName"]
                self.DataPath + "class-index-2-class-code-name.jsonc", AllowComment=True
            )
            for ClassIndexStr, ClassCodeName in ClassIndex2ClassCodeJson.items():
                ClassIndex = int(ClassIndexStr)
                ClassCode = ClassCodeName[0]
                ClassName = ClassCodeName[1]
                self.ClassCode2ClassIndex[ClassCode] = ClassIndex
                self.ClassIndex2ClassName[ClassIndex] = ClassName
            DLUtils.file.Obj2File(self.ClassCode2ClassIndex, ClassIndex2ClassCodeNameFilePath)
        else:
            self.ClassCode2ClassIndex = DLUtils.File2Obj(ClassIndex2ClassCodeNameFilePath)
        TrainDataDir = os.path.join(self.DataPath, "ILSVRC/Data/CLS-LOC", "train")
        ValDataDir = os.path.join(self.DataPath, "ILSVRC/Data/CLS-LOC", "val")

        ValFileName2ClassCodeFilePath = self.DataPath + "val-file-name-2-class-code.dat"
        if not DLUtils.ExistsFile(ValFileName2ClassCodeFilePath):
            self.ValFileName2ClassCode = DLUtils.JsonFile2Dict(
                DLUtils.file.ChangeFileNameSuffix(ValFileName2ClassCodeFilePath, ".jsonc")
            )
            DLUtils.file.Obj2File(self.ValFileName2ClassCode, ValFileName2ClassCodeFilePath)
        else:
            self.ValFileName2ClassCode = DLUtils.File2Obj(ValFileName2ClassCodeFilePath)
        
        ValFileName2ClassIndexFilePath = self.DataPath + "val-file-name-2-class-index.dat"
        ValFileName2ClassIndexNameFilePath = self.DataPath + "val-file-name-2-class-index-name.dat"
        if not DLUtils.ExistsFile(ValFileName2ClassIndexFilePath):
            self.ValFileName2ClassIndex = {}
            self.ValFileName2ClassName = {}
            self.ValFileName2ClassIndexName = {}
            for ValFileName, ClassCode in self.ValFileName2ClassCode.items():
                ClassIndex = self.ClassCode2ClassIndex[ClassCode]
                ClassName = self.ClassIndex2ClassName[ClassIndex]
                self.ValFileName2ClassIndex[ValFileName] = ClassIndex
                self.ValFileName2ClassName[ValFileName] = ClassName
                self.ValFileName2ClassIndexName[ValFileName] = [ClassIndex, ClassName]
            DLUtils.file.Obj2File(self.ValFileName2ClassIndex, ValFileName2ClassIndexFilePath)
            DLUtils.file.Obj2File(self.ValFileName2ClassIndexName, ValFileName2ClassIndexNameFilePath)
            DLUtils.file.JsonDict2JsonFile(
                self.ValFileName2ClassIndexName,
                DLUtils.file.ChangeFileNameSuffix(ValFileName2ClassIndexNameFilePath, ".jsonc"),
                Mode="Simple"
            )
        else:
            self.ValFileName2ClassIndex = DLUtils.file.File2Obj(ValFileName2ClassIndexFilePath)

        Train = self.Train = DLUtils.Param()
        Validation = self.Validation = DLUtils.Param()
        Train.InList = []
        Train.OutList = []
        Validation.InList = []
        Validation.OutList = []
        
        from functools import cmp_to_key
        TrainFileName2ClassIndexFilePath = self.DataPath + "train-file-name-2-class-index.dat"
        if not DLUtils.file.ExistsFile(TrainFileName2ClassIndexFilePath):
            self.TrainFileName2ClassIndex = {}
            # prepare train data
            for SubFolderName in os.listdir(TrainDataDir):
                ClassCode = SubFolderName
                ClassIndex = self.ClassCode2ClassIndex[ClassCode]
                ImageClassFolderPath = os.path.join(TrainDataDir, SubFolderName)
                TrainFileNameList = DLUtils.ListAllFileNames(ImageClassFolderPath)
                TrainFileNameList.sort(
                    # cmp=DLUtils.utils.NaturalCmp
                    key=cmp_to_key(DLUtils.utils.NaturalCmp)
                )
                for TrainFileName in TrainFileNameList:
                    # TrainFilePath = os.path.join(ImageClassFolderPath, TrainFileName)
                    self.TrainFileName2ClassIndex[SubFolderName + "/" + TrainFileName] = ClassIndex

            DLUtils.Obj2File(self.TrainFileName2ClassIndex, TrainFileName2ClassIndexFilePath)
        else:
            self.TrainFileName2ClassIndex = DLUtils.File2Obj(TrainFileName2ClassIndexFilePath)

        TrainFileNameList = list(self.TrainFileName2ClassIndex.keys())
        Train.InList = [
            os.path.join(TrainDataDir, TrainFileName) for TrainFileName in TrainFileNameList
        ]       
        Train.OutList = list(self.TrainFileName2ClassIndex.values())
        
        # prepare validation data
        # for ImageFileName in DLUtils.ListAllFiles(ValDataDir):
            # ImageClassCode = self.ValFileName2ClassCode[ImageFileName]
            # ImageClassIndex = self.ClassCode2ClassIndex[ImageClassCode]
            # ImageFilePath = os.path.join(ValDataDir, ImageFileName)
            # Validation.InList.append(ImageFilePath)
            # Validation.OutList.append(ImageClassIndex)
        ValFileNameList = list(self.ValFileName2ClassIndex.keys())
        Validation.InList = [
            os.path.join(ValDataDir, ValFileName) for ValFileName in ValFileNameList
        ]            
        Validation.OutList = list(self.ValFileName2ClassIndex.values())
        # Validation.OutList = DLUtils.ToNpArray(Validation.OutList, DataType="int8")
        self.DataLoaderList = set() # dataloader generated by this instance
        return super().Init(IsSuper=False, IsRoot=IsRoot)

class DataFetcher(DLUtils.train.DataFetcherForEpochBatchTrain):
    def __init__(self, InList, OutList, Transform=None):
        self.InList = InList
        self.OutList = OutList
        self.Transform = Transform
        self.DataNum = len(self.InList)
        super().__init__()
    def __len__(self):
        return self.DataNum
    def __getitem__(self, Index):
        Image = Im.open(self.InList[Index]).convert("RGB")
        if self.Transform:
            Image = self.Transform(Image)
        ClassIndex = self.OutList[Index]
        return Image, ClassIndex

# provide batch
class DataLoader(DLUtils.train.DataLoaderForEpochBatchTrain):
    pass

if __name__ == "__main__":
    pass