import os
import torch
from PIL import Image as Im
import json

import DLUtils

# adapted from
# https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
class ImageNet1k(DLUtils.AbstractModule):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("DataSetPath", "DataPath"): "Data.Path",
    })
    def __init__(self, Transform=None, **Dict):
        super().__init__(**Dict)
        self.Transform = Transform
    def DataLoader(self, **Dict):
        _DataLoader = DataLoader(
            DataFetcher=DataFetcher(
                InList=self.InList,
                OutList=self.OutList,
                Transform=self.Transform
            ),
            **Dict
        ).Init()
        self.DataLoaderList.add(_DataLoader)
        return _DataLoader
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # dataset folder path setting
        assert Param.hasattr("Data.Path")
        Param.Data.Path = DLUtils.StandardizePath(Param.Data.Path)
        self.DataPath = Param.Data.Path
        
        self.InList = [] # input data sample list
        self.OutList = []
        
        self.Mode = Param.setdefault("Mode", "Validation")
        
        self.ClassCode2ClassIndex = {}
        # with open(os.path.join(self.DataPath, "imagenet-2012-1k-class-index.json"), "rb") as f:
        #     ClassCode2ClassIndexJson = json.load(f)
        ClassCode2ClassIndexJson = DLUtils.JsonFile2Dict(
            self.DataPath + "imagenet-2012-1k-class-index.json"
        )
        for ClassIndexStr, ClassCode in ClassCode2ClassIndexJson.items():
            self.ClassCode2ClassIndex[ClassCode[0]] = int(ClassIndexStr)
        
        # with open("ILSVRC2012_val_labels.json", "rb") as f:
        #     self.ValImageName2ClassCode = json.load(f)
        self.ValImageName2ClassCode = DLUtils.JsonFile2Dict(
            os.path.join(self.DataPath, "ILSVRC2012_val_labels.json")
        )
        
        if self.Mode in ["train", "Train"]:
            self.Mode = "Train"
            DataDir = os.path.join(self.DataPath, "ILSVRC/Data/CLS-LOC", "train")
        elif self.Mode in ["val", "validation", "Validation"]:
            self.Mode = "Validation"
            DataDir = os.path.join(self.DataPath, "ILSVRC/Data/CLS-LOC", "val")
        else:
            raise Exception(self.Mode)
        
        if self.Mode == "Train":
            for ImageFolderName in os.listdir(DataDir):
                ImageClassCode = ImageFolderName
                ImageClassIndex = self.ClassCode2ClassIndex[ImageClassCode]
                ImageClassFolderPath = os.path.join(DataDir, ImageFolderName)
                for ImageFileName in DLUtils.ListAllFileNames(ImageClassFolderPath):
                    ImageFilePath = os.path.join(ImageClassFolderPath, ImageFileName)
                    self.InList.append(ImageFilePath)
                    self.OutList.append(ImageClassIndex)
        elif self.Mode == "Validation":
            for ImageFileName in DLUtils.ListAllFiles(DataDir):
                ImageClassCode = self.ValImageName2ClassCode[ImageFileName]
                ImageClassIndex = self.ClassCode2ClassIndex[ImageClassCode]
                ImageFilePath = os.path.join(DataDir, ImageFileName)
                self.InList.append(ImageFilePath)
                self.OutList.append(ImageClassIndex)
        else:
                raise Exception()
        self.DataLoaderList = set() # dataloader generated by this instance
        return super().Init(IsSuper=False, IsRoot=IsRoot)
class DataFetcher(torch.utils.data.Dataset):
    def __init__(self, InList, OutList, Transform=None):
        self.InList = InList
        self.OutList = OutList
        self.Transform = Transform
    def __len__(self):
        return len(self.InList)
    def __getitem__(self, Index):
        Image = Im.open(self.InList[Index]).convert("RGB")
        if self.Transform:
            Image = self.Transform(Image)
        ClassIndex = self.OutList[Index]
        return Image, ClassIndex

# provide batch
class DataLoader(DLUtils.train.DataLoaderForEpochBatchTrain):
    pass