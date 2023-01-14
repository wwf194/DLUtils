import torch
import numpy as np

import DLUtils
class UNet(DLUtils.module.AbstractNetwork):
    def __init__(self, **Dict):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            InChannelNum (int): number of input channels
            OutChannelNum (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the Out.
                            This de introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in Dict.items():
            if Key in ["InNum", "InputNum"]:
                Param.Channel.In.Num = Value
            elif Key in ["OutNum", "OutputNum"]:
                Param.Channel.Out.Num = Value
            elif Key in ["BatchNorm"]:
                Param.BatchNorm.Enable = bool(Value)
            elif Key in ["Block.Num", "BlockNum"]:
                Param.Block.Num = Value
            else:
                raise Exception()
        return super().SetParam(**Dict)
    def Receive(self, Image):
        # Image: [BatchSize, ChannelNum, Width, Height]
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Block.setdefault("Num", 5)
        Param.Channel.Base.setdefault("Num", 64)
        if not self.IsLoad():
            DownSamplePath = UNetDownPath()
            InNum = Param.Channel.In.Num
            OutNum = Param.Channel.Base.Num
            Param.Channel.NumList = []
            for Index in range(Param.Block.Num):
                DownSamplePath.AddSubModule(
                    "Block%d"%Index,
                    UNetDownSampleBlock(
                        InNum=InNum, OutNum=OutNum,
                        Padding="KeepFeatureMapHeightWidth",
                        KernelSize=3, BatchNorm=True
                    )
                )
                InNum = OutNum
                OutNum = InNum * 2
            UpSamplePath = UNetUpPath()
            OutNum = InNum // 2
            for Index in range(Param.Block.Num):
                UpSamplePath.AddSubModule(
                    "Block%d"%Index,
                    UNetUpSampleBlock(
                        InNum=InNum, OutNum=OutNum,
                        Padding=1, KernelSize=2
                    )
                )
                InChannelNum = OutNum
                InNum = OutNum
                OutNum = InNum // 2
            OutputLayer = DLUtils.network.Conv2D(
                InNum=OutNum, OutNum=Param.Out.Num,
                Padding="KeepFeatureMapHeightWidth",
                KernelSize=3
            )
            self.AddSubModule(
                DownSamplePath=DownSamplePath, 
                UpSamplePath=UpSamplePath,
                OutputLayer=OutputLayer
            )
        super().Init(IsSuper=True, IsRoot=IsRoot)
    def Receive(self, In):
        Mid = self.DownSamplePath(In)
        SkipInList = Mid["SkipOut"]
        SkipInList = reversed(SkipInList)
        DownIn = Mid["Down"]
        Out = self.UpSamplePath(DownIn=DownIn, SkipInList=SkipInList)
        Out = self.OutputLayer(Out)
        return Out

from .ModuleSquence import ModuleSequence
class UNetDownPath(ModuleSequence):
    def Receive(self, In):
        Down = In
        SkipOut = []
        for Index, Block in enumerate(self.ModuleList):
            Out = Block(Down)
            SkipOut.append(Out["Skip"])
            Down = Out["Down"]
        return {
            "SkipOut": SkipOut,
            "Down": Down
        }
class UNetUpPath(ModuleSequence):
    def Receive(self, DownIn, SkipInList):
        for Index, Block in enumerate(self.ModuleList):
            SkipIn = SkipInList[Index] # Already reversed
            Up = Block(DownIn=Up, SkipIn=SkipIn)
            DownIn = Up
        return Up

class UNetDownSampleBlock(DLUtils.module.AbstractNetwork):
    def __init__(self, **Dict):
        super().__init__()
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["InNum", "In.Num"]:
                Param.Kernel.In.Num = Value
            elif Key in ["OutNum", "Out.Num"]:
                Param.Kernel.Out.Num = Value
            elif Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
            elif Key in ["Padding"]:
                Param.Padding = Value
            elif Key in ["Stride"]:
                Param.Stride = Value
            elif Key in ["GroupNum", "NumGroup", "Group.Num"]:
                Param.Group.Num = Value
            elif Key in ["BatchNorm"]:
                Param.BatchNorm.Enable = bool(Value)
            else:
                _Dict[Key] = Value
        super().SetParam(**_Dict)
        return self
    def Receive(self, In):
        Out = self.Conv(In)
        Down = self.DownPool(Out)
        return {
            "Skip": Out,
            "Down": Down
        }
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not self.IsLoad():
            Param.setdefault("Stride", 1)
            Param.setdefault("Padding", 1)
            Param.Kernel.setdefault("Size", 3)
            Param.BatchNorm.setdefault("Enable", False)
            Conv = DLUtils.network.ModuleSequence()
            
            Conv.AddSubModule(
                "Conv1", 
                DLUtils.network.Conv2D(
                    Param.In.Num, Param.Out.Num,
                    KernelSize=Param.Kernel.Size,
                    Padding=Param.Padding
                )
            ).AddSubModule(
                "NonLinear1", 
                DLUtils.NonLinear.ReLU()
            ).AddSubModule(
                "Conv2", DLUtils.network.Conv2D(
                    Param.Out.Num, Param.Out.Num,
                    KernelSize=Param.Kernel.Size,
                    Padding=Param.Padding
                )
            ).AddSubModule(
                "NonLinear2", DLUtils.NonLinear.ReLU()
            )
            if Param.BatchNorm.Enable:
                Conv.AddSubModule(
                    "Norm", DLUtils.norm.BatchNorm2D()
                )
            
            self.AddSubModule("Conv", Conv)
            self.AddSubModule("DownPool", DLUtils.network.MaxPool2D(KernelSize=2))
        if not hasattr(self, "Norm"):
            self.Norm = lambda x:x
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class UNetUpSampleBlock(ModuleSequence):
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["InNum", "In.Num"]:
                Param.Kernel.In.Num = Value
            elif Key in ["OutNum", "Out.Num"]:
                Param.Kernel.Out.Num = Value
            elif Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
            elif Key in ["Padding"]:
                Param.Padding = Value
            elif Key in ["Stride"]:
                Param.Stride = Value
            elif Key in ["GroupNum", "NumGroup", "Group.Num"]:
                Param.Group.Num = Value
            else:
                _Dict[Key] = Value
        super().SetParam(**_Dict)
        return self
    def Receive(self, DownIn, SkipIn):
        In = self.UpSample(DownIn)
        _SkipIn = self.GetFeatureMapCenterRegion(SkipIn, In.size(2), In.size(3))
        Mid = torch.cat([In, _SkipIn], dim=1) # [BatchNum, ChannelNumSkipIn + ChannelNumDownUpIn, Height, Width]
        Up = self.Conv(Mid)
        return Up
    def GetFeatureMapCenterRegion(self, FeatureMap, Height, Width):
        BatchNum, ChannelNum, FeatureMapHeight, FeatureMapwidth = FeatureMap.size()
        YStart = (FeatureMapHeight - Height) // 2
        XStart = (FeatureMapwidth - Width) // 2
        YEnd = YStart + Height
        XEnd = XStart + Width
        return FeatureMap[:, :, YStart:YEnd, XStart:XEnd]
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Kernel.setdefault("Size", 2)
        Param.setdefault("Stride", 2)
        Param.setdefault("Padding", 1)
        if not self.IsLoad():
            self.AddSubModule(
                "UpSample", DLUtils.network.Conv2D(
                    Param.In.Num, Param.Out.Num,
                    KernelSize=Param.Kernel.Size,
                    Stride=Param.Stride,
                    Padding = Param.Padding
                )
            )
            self.AddSubModule(
                "Conv", DLUtils.network.Conv2D(Param.In.Num, Param.Out.Num)
            )
        return super().Init(IsSuper, IsRoot)