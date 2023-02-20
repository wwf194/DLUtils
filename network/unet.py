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
        _Dict = {}
        for Key, Value in Dict.items():
            if Key in ["BatchNorm"]:
                Param.BatchNorm.Enable = bool(Value)
            elif Key in ["Block.Num", "BlockNum"]:
                Param.Block.Num = Value
            else:
                _Dict[Key] = Value
        return super().SetParam(**_Dict)
    def Receive(self, Image):
        # Image: [BatchSize, ChannelNum, Width, Height]
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Block.setdefault("Num", 5)
        BaseNum = Param.Base.setdefault("Num", 64)
        if not self.IsLoad():
            DownPath = UNetDownPath().SetParam(
                InNum = Param.In.Num,
                BaseNum = BaseNum,
                BlockNum = Param.Block.Num + 1,
                OutNum = Param.Out.Num    
            )
            MidNum = Param.Base.Num * (2 ** Param.Block.Num)
            UpPath = UNetUpPath().SetParam(
                InNum = MidNum,
                BlockNum = Param.Block.Num,
                OutNum = BaseNum  
            )

            OutputLayer = DLUtils.network.Conv2D(
                InNum=BaseNum, OutNum=Param.Out.Num,
                Padding="KeepFeatureMapHeightWidth",
                KernelSize=3
            )
            self.AddSubModule(
                DownPath=DownPath, 
                UpPath=UpPath,
                OutputLayer=OutputLayer
            )
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def Receive(self, In):
        """
        In : Images of shape [BatchNum, ChannelNum, Height, Width]
        """
        Mid = self.DownPath(In)
        SkipInList = Mid["SkipOut"]
        SlipInListReversed = list(reversed(SkipInList))

        # DownIn = Mid["Down"]
        Out = self.UpPath(
            DownIn=SlipInListReversed[0], 
            SkipInList=SlipInListReversed[1:]
        )
        Out = self.OutputLayer(Out)
        return Out

from .ModuleGroup import ModuleSeries
class UNetDownPath(ModuleSeries):
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
    def Init(self, IsSuper=False, IsRoot=True):
        if not self.IsLoad():
            Param = self.Param
            InNum = Param.In.Num
            OutNum = Param.Base.Num
            Param.NumList = []
            for Index in range(Param.Block.Num):
                self.AddSubModule(
                    "Block%d"%Index,
                    UNetDownSampleBlock(
                        InNum=InNum, OutNum=OutNum,
                        Padding="KeepFeatureMapHeightWidth",
                        KernelSize=3, BatchNorm=True
                    )
                )
                InNum = OutNum
                OutNum = InNum * 2
        return super().Init(IsSuper=False, IsRoot=IsRoot)
class UNetUpPath(ModuleSeries):
    def Receive(self, DownIn, SkipInList):
        for Index, Block in enumerate(self.ModuleList):
            SkipIn = SkipInList[Index] # Already reversed
            Up = Block(DownIn=DownIn, SkipIn=SkipIn)
            DownIn = Up
        return Up
    def Init(self, IsSuper=False, IsRoot=True):
        if not self.IsLoad():
            Param = self.Param
            InNum = Param.In.Num
            OutNum = InNum // 2
            for Index in range(Param.Block.Num):
                self.AddSubModule(
                    "Block%d"%Index,
                    UNetUpSampleBlock(
                        InNum=InNum, OutNum=OutNum,
                        Padding=1, KernelSize=2
                    )
                )
                InNum = OutNum
                OutNum = InNum // 2
        return super().Init(IsSuper=True, IsRoot=IsRoot)
class UNetDownSampleBlock(DLUtils.module.AbstractNetwork):
    def __init__(self, **Dict):
        super().__init__()
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
            elif Key in ["Padding"]:
                Param.Padding = Value
            elif Key in ["Stride"]:
                Param.Stride = Value
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
            Conv = DLUtils.network.ModuleSeries()
            
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
            )
            if Param.BatchNorm.Enable:
                Conv.AddSubModule(
                    "Norm1", DLUtils.norm.BatchNorm2D(FeatureNum=Param.Out.Num)
                )
            Conv.AddSubModule(
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
                    "Norm2", DLUtils.norm.BatchNorm2D(FeatureNum=Param.Out.Num)
                )
            self.AddSubModule("Conv", Conv)
            self.AddSubModule("DownPool", DLUtils.network.MaxPool2D(KernelSize=2))
        if not hasattr(self, "Norm"):
            self.Norm = lambda x:x
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class UNetUpSampleBlock(ModuleSeries):
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
            elif Key in ["Padding"]:
                Param.Padding = Value
            elif Key in ["Stride"]:
                Param.Stride = Value
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
                "UpSample", DLUtils.network.UpConv2D(
                    # OutNum == InNum // 2
                    Param.In.Num, Param.Out.Num,
                    KernelSize=Param.Kernel.Size,
                    Stride=Param.Stride,
                    Padding=Param.Padding
                )
            )
            Param.Conv.setdefault("Stride", 1)
            self.AddSubModule(
                "Conv", DLUtils.network.Conv2D(
                    # OutNum == InNum // 2
                    Param.In.Num, Param.Out.Num,
                    KernelSize=Param.Kernel.Size,
                    Stride=Param.Conv.Stride,
                    Padding=Param.Padding
                )
            )
            Param.delattr("Conv")
        return super().Init(IsSuper, IsRoot)

_SetParamMap = DLUtils.IterableKeyToElement({
    ("In", "InNum", "In.Num"): "In.Num",
    ("Out", "OutNum", "Out.Num"): "Out.Num",
    ("GroupNum", "NumGroup", "Group.Num"): "Group.Num",
    "BlockNum": "Block.Num",
    "BaseNum": "Base.Num",

})
UNet.SetParamMap = _SetParamMap
UNetDownPath.SetParamMap = _SetParamMap
UNetUpPath.SetParamMap = _SetParamMap
UNetDownSampleBlock.SetParamMap = _SetParamMap
UNetUpSampleBlock.SetParamMap = _SetParamMap