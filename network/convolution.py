import torch
import torch.nn.functional as F
import numpy as np

import DLUtils
ConvParamMap = DLUtils.IterableKeyToElement({
    ("In.Num"): "InNum", # input feature map num
    ("Out.Num"): "OutNum", # output feature map num
})

def GetParamMapDefaultConv():
    return DLUtils.ExpandIterableKey({
        ("InNum", "InChannelNum"): "In.Num",
        ("OutNum", "OutChannelNum"): "Out.Num",
        ("Stride"): "Stride",
        ("KernelSize", "Kernel.Size"): "Kernel.Size",
        ("Padding"): "Padding.Value",
        ("GroupNum", "NumGroup", "Group.Num"): "Group.Num",
        ("OutputPadding", "OutPadding"): "Padding.Additional",
        ("Padding"): "Padding.Value",
        ("NonLinear"): "NonLinear.Type"
    })
ParamMapDefaultConv = GetParamMapDefaultConv()

class Conv2D(DLUtils.module.AbstractNetwork):
    ParamMap = GetParamMapDefaultConv()
    def __init__(self, InNum=None, OutNum=None, Stride=None, **Dict):
        super().__init__()
        Param = self.Param
        if InNum is not None:
            Dict["InNum"] = InNum
        if OutNum is not None:
            Dict["OutNum"] = OutNum
        if Stride is not None:
            Dict["Stride"] = Stride
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
                assert isinstance(Value, int)
            elif Key in ["Padding"]:
                Param.Padding.Value = Value
                # assert isinstance(Value, int) or Value in ["KeepFeatureMapHeightWidth"]
            elif Key in ["Stride"]:
                Param.Stride = Value
                assert isinstance(Value, int)
            else:
                _Dict[Key] = Value
        super().SetParam(**_Dict)
        return self
    def Receive(self, In):
        Out = F.conv2d(
            input=In, weight=self.Kernel, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Out)
        return Output
    def SetKernel(self, Kernel):
        Param = self.Param
        self.SetTrainParam(
            "Kernel", "Kernel.Value", Kernel
        ) # (OutNum, InNum // GroupNum, KernelWidth, KernelHeight)
        self.Weight = self.Kernel
        Param.In.Num = Param.Kernel.In.Num * Param.Group.Num
        Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
        Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
        Param.Kernel.In.Num = Kernel.shape[1]
        Param.Kernel.Height = Kernel.shape[2] # Height at dim 2
        Param.Kernel.Width = Kernel.shape[3] # Width at dim 3
        return self
    def SetBias(self, Bias, Trainable=True):
        Param = self.Param
        if isinstance(Bias, str):
            if Bias in ["zeros"]:
                Bias = np.zeros((Param.Out.Num))
            else:
                raise Exception()
        
        if Trainable:
            self.SetTrainParam("Bias", "Bias.Value", Bias)
            Param.Bias.Trainable = True
        else:
            self.SetTensor("Bias", "Bias.Value", Bias)
        return self
    def SetNonLinear(self):
        Param = self.Param

        return self
    def Build(self, IsSuper=False, IsRoot=True):
        if not IsSuper:
            Param = self.Param
            if self.IsInit():
                Param.setdefault("Stride", 1)
                Param.Padding.setdefault("Value", "KeepFeatureMapHeightWidth")

                Param.setdefault("Dilation", 1)
                Param.Group.setdefault("Num", 1)
                
                if Param.Kernel.hasattr("Size"):
                    if not Param.Kernel.hasattr("Height"):
                        Param.Kernel.Height = Param.Kernel.Size
                    if not Param.Kernel.hasattr("Width"):
                        Param.Kernel.Width = Param.Kernel.Size
                else:
                    assert Param.Kernel.hasattr("Height")
                    assert Param.Kernel.hasattr("Width")
                    if Param.Kernel.Height == Param.Kernel.Width:
                        Param.Kernel.Size = Param.Kernel.Height

                # set kernel / weight
                # torch group convolution kernel: 
                #   (OutNum, InNum // GroupNum, KernelHeight, KernelWidth)
                #   In dimension during computation will be divided into groups.
                assert Param.In.Num % Param.Group.Num == 0
                assert Param.Out.Num % Param.Group.Num == 0
                Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
                Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
                
                if not Param.Kernel.hasattr("Data"):
                    # [OutNum // GroupNum, InNum, KernelHeight, KernelWidth]
                    Param = self.Param
                    self.SetTrainParam(
                        Name="Kernel", Path="Kernel.Data",
                        Data=DLUtils.DefaultConv2DKernel(
                            (
                                Param.In.Num, Param.Out.Num, 
                                Param.Kernel.Height, Param.Kernel.Width
                            ),
                            GroupNum=Param.Group.Num,
                            # NonLinear="ReLU"
                            # default torch weight initialization does not consider NonLinear.
                        ) # (InNum, OutNum // GroupNum, KernelHeight, KernelWidth)
                    )

                # bias setting
                Param.Bias.setdefault("Enable", True)
                if Param.Bias.Enable:
                    Param.Bias.setdefault("Trainable", True)
                    if not Param.Bias.hasattr("Data"):
                        self.SetTensor(
                            Name="Bias", Path="Bias.Data",
                            Data=DLUtils.DefaultConv2DBias(
                                (
                                    Param.In.Num, Param.Out.Num, 
                                    Param.Kernel.Height, Param.Kernel.Width
                                ),
                                GroupNum=Param.Group.Num
                            )
                        )
                        if Param.Bias.Trainable:
                            self.SetTrainable("Bias")
                else:
                    Param.Bias.Data = 0.0                

                # set nonlinear setting
                Param.NonLinear.setdefault("Enable", True)
                if Param.NonLinear.Enable:
                    Param.NonLinear.setdefault("Type", "ReLU")
                    NonLinearModule = DLUtils.network.NonLinearTransform(Param.NonLinear.Type)
                    self.AddSubModule("NonLinear", NonLinearModule)

            # nonlinear setting
            if Param.NonLinear.Enable:
                assert self.HasSubModule("NonLinear")
            else:
                self.NonLinear = lambda x:x
            
            # padding setting
            Padding = Param.Padding.Value
            if isinstance(Padding, int):
                self.Padding = Padding
            elif isinstance(Padding, str):
                if Padding in ["KeepFeatureMapHeightWidth"]:
                    self.Padding = "same"
                else:
                    raise Exception()
            elif isinstance(Padding, list) or isinstance(Padding, tuple):
                self.Padding = tuple(Padding)
            else:
                raise Exception()
            self.Dilation = Param.Dilation # default 1
            self.GroupNum = Param.Group.Num
            self.Stride = Param.Stride
            self.Bias = Param.Bias.Data

        return super().Init(IsSuper=True, IsRoot=IsRoot)

class UpConv2D(Conv2D):
    """
    process of 2D upconv / 2D transposed convolution.
        stride / expand input image.
            intervel between neighboring input image pixel across X/width  direction would b strideX - 1.
            intervel between neighboring input image piexl across Y/height direction would b strideY - 1.
            same stride in done for all feature channels.
            interval positions created by expansion can be seen as new elements with value 0.
        
    """
    def __init__(self, InNum=None, OutNum=None, Stride=None, **Dict):
        super().__init__(InNum=InNum, OutNum=OutNum, Stride=Stride, **Dict)
    def Receive(self, In):
        """
        In: (BatchNum, ChannelNum, Height, Width)
        """
        Output = F.conv_transpose2d(
            input=In, weight=self.Kernel, bias=self.Bias,
            stride=self.Stride, padding=self.Padding,
            dilation=self.Dilation, groups=self.GroupNum,
            output_padding=self.OutputPadding
        )
        Output = self.NonLinear(Output)
        return Output
    # def SetKernel(self, Kernel):
    #     Param = self.Param
    #     Param.In.Num = Kernel.shape[0]
    #     Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
    #     Param.Kernel.Out.Num = Kernel.shape[1]
    #     Param.Out.Num = Param.Kernel.Out.Num * Param.Group.Num
    #     Param.Kernel.Height = Kernel.shape[2] # Height at dim 2
    #     Param.Kernel.Width = Kernel.shape[3] # Width at dim 3
    def Build(self, IsSuper=False, IsRoot=True):
        if not IsSuper:
            Param = self.Param
            if self.IsInit():
                Param.setdefault("Stride", 1)
                Param.Padding.setdefault("Value", 1)
        
                Param.setdefault("Dilation", 1)
                Param.Group.setdefault("Num", 1)
                
                if Param.Kernel.hasattr("Size"):
                    if not Param.Kernel.hasattr("Height"):
                        Param.Kernel.Height = Param.Kernel.Size
                    if not Param.Kernel.hasattr("Width"):
                        Param.Kernel.Width = Param.Kernel.Size
                else:
                    assert Param.Kernel.hasattr("Height")
                    assert Param.Kernel.hasattr("Width")
                    if Param.Kernel.Height == Param.Kernel.Width:
                        Param.Kernel.Size = Param.Kernel.Height

                # torch group convolution kernel: 
                #   (OutNum, InNum // GroupNum, KernelHeight, KernelWidth)
                #   In dimension during computation will be divided into groups.
                assert Param.In.Num % Param.Group.Num == 0
                assert Param.Out.Num % Param.Group.Num == 0
                Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
                Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
                Param.NonLinear.setdefault("Enable", False)

                if not Param.Kernel.hasattr("Data"):
                    # (OutNum // GroupNum, InNum, KernelHeight, KernelWidth)
                    self.SetTrainParam(
                        Name="Kernel", Path="Kernel.Data",
                        Data=DLUtils.DefaultUpConv2DKernel(
                            (
                                Param.In.Num, Param.Out.Num, 
                                Param.Kernel.Height, Param.Kernel.Width
                            ),
                            GroupNum=Param.Group.Num,
                            # NonLinear="ReLU"
                            # default torch weight initialization does not consider NonLinear.
                        ) # (InNum, OutNum // GroupNum, KernelWidth, KernelHeight)
                    )

                # bias setting
                Param.Bias.setdefault("Enable", True)
                if Param.Bias.Enable:
                    Param.Bias.setdefault("Trainable", True)
                    if not Param.Bias.hasattr("Value"):
                        Bias = DLUtils.DefaultUpConv2DBias(
                            (
                                Param.In.Num, Param.Out.Num, 
                                Param.Kernel.Height, Param.Kernel.Width
                            ),
                            GroupNum=Param.Group.Num,
                        )
                        if Param.Bias.Trainable:
                            self.SetTrainParam("Bias", "Bias.Value", Bias)
                        else:
                            self.SetUnTrainableParam("Bias", "Bias.Value", Bias)
                        # Param.Tensor.add("Bias")

                # set nonlinear setting
                self.SetNonLinear()

            if not Param.Bias.Enable:
                self.Bias = 0.0

            self.Dilation = Param.Dilation
            self.GroupNum = Param.Group.Num
            self.Stride = Param.Stride

            # padding value setting
            """
            pytorch implementation
                padding_actual = dilation * (kernel_size - 1) - padding_input_value
                therefore
                padding_input_value = dilation * (kernel_size - 1) - padding_actual
            """
            assert isinstance(Param.Padding.Value, int)
            self.Padding = self.Dilation * (Param.Kernel.Size - 1) - Param.Padding.Value
            if Param.Padding.hasattr("Additional"):
                self.OutputPadding = Param.Padding.Additional
            else:
                self.OutputPadding = 0

            if Param.NonLinear.Enable:
                assert Param.SubModules.hasattr("NonLinear")
            else:
                self.NonLinear = lambda x:x
        return super().Init(IsSuper=True, IsRoot=IsRoot)

