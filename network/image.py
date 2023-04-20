
import DLUtils
# from einops import rearrange, reduce, repeat
import torch
class Image2PatchList(DLUtils.module.AbstractNetwork):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("Width", "PatchWidth"): "Patch.Width",
        ("Height", "PatchHeight"): "Patch.Height",
        ("ImageWidth"): "Image.Width",
        ("ImageHeight"): "Image.Height",
        ("PatchNumX", "PatchXNum"): "Patch.XNum",
        ("PatchNumY", "PatchYNum"): "Patch.YNum"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def Receive(self, X:torch.Tensor):
        # X: (BatchSize, ChannelNum, Height, Width)
        # the following dimension operation could also be done using einops
        BatchSize = X.size(0)
        ChannelNum = X.size(1)
        X = X.view(BatchSize, ChannelNum, self.PatchYNum, self.PatchHeight, self.PatchXNum, self.PatchWidth)
            # X: (BatchSize, ChannelNum, PatchYNum, PatchHeight, PatchXNum, PatchWidth)
        X = X.permute(0, 2, 4, 3, 5, 1)
            # X: (BatchSize, PatchYNum, PatchXNum, PatchHeight, PatchWidth, ChannelNum)
        X = X.reshape(BatchSize, self.PatchYNum * self.PatchXNum, self.PatchHeight * self.PatchWidth, ChannelNum)
        return X
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Patch.hasattr("XNum")
        assert Param.Patch.hasattr("YNum")
        
        self.PatchXNum = Param.Patch.XNum
        self.PatchYNum = Param.Patch.YNum
        if Param.hasattr("Image.Width") and Param.hasattr("Image.Height"):
            self.PatchWidth = Param.Image.Width // Param.Patch.XNum
            self.PatchHeight = Param.Image.Height // Param.Patch.YNum
        else:
            raise Exception() # to be implemented

        return super().Init(IsSuper=True, IsRoot=IsRoot)


import torchvision
class CenterCrop(DLUtils.module.AbstractNetwork):
    def __init__(self, Width=None, Height=None, **Dict):
        if Width is not None:
            Dict["Width"] = Width
            if Height is not None:
                Dict["Height"] = Height
            else:
                Dict["Height"] = Width
        super().__init__(**Dict)
    def Receive(self, X):
        # X: (..., Height, Width)
        # image smaller than (Param.Height, Param.Width) will be patched with zero.
        return self.module(X)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("Height")
        assert Param.hasattr("Width")
        assert isinstance(Param.Height, int)
        assert isinstance(Param.Width, int)
        self.module = torchvision.transforms.CenterCrop((Param.Height, Param.Width))
        return super().Init(IsSuper=True, IsRoot=IsRoot)