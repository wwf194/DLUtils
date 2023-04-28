
import DLUtils
# from einops import rearrange, reduce, repeat
import torch
class Image2PatchList(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("Width", "PatchWidth"): "Patch.Width",
        ("Height", "PatchHeight"): "Patch.Height",
        ("ImageWidth"): "Image.Width",
        ("ImageHeight"): "Image.Height",
        ("PatchNumX", "PatchXNum"): "Patch.NumX",
        ("PatchNumY", "PatchYNum"): "Patch.NumY"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def Receive(self, X:torch.Tensor):
        # X: (BatchSize, ChannelNum, Height, Width)
        # the following dimension operation could also be done using einops
        BatchSize = X.size(0)
        ChannelNum = X.size(1)
        X = X.view(BatchSize, ChannelNum, self.PatchNumY, self.PatchHeight, self.PatchNumX, self.PatchWidth)
            # X: (BatchSize, ChannelNum, PatchYNum, PatchHeight, PatchNumX, PatchWidth)
        X = X.permute(0, 2, 4, 3, 5, 1)
            # X: (BatchSize, PatchYNum, PatchNumX, PatchHeight, PatchWidth, ChannelNum)
        X = X.reshape(BatchSize, self.PatchNumY * self.PatchNumX, self.PatchHeight * self.PatchWidth * ChannelNum)
        # Rearrange('BatchSize ChannelNum (Height PatchHeight) (Width PatchWidth) -> BatchSize (Height Width) \
        # (PatchNumY PatchNumX ChannelNum)', PatchHeight=PatchHeight, p2=PatchWidth),
        return X
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Patch.hasattr("NumX")
        assert Param.Patch.hasattr("NumY")
        
        self.PatchNumX = Param.Patch.NumX
        self.PatchNumY = Param.Patch.NumY
        if Param.hasattr("Image.Width") and Param.hasattr("Image.Height"):
            self.PatchWidth = Param.Image.Width // Param.Patch.NumX
            self.PatchHeight = Param.Image.Height // Param.Patch.NumY
        else:
            raise Exception() # to be implemented

        return super().Init(IsSuper=True, IsRoot=IsRoot)


import torchvision
class CenterCrop(DLUtils.module.AbstractNetwork):
    def __init__(self, Height=None, Width=None, **Dict):
        if Height is not None:
            Dict["Height"] = Height
            if Width is not None:
                Dict["Width"] = Width
            else:
                Dict["Width"] = Height
        super().__init__(**Dict)
        
        a = 1
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