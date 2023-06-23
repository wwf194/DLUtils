import torch
try:
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
except Exception:
    pass
# from .module import TorchModule, TorchModuleWrapper

def GetTensorByteNum(Tensor): # Byte
    return Tensor.nelement() * Tensor.element_size()

def GetTensorElementNum(Tensor): # Byte
    return Tensor.nelement()

def SampleFrom01NormalDistributionTorch(Shape):
    return torch.randn(Shape)

import DLUtils
def TorchModel2File(Model, FilePath):
    DLUtils.file.EnsureFileDir(FilePath)
    torch.save(Model.state_dict(), FilePath)


