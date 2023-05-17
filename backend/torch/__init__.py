from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
# from .module import TorchModule, TorchModuleWrapper

def GetTensorByteNum(Tensor): # Byte
    return Tensor.nelement() * Tensor.element_size()

def GetTensorElementNum(Tensor): # Byte
    return Tensor.nelement()