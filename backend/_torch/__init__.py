import torch
try:
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
except Exception:
    pass
from .module import TorchModule, TorchModuleWrapper

def GetTensorByteNum(Tensor): # Byte
    return Tensor.nelement() * Tensor.element_size()

def GetTensorElementNum(Tensor): # Byte
    return Tensor.nelement()

def SampleFrom01NormalDistributionTorch(Shape):
    return torch.randn(Shape)

def ListTorchModelTrainParam(model):
    Dict = dict(model.named_parameters())
    TrainParamList = []
    for name, value in Dict.items():
        TrainParamList.append(name)
    return TrainParamList

def GetTorchModelTrainParam(model):
    Dict = dict(model.named_parameters())
    return Dict
GetModelTrainParam = GetTorchModelTrainParam

def GetTorchModelTensor(model):
    # include parameter and buffer
    Dict = dict(model.named_parameters())
    Dict.update(
        dict(model.named_buffers())
    )
    return Dict
GetModelTrainParam = GetTorchModelTrainParam


from .module import TorchModelWithAdditionalParam2File, File2TorchModelWithAdditionalParam
from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
from .format import NullParameter, ToTorchTensor