try:
    import torch
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
    from .module import TorchModule, TorchModuleWrapper
except Exception:
    pass

import DLUtils
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

def TensorWithSameShapeAndValue1(Tensor: torch.Tensor):
    return torch.ones_like(Tensor)

def AddDimension(Tensor, DimIndex: int):
    # Tensor[None, :, :] # another way.
    TensorNew = torch.unsqueeze(Tensor, DimIndex)
    return TensorNew
GetModelTrainParam = GetTorchModelTrainParam


def TorchLinearInitWeightBias(InNum, OutNum):
    module = torch.nn.Linear(in_features=InNum, out_features=OutNum, bias=True)
    Dict = dict(module.named_parameters())
    Weight = Dict["weight"] # (OutNum, InNum)
    Bias = Dict["bias"] # (OutNum, InNum)
    Weight = DLUtils.ToNpArray(Weight)
    Bias = DLUtils.ToNpArray(Bias)
    return DLUtils.ToTorchTensor(Weight), DLUtils.ToTorchTensor(Bias)

try:
    from .module import TorchModelWithAdditionalParam2File, File2TorchModelWithAdditionalParam
    from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
except Exception:
    pass
from .format import NullParameter, ToTorchTensor