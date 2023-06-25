<<<<<<< HEAD
import torch
try:
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
except Exception:
    pass
from .module import TorchModule, TorchModuleWrapper
=======

try:
    import torch
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
    from .module import TorchModule, TorchModuleWrapper
except Exception:
    pass
>>>>>>> 426047aa2b8d15bb4de6474c91a842bf2b77945b

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
<<<<<<< HEAD
from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
=======

try:
    from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
except Exception:
    pass
>>>>>>> 426047aa2b8d15bb4de6474c91a842bf2b77945b
from .format import NullParameter, ToTorchTensor