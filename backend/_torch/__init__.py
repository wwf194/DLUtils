import DLUtils
try:
    import torch
except Exception:
    pass
else:
    def TorchTensorElementNum(Tensor:torch.Tensor):
        return Tensor.numel()
    def TorchTensorSize(Tensor:torch.Tensor):
        if Tensor.data.is_floating_point():
            return Tensor.numel() * torch.finfo(Tensor.data.dtype).bits
        else:
            return Tensor.numel() * torch.iinfo(Tensor.data.dtype).bits
    def GPUNum():
        return torch.cuda.device_count()
    def SampleFrom01NormalDistributionTorch(Shape):
        return torch.randn(Shape) # sampple from N(0, 1)
    def SetSeedForTorch(Seed: int):
        torch.manual_seed(Seed)

try:
    from .format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
    from .module import TorchModule, TorchModuleWrapper
except Exception:
    pass

def GetTensorByteNum(Tensor): # Byte
    return Tensor.nelement() * Tensor.element_size()

def GetTensorElementNum(Tensor): # Byte
    return Tensor.nelement()


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

def TensorWithSameShapeAndValue1(Tensor):
    # assert isinstance(Tensor, torch.Tensor)
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
    Weight = Weight.transpose(1, 0)
    return DLUtils.ToTorchTensor(Weight), DLUtils.ToTorchTensor(Bias)




def TorchTrainParamStat(tensor, verbose=False, ReturnType="PyObj"):
    statistics = {
        "Min": torch.min(tensor).item(),
        "Max": torch.max(tensor).item(),
        "Mean": torch.mean(tensor).item(),
        "Std": torch.std(tensor).item(),
        "Var": torch.var(tensor).item()
    }
    if ReturnType in ["Dict"]:
        return statistics
    elif ReturnType in ["PyObj"]:
        return DLUtils.PyObj(statistics)
    else:
        raise Exception()

try:
    from .module import TorchModelWithAdditionalParam2File, File2TorchModelWithAdditionalParam
    from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
except Exception:
    pass
try:
    from .format import NullParameter, ToTorchTensor
except Exception:
    pass