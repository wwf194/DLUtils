from __future__ import annotations
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    F = DLUtils.LazyImport("torch.nn.functional")
if TYPE_CHECKING:
    from .train import DataLoaderForEpochBatchTrain, DataFetcherForEpochBatchTrain

def TorchTensorElementNum(Tensor: torch.Tensor):
    # assert isinstance(Tensor, torch.Tensor)
    return Tensor.numel()

def TorchTensorSize(Tensor:torch.Tensor):
    # assert isinstance(Tensor, torch.Tensor)
    if Tensor.data.is_floating_point():
        return Tensor.numel() * torch.finfo(Tensor.data.dtype).bits
    else:
        return Tensor.numel() * torch.iinfo(Tensor.data.dtype).bits

def GPUNumTorch():
    return torch.cuda.device_count()
GetGPUNum = GPUNum = GPUNumTorch

def SampleFrom01NormalDistributionTorch(Shape):
    return torch.randn(Shape) # sampple from N(0, 1)

def SetSeedForTorch(Seed: int):
    torch.manual_seed(Seed)

def ReportGPUUseageOfCurrentProcess():
    """
        report total / reserved / allocated / free(unallocated inside reservide) memory of the process of each GPU.
        note that 
    """
    GPUList = [torch.cuda.device(GPUIndex) for GPUIndex in range(torch.cuda.device_count())]
    for GPUIndex in range(torch.cuda.device_count()):
        MemoryTotal = torch.cuda.get_device_properties(GPUIndex).total_memory
        MemoryReserved = torch.cuda.memory_reserved(GPUIndex)
        MemoryAllocated = torch.cuda.memory_allocated(GPUIndex)
        MemoryUnallocated = MemoryAllocated - MemoryReserved
        # note that other process could also have reserved some memory on same GPU.
        # _MemoryFree = MemoryTotal - MemoryReserved
        # print("GPU %d. Total: %010d. Reserved: %010d. Allocated: %010d Free: %010d"%(
        print("GPU %d. Total: %s. Reserved: %s. Allocated: %s Free: %s"%(
            GPUIndex, 
            DLUtils.Size2Str(MemoryTotal), 
            DLUtils.Size2Str(MemoryReserved), 
            DLUtils.Size2Str(MemoryAllocated), 
            DLUtils.Size2Str(MemoryUnallocated)
        ))

def GetBatchNum(TorchDataLoader):
    return len(TorchDataLoader)

def ToOneHot(Data, ClassNum):
    # assert len(Data.shape) == 1
    return F.one_hot(Data.long(), num_classes=ClassNum)

from .format import (
    ToTorchTensor, ToTorchTensorOrNum,
    NpArrayToTorchTensor, NpArray2TorchTensor,
    NpArray2Tensor, NpArrayToTensor,
    TorchTensorToNpArray, TorchTensor2NpArray,
    TensorToNpArray, Tensor2NpArray
)

def SampleFrom01NormalDistributionTorch(Shape):
    return torch.randn(Shape) # sampple from N(0, 1)

try:
    from .module import (
        TorchModule,
        TorchModuleWrapper
    )
except Exception:
    pass

def __getattr__(Name):
    if Name in ["TorchModule"]:
        from ._module import TorchModule as _TorchModule
        global TorchModule
        TorchModule = _TorchModule
        return TorchModule
    elif Name in ["TorchModuleWrapper"]:
        from ._module import TorchModuleWrapper as _TorchModuleWrapper
        global TorchModuleWrapper
        TorchModuleWrapper = _TorchModuleWrapper
        return TorchModuleWrapper
    elif Name in ["TorchModelWithAdditionalParamToFile"]:
        from ._module import TorchModelWithAdditionalParamToFile as _TorchModelWithAdditionalParamToFile
        global TorchModelWithAdditionalParamToFile
        TorchModelWithAdditionalParamToFile = _TorchModelWithAdditionalParamToFile
        return TorchModelWithAdditionalParamToFile
    if Name in ["DataFetcherForEpochBatchTrain"]:
        from .train import DataFetcherForEpochBatchTrain as _DataFetcherForEpochBatchTrain
        global DataFetcherForEpochBatchTrain
        DataFetcherForEpochBatchTrain = _DataFetcherForEpochBatchTrain
        return DataFetcherForEpochBatchTrain
    elif Name in ["DataLoaderForEpochBatchTrain"]:
        from .train import DataLoaderForEpochBatchTrain as _DataLoaderForEpochBatchTrain
        global DataLoaderForEpochBatchTrain
        DataLoaderForEpochBatchTrain = _DataLoaderForEpochBatchTrain
        return DataLoaderForEpochBatchTrain
    else:
        raise Exception(Name)

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

def IsTraining(module: torch.nn.Module):
    return module.training
IsTrain = IsTraining

def IsEval(module: torch.nn.Module):
    return not module.training

try:
    from .module import (
        TorchModelWithAdditionalParamToFile,
        File2TorchModelWithAdditionalParam
    )
    from .module import TorchModel2File, File2TorchModel, TorchModelWrapper
except Exception:
    pass
try:
    from .format import NullParameter, ToTorchTensor
except Exception:
    pass