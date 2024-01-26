from __future__ import annotations
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")
    
def NullParameter(Shape=(1)):
    return nn.Parameter(torch.empty(Shape))
def ToTorchTensor(Data, Device=None):
    if isinstance(Data, np.ndarray):
        _Data = NpArray2Tensor(Data)
    elif isinstance(Data, list):
        _Data = NpArray2Tensor(DLUtils.ListToNpArray(Data))
    elif isinstance(Data, torch.Tensor):
        _Data = Data
    elif isinstance(Data, float):
        raise NotImplementedError()
    else:
        raise Exception(type(Data))
    if Device is not None:
        _Data = _Data.to(Device)
    return _Data

def ToTrainableTorchTensor(Data):
    if isinstance(Data, np.ndarray):
        return NpArray2Tensor(Data, RequiresGrad=True)
    elif isinstance(Data, list):
        return NpArray2Tensor(DLUtils.ListToNpArray(Data), RequiresGrad=True)
    elif isinstance(Data, torch.Tensor):
        Data.requires_grad = True
        return Data
    else:
        raise Exception(type(Data))

def ToTorchTensorOrNum(Data):
    if isinstance(Data, float):
        return Data
    elif isinstance(Data, int):
        return Data
    else:
        return ToTorchTensor(Data)

def ToGivenDataTypeTorch(Data, DataType=None):
    if DataType is None:
        DataType = torch.float32
    if Data.dtype == DataType:
        return Data
    else:
        return Data.to(DataType)
Tensor2GivenDataType = ToGivenDataTypeTorch

def NpArrayToTorchTensor(Data, Location="cpu", DataType=None, RequiresGrad=False):
    if DataType is None:
        DataType = torch.float32
    Data = torch.from_numpy(Data)
    Data = Tensor2GivenDataType(Data, DataType)
    Data = Data.to(Location)
    Data.requires_grad = RequiresGrad
    return Data
NpArray2Tensor = NpArrayToTensor = NpArray2TorchTensor = NpArrayToTorchTensor

def TorchTensorToNpArray(data):
    data = data.detach().cpu().numpy()
    return data # data.grad will be lost.
Tensor2NpArray = TensorToNpArray = TorchTensor2NpArray = TorchTensorToNpArray

def Tensor2Str(data):
    return DLUtils.utils.NpArray2Str(Tensor2NpArray(data))

def Tensor2File(data, SavePath):
    DLUtils.EnsureFileDir(SavePath)
    np.savetxt(SavePath, DLUtils.Tensor2NpArray(data))

def Tensor2NumpyOrFloat(data):
    try:
        _data = data.item()
        return _data
    except Exception:
        pass
    data = data.detach().cpu().numpy()
    return data