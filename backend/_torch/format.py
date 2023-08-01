import numpy as np
import torch
import torch.nn as nn
import DLUtils


def NullParameter(Shape=(1)):
    return nn.Parameter(torch.empty(Shape))
def ToTorchTensor(Data, Device=None):
    if isinstance(Data, np.ndarray):
        _Data = NpArray2Tensor(Data)
    elif isinstance(Data, list):
        _Data = NpArray2Tensor(DLUtils.List2NpArray(Data))
    elif isinstance(Data, torch.Tensor):
        _Data = Data
    else:
        raise Exception(type(Data))
    if Device is not None:
        _Data = _Data.to(Device)
    return _Data

def ToTorchTensorOrNum(data):
    if isinstance(data, float):
        return data
    elif isinstance(data, int):
        return data
    else:
        return ToTorchTensor(data)

def ToGivenDataTypeTorch(Data, DataType=torch.float32):
    if Data.dtype == DataType:
        return Data
    else:
        return Data.to(DataType)
Tensor2GivenDataType = ToGivenDataTypeTorch

def NpArray2TorchTensor(Data, Location="cpu", DataType=torch.float32, RequiresGrad=False):
    Data = torch.from_numpy(Data)
    Data = Tensor2GivenDataType(Data, DataType)
    Data = Data.to(Location)
    Data.requires_grad = RequiresGrad
    return Data

NpArray2Tensor = NpArray2TorchTensor