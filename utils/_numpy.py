# from __future__ import annotations
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = DLUtils.GetLazyNumpy()
    pd = DLUtils.LazyImport("pandas")

def NpArray2DToStr(Data, ColName=None, RowName=None, **Dict):
    # assert IsPandasImported
    assert len(Data.shape) == 2
    DataDict= {}
    if ColName is None:
        ColName = ["Col %d"%ColIndex for ColIndex in range(Data.shape[1])]
    for ColIndex, Name in enumerate(ColName):
        DataDict[Name] = Data[:, ColIndex]
    if RowName is not None:
        # to be implemented
        pass
    return pd.DataFrame(Data).to_string()
NpArray2D2Str = NpArray2DToStr

def ToNpArray(Data, DataType=None):
    if DataType is None:
        DataType = np.float32
    if isinstance(Data, np.ndarray):
        return Data
    elif isinstance(Data, list) or isinstance(Data, tuple):
        return np.array(Data, dtype=DataType)
    elif isinstance(Data, float):
        return np.asarray([Data],dtype=DataType)
    elif IsTorchImported and isinstance(Data, torch.Tensor):
        return DLUtils.torch.TensorToNpArray(Data)
    else:
        raise Exception(type(Data))

def ListToNpArray(Data, DataType=None):
    if DataType is not None:
        return np.asarray(Data, dtype=DataType)
    else:
        return np.asarray(Data)

def ListToNpArrayFloat32(List):
    return np.asarray(List, dtype=np.float32)

def NpArrayToList(Data):
    return Data.tolist()
NpArray2List = NpArrayToList

def ToMean0Std1(Data):
    return (Data - Data.mean()) / Data.std()

def ToNpArrayOrNum(Data, DataType=None):
    if DataType is None:
        DataType = np.float32
    if isinstance(Data, float):
        return Data
    if isinstance(Data, int):
        return Data
    Data = ToNpArray(Data)
    if Data.size == 0: # empty array
        return None
    elif Data.size == 1: # single element array
        return Data.reshape(1)[0]
    else:
        return Data

def ToNpArrayIfIsTorchTensor(Data):
    if isinstance(Data, torch.Tensor):
        return DLUtils.ToNpArray(Data), False
    else:
        return Data, True

def NpArray2DToTextFile(Data, ColName=None, RowName=None, WriteStat=True, SavePath=None):
    assert SavePath is not None
    StrList = []
    StrShape = "Shape: %s\n"%(str(Data.shape))
    StrList.append(StrShape)
    if len(Data.shape) == 1:
        Data = Data[:, np.newaxis]
    Str = NpArray2D2Str(Data, ColName=ColName, RowName=RowName, SavePath=SavePath)
    StrList.append(Str)
    if WriteStat:
        StrStat = DLUtils.math.NpArrayStatisticsStr(Data, verbose=False)
        StrList.append(StrStat)
    DLUtils.Str2File(
        "".join(StrList), 
        FilePath=SavePath
    )
NpArray2D2TextFile = NpArray2DToTextFile

def NpArrayToTextFile(Data, SavePath, **Dict):
    DimNum = len(Data.shape)    
    if DimNum == 2:
        NpArray2DToTextFile(Data, SavePath=SavePath, **Dict)
    else:
        raise Exception()
NpArray2TextFile = NpArrayToTextFile

def SetSeedForNumpy(Seed: int):
    np.random.seed(Seed)
    return

def FlattenNpArray(Data):
    return Data.flatten()

def EnsureFlatNp(Data):
    return Data.flatten()
EnsureFlat = EnsureFlatNp

def NpArrayToList(Data):
    # Data: np.ndarray
    return Data.tolist()
NpArray2List = NpArrayToList

def SoftMaxNp(List):
    x = np.asarray(List, dtype=np.float32)
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
