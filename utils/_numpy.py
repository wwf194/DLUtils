# from __future__ import annotations
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = DLUtils.GetLazyNumpy()
    pd = DLUtils.LazyImport("pandas")

def NpArray2D2Str(Data, ColName=None, RowName=None, **Dict):
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

def NpArrayToTextFile(Data, SavePath, **Dict):
    DimNum = len(Data.shape)    
    if DimNum == 2:
        NpArray2DToTextFile(Data, SavePath=SavePath, **Dict)
    else:
        raise Exception()

def SetSeedForNumpy(Seed: int):
    np.random.seed(Seed)
    return

def FlattenNpArray(Data):
    return Data.flatten()

def EnsureFlatNp(Data):
    return Data.flatten()
EnsureFlat = EnsureFlatNp

def NpArray2List(Data):
    # Data: np.ndarray
    return Data.tolist()