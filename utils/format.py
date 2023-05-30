import pandas as pd
import torch
import DLUtils
import numpy as np

KB = 1024
MB = 1048576
GB = 1073741824
TB = 1099511627776

def ByteNum2Str(ByteNum):
    if ByteNum < KB:
        Str = "%d B"%ByteNum
    elif ByteNum < MB:
        Str = "%.3f KB"%(1.0 * ByteNum / KB)
    elif ByteNum < GB:
        Str = "%.3f MB"%(1.0 * ByteNum / MB)
    elif ByteNum < TB:
        Str = "%.3f GB"%(1.0 * ByteNum / GB)
    else:
        Str = "%.3f TB"%(1.0 * ByteNum / TB)
    return Str

def Num2Str1024(Num):
    if Num < KB:
        Str = "%d"%Num
    elif Num < MB:
        Str = "%.3f K"%(1.0 * Num / KB)
    elif Num < GB:
        Str = "%.3f M"%(1.0 * Num / MB)
    elif Num < TB:
        Str = "%.3f G"%(1.0 * Num / GB)
    else:
        Str = "%.3f T"%(1.0 * Num / TB)
    return Str
Num2Str = Num2Str1024

def NpArray2Str(Data, **Dict):
    DimNum = len(Data.shape)    
    if DimNum == 2:
        DataStr = NpArray2D2Str(Data, **Dict)
        Name = Dict.setdefault("Name", "NpArray")
        Shape = str(Data.shape)
        Info = "{0}. Shape: {1}".format(Name, list(Shape))
        Dim = "Dim 0 / Dim 1"
        return "\n".join([Name, Shape, Dim, DataStr])
    else:
        raise Exception()

def NpArray2D2Str(Data, ColName=None, RowName=None, **Dict):
    assert len(Data.shape) == 2
    DataDict= {}
    if ColName is None:
        ColName = ["Col %d"%ColIndex for ColIndex in range(Data.shape[1])]
    for ColIndex, Name in enumerate(ColName):
        DataDict[Name] = Data[:, ColIndex]
    if RowName is not None:
        # to be implemented
        pass
    
    StrList = ["Dim 0 / Dim 1\n"]
    StrList.append(pd.DataFrame(Data).to_string())
    return "".join(StrList)
def NpArray2D2TextFile(Data, SavePath=None, ColName=None, RowName=None):
    Str = NpArray2D2Str(Data, ColName=ColName, RowName=RowName, SavePath=SavePath)
    DLUtils.Str2File(Str, SavePath)

def NpArray2TextFile(Data, SavePath, **Dict):
    DimNum = len(Data.shape)    
    if DimNum == 2:
        NpArray2D2TextFile(Data, SavePath=SavePath, **Dict)
    else:
        raise Exception()