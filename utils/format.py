
import DLUtils
import numpy as np
import re
try:
    import pandas as pd
except Exception:
    pass
B = 1
KB = 1024
MB = 1048576
GB = 1073741824
TB = 1099511627776

def ToByteNum(SizeStr):
    if isinstance(SizeStr, int):
        return SizeStr
    assert isinstance(SizeStr, str)
    Pattern = r"(.*)([Byte|B|k|K|kb|kB|KB|MB|MiB|mb|m|g|gb|GB|])"
    Result = re.match(Pattern, SizeStr)
    if Result is None:
        raise Exception()
    NumStr = Result.group(1)
    UnitStr = Result.group(2)
    Num = int(NumStr)
    if UnitStr in ["B", "b", "byte", "Byte"]:
        Unit = B
    elif UnitStr in ["K", "k", "B", "kb", "KB"]:
        Unit = KB
    elif Unit in ["m", "B", "MB", "MiB"]:
        Unit = MB
    elif Unit in ["g", "G", "GB", "Gygabyte"]:
        Unit = GB
    else:
        raise Exception()
    return Num * Unit

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
    return pd.DataFrame(Data).to_string()
def NpArray2D2TextFile(Data, ColName=None, RowName=None, WriteStat=True, SavePath=None):
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

def NpArray2TextFile(Data, SavePath, **Dict):
    DimNum = len(Data.shape)    
    if DimNum == 2:
        NpArray2D2TextFile(Data, SavePath=SavePath, **Dict)
    else:
        raise Exception()

def Int201String(Int, _0bPrefix=False, LeadingZero=False, Length=None):
    Pattern = []
    if _0bPrefix:
        Pattern.append("#")
    if LeadingZero:
        Pattern.append("0")
    if Length is not None:
        Pattern.append(str(Length))
    Pattern = "".join(Pattern)
    if Length is None:
        return format(Int, Pattern)
    
    # return "{0:b}".format(Int)
IntTo01String = Int201String

def String012Int(String, MostSignificantBit="Right", BigEndian=False):
    Int = 0
    DLUtils.RemovePrefix(String, "0b", MustMatch=False)
    if MostSignificantBit in ["big", "Big"] or BigEndian:
        Base = 1
        for Index in len(String):
            if String[Index] == "0":
                pass
            elif String["Index"] == "1":
                Int += Base
            else:
                raise Exception()
            Base *= 2
    else:
        raise NotImplementedError()

String01ToInt = String012Int
def Curve2TextFile(Dict, SavePath):
    Index = 0
    ColName = []
    ValueList = []
    for Key, Value in Dict.items():
        ValueList.append(Value)
        ColName.append(Key)    
    Data = np.array(ValueList).transpose(1, 0)
    NpArray2D2TextFile(Data, ColName, SavePath=SavePath)
