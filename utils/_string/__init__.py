import re
import string
import warnings
import DLUtils
import sys
from io import StringIO, BytesIO, TextIOWrapper

# String Related Functions
def LStrip(Str, Prefix):
    if Str[0:len(Prefix)] == Prefix:
        return Str.lstrip(Prefix)
    else:
        return str(Str)

def RStrip(Str, Suffix):
    if Str[- len(Suffix):] == Suffix:
        return Str.rstrip(Suffix)
    else:
        return str(Str)

def HasSuffix(Str, Suffix: str):
    if Suffix.startswith("."):
        Suffix = Suffix.lstrip(".")
    MatchPattern = re.compile(r'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    _HasSuffix = MatchResult is not None
    return _HasSuffix

def RemoveSuffixIfExist(Str, Suffix):
    MatchPattern = re.compile(rf'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        return str(Str)
    else:
        return MatchResult.group(1)
    
RemoveStrSuffixIfExist = RemoveSuffixIfExist    

def RemovePrefix(Str, Prefix:str, MustMatch=False):
    MatchPattern = re.compile(rf'%s(.*)'%Prefix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        if MustMatch:
            raise Exception('%s does not have suffix %s'%(Str, Prefix))
            # return None
        else:
            return Str
    else:
        return MatchResult.group(1)

def RemoveSuffix(Str, Suffix, MustMatch=True):
    MatchPattern = re.compile(rf'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        if MustMatch:
            raise Exception('%s does not have suffix %s'%(Str, Suffix))
            # return None
        else:
            return Str
    else:
        return MatchResult.group(1)

def Bytes2Str(Bytes:bytes, Encoding="utf-8", ErrorHandleMethod="RaiseException"):
    if ErrorHandleMethod in ["RaiseExcetion"]:
        errors = "strict"
    else:
        errors = "replace"  

    return Bytes.decode(Encoding, errors=errors)

def Str2Bytes(Str:str, Encoding="utf-8"):

    return Str.encode(Encoding)

import re
def SelectStrWithPatternFromList(List, Pattern):
    StrPattern = re.compile(StrPattern)
    for Str in List:
        assert isinstance(Str, str)
        
def Test():
    # python 3 string all uses unicode.
    Str = "你好，世界！"

def CodePoint2Char(CodePointInt):
    # CodePointNum: Int
    return chr(CodePointInt)
Int2Char = UnicodePoint2Char = CodePoint2Char

def Char2CodePoint(Char):
    return ord(Char)
Char2Num = Char2UnicodePoint = Char2CodePoint

def CharListAZ():
    return list(string.ascii_uppercase)

def CharListaz():
    return list(string.ascii_lowercase)

def CharListazAZ():
    return list(Straz() + CharListAZ())

def CharList09():
    return list(string.digits)

def Str09():
    return string.digits

def CharListAZ09():
    return CharListAZ() + CharList09()

def Straz():
    return string.ascii_lowercase

def StrAZ():
    return string.ascii_uppercase

def CharListazAZ09():
    return list(Straz() + StrAZ() + Str09())

def RandomStrazAZ09(Length):    
    return "".join(DLUtils.math.RandomSelectRepeat(
        CharListazAZ09(), Length
    ))
def RandomStr(Length, CharList="a-z"):
    if isinstance(CharList, str):
        if CharList in ["a-z"]:
            CharList = list(string.ascii_lowercase)
        else:
            CharList = [Char for Char in CharList]
    # print(CharList)
    return "".join(DLUtils.math.RandomSelectRepeat(CharList, Length))

def NaturalCmp(StrA, StrB):
    # judge whether StrA > StrB
    if len(StrB) > len(StrA):
        return -1 # StrB is larger
    else:
        return StrA > StrB

def Bytes2Hex(Bytes):
    return Bytes.hex()

def HexStr2Bytes(HexStr):
    return bytes.fromhex(HexStr)

OutPipe = sys.__stdout__
IndentLevel = 0
PrintBuf = StringIO()
Write2StdOut = None
def SetFileStrOut(FilePath):
    DLUtils.file.StandardizeFilePath(FilePath)
    global FileStrOut
    FileStrOut = open(FilePath, "w")
    SetStdOut(FileStrOut)
    
def CloseFileStrOut(FileStrOut=None):
    ResetStdOut()
    if FileStrOut is None:
        try:
            FileStrOut.close()
        except Exception:
            pass
    else:
        FileStrOut.close()

def RedirectSysStdOutTo(Pipe):
    pass

def OutputTo(Pipe):
    global OutPipe
    global IndentLevel
    global Write2StdOut, WriteBytes2StdOut, WriteStr2StdOut
    sys.stdout = Pipe
    OutPipe = Pipe
    if hasattr(OutPipe, "buffer"):
        WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.buffer.write(Bytes)
        WriteStr2StdOut = lambda StdOut, Str: StdOut.buffer.write(Str.encode("utf-8"))
    else:
        WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.write(Bytes)
        WriteStr2StdOut = lambda StdOut, Str: StdOut.write(Str.decode("utf-8"))
Output2 = SetStdOut = OutputTo

def AddIndentLevel():
    global IndentLevel
    IndentLevel += 1
IncreaseIndent = IncreaseIndentLevel = AddIndent = AddIndentLevel

def DecreaseIndentLevel():
    global IndentLevel
    if IndentLevel > 0:
        IndentLevel -= 1
DecreaseIndent = DecreaseIndentLevel

def SetIndentLevel(Indent: int):
    global IndentLevel
    IndentLevel = Indent

def Write(*List, **Dict):
    global OutPipe, IndentLevel
    PrintTo(OutPipe, *List, Indent=IndentLevel, **Dict)

def SetStdErrOut(Pipe):
    SetStdOut(Pipe)
    sys.stderr = Pipe

def GetStdOut():
    global OutPipe
    return OutPipe
GetOutPipe = GetStdOut

def Print2StdErr(*List, **Dict):
    print(*List, file=sys.stderr, **Dict)

def ResetOutPipe():
    global OutPipe
    OutPipe = sys.__stdout__
    global Write2StdOut, WriteBytes2StdOut, WriteStr2StdOut
    if hasattr(OutPipe, "buffer"):
        WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.buffer.write(Bytes)
        WriteStr2StdOut = lambda StdOut, Str: StdOut.buffer.write(Str.encode("utf-8"))
    else:
        WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.write(Bytes.decode("utf-8"))
        WriteStr2StdOut = lambda StdOut, Str: StdOut.write(Str)
ResetStdOut = ResetOutPipe

def PrintStr2Pipe(Pipe, Str, Indent=None, Flush=True):
    if Pipe is None:
        global OutPipe
        Pipe = OutPipe
    if Indent is None:
        Indent = 0
    if Indent > 0:
        if Str.endswith("\n"):
            IsEndWithNewLine = True
            Str = Str.rstrip("\n")
        else:
            IsEndWithNewLine = False
        StrList = Str.split("\n")
        for Index, Line in enumerate(StrList):
            StrList[Index] = "".join(["\t" for _ in range(Indent)] + [Line])
        Str = "\n".join(StrList)
        if IsEndWithNewLine:
            Str = Str + "\n"
    if hasattr(Pipe, "buffer"):
        Pipe.buffer.write(Str.encode("utf-8"))
    else:
        Pipe.write(Str)
    if Flush and hasattr(Pipe, "flush"):
        Pipe.flush()

def PrintPIDTo(Pipe, Indent=None):
    Str = "PID: " + str(DLUtils.system.CurrentPID()) + "\n"
    PrintStr2Pipe(Pipe, Str, Indent=Indent)

def PrintCurrentTimeTo(Pipe, Indent=None, Format=None, Prefix="Time: "):
    if Indent is not None:
        assert isinstance(Indent, int), Indent
    TimeStr = DLUtils.time.CurrentTimeStr(Format=Format) + "\n"
    if Prefix is not None:
        TimeStr = Prefix + TimeStr
    PrintStr2Pipe(Pipe, TimeStr, Indent=Indent)
PrintTimeStrTo = PrintTimeStr2 = PrintCurrentTimeTo
PrintCurrentTimeStrTo = PrintCurrentTimeTo = PrintCurrentTime2 = PrintCurrentTimeTo

def PrintTimeStr(Indent=None, Format=None, Prefix="Time: "):
    global OutPipe
    if Indent is None:
        global IndentLevel
        Indent = IndentLevel
    PrintTimeStrTo(OutPipe, Indent=Indent, Format=Format, Prefix=Prefix)

import time
def PrintHeartBeatTo(Pipe, Indent=None):
    Count = 0
    _Indent = Indent
    while True:
        if Count % 10 == 9:
            End = "\n"
            PrintUTF8To(Pipe, str(Count), end=End, Indent=_Indent)
            _Indent = Indent
        elif Count:
            End = " "
            PrintUTF8To(Pipe, str(Count), end=End, Indent=_Indent)
            _Indent = None
        Count += 1
        time.sleep(1.0)

def PrintWithParam(*List, **Dict):
    PrintBuf.seek(0)
    PrintBuf.truncate(0)
    print(*List, **Dict, file=PrintBuf)
    Str = PrintBuf.getvalue()
    PrintBuf.flush()
    return Str

def PrintUTF8To(Pipe, *List, Indent=None, **Dict):
    Str = PrintWithParam(*List, **Dict)
    if Indent is None:
        global IndentLevel
        Indent = IndentLevel
    PrintStr2Pipe(Pipe, Str, Indent=Indent)
PrintTo = PrintUTF8To

def Print2OutPipeUTF8(*List, Indent=None, **Dict):
    Str = PrintWithParam(*List, **Dict)
    Result = PrintStr2Pipe(OutPipe, Str, Indent=Indent)
    return Result
Print2OutPipe = Print2OutPipeUTF8
Print = Print2OutPipe

def PrintWithTimeStr(*List, Encoding="utf-8", Indent=None, **Dict):
    Buf = StringIO()
    # faster than reusing.
    # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
    print(*List, **Dict, file=Buf)
    Str = Buf.getvalue()
    if Str.endswith("\n"):
        Str = Str[:-1] + " time: %s."%DLUtils.system.CurrentTimeStr() + "\n"
    else:
        Str = Str + " time: %s."%DLUtils.system.CurrentTimeStr()
    Result = WriteBytes2StdOut(OutPipe, Str.encode(Encoding))
    OutPipe.flush()
    return Result
try:
    from bitarray import bitarray
except Exception:
    if DLUtils.Verbose:
        warnings.warn("lib bitarray not found")
    pass
else:
    def Bytes201Str(Bytes: bytes, MostSignificantBit="Right"):
        if MostSignificantBit in ["Right", "right", "R", "r"]:
            bits = bitarray(endian='big')
        else:
            bits = bitarray(endian="small")
        bits.frombytes(Bytes)
        return bits.to01()
    ByteArrayTo01Str = Bytes201Str

def Int201Str(Int, _0bPrefix=False, LeadingZero=False, Length=None):
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
IntTo01Str = Int201Str

def Str012Int(String, MostSignificantBit="Right", BigEndian=False):
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

ZeroOneStrToInt = ZeroOneStr2Int = Str01ToInt = Str01ToInt = String01ToInt = Str012Int