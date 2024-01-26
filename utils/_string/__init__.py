import sys
import string, re
import warnings, time
import DLUtils
from io import StringIO

# String Related Functions
def LStrip(Str, Prefix):
    if Str[0:len(Prefix)] == Prefix:
        return Str.lstrip(Prefix)
    else:
        return str(Str)

def RStrip(Str: str, Suffix):
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

def SelectStrWithPatternFromList(List, StrPattern):
    StrPattern = re.compile(StrPattern)
    for Str in List:
        assert isinstance(Str, str)

def CodePoint2Char(CodePointInt):
    # CodePointNum: Int
    return chr(CodePointInt)
Int2Char = UnicodePoint2Char = CodePoint2Char

def Char2CodePoint(Char):
    return ord(Char)
Char2Num = Char2UnicodePoint = Char2CodePoint

def CharListAZ():
    return list(string.ascii_uppercase)
def StrAZ():
    return string.ascii_uppercase

def CharListaz():
    return list(string.ascii_lowercase)
def Straz():
    return string.ascii_lowercase

def CharListazAZ():
    return list(StrAZ()() + CharListAZ())

def CharList09():
    return list(string.digits)

def StrDigit():
    return string.digits
Str09 = StrDigit

def CharListAZ09():
    return CharListAZ() + CharList09()



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

def ByteArrayToHex(ByteArray: bytes):
    return ByteArray.hex()
Bytes2Hex = ByteArrayToHex

def ByteArrayTo01Str(ByteArray: bytes):
    StrList = []
    for Byte in ByteArray:
        StrList.append(IntTo01Str(Byte, Prefix=None, DigitNum=8))
    return "".join(StrList)
Bytes201Str = BytesTo01Str = ByteArray201Str = ByteArrayTo01Str

def ToHex(Var, Prefix="0x", Case="Upper", MostSignificantByte="Left", DigitNum=None):
    if isinstance(Var, bytes):
        return ByteArrayToHex(Var, Prefix=Prefix, Case=Case, MostSignificantByte=MostSignificantByte)
    elif isinstance(Var, int):
        return IntToHex(Var, Prefix=Prefix, Case=Case, DigitNum=DigitNum)
    else:
        raise Exception()
 
def IntToHex(Int, Prefix="", Case="Upper", DigitNum=None):
    if Case in ["upper", "Upper", "u", "U"]:
        if DigitNum is None:
            return Prefix + "%X"%Int
        else:
            assert isinstance(DigitNum, int)
            return Prefix + ("%X"%Int).zfill(DigitNum)
    elif Case in ["lower", "Lower", "l", "L"]:
        if DigitNum is None:
            return Prefix + "%x"%Int
        else:
            assert isinstance(DigitNum, int)
            return Prefix + ("%X"%Int).zfill(DigitNum)
    else:
        raise Exception()

def IntTo01Str(Int: int, Prefix=None, DigitNum=None):
    if Prefix is None:
        Prefix = ""
    if Int < 0:
        Prefix += "-"
        Int = -Int
    _DigitList = []

    if Int == 0:
        _DigitList = ["0"]
    else:
        while(Int > 0):
            if Int % 2 == 0:
                _DigitList.append("0")
            else:
                _DigitList.append("1")
            Int = Int // 2
    
    if DigitNum is None:
        DigitList = _DigitList
    else:
        if DigitNum > len(_DigitList):
            for Index in range(DigitNum - len(_DigitList)):
                _DigitList.append("0")
    _DigitList = _DigitList[::-1]

    return Prefix + "".join(DigitList)
Int201Str = IntTo01Str

def Str01ToInt(Str, MostSignificantBit="Right"):
    Int = 0
    if MostSignificantBit in ["right", "Right", "r", "R"]: # big endian. natural. Str[0] is least significant bit.
        pass
    else:
        Str = Str[::-1]
    Base = 1
    for Index in len(Str):
        if Str[Index] == "0":
            pass
        elif Str[Index] == "1":
            Int += Base
        else:
            raise Exception()
        Base *= 2
    return Int
ZeroOneStrToInt = ZeroOneStr2Int = String01ToInt = Str012Int = Str01ToInt

def ByteArrayToHex(ByteArray, Prefix="0x", Case="Upper", MostSignificantByte="Left", Endian=None):
    if MostSignificantByte in ["right", "Right", "r", "R"]: # ByteArray[0] is least significant
        pass
    else:
        ByteArray = ByteArray[::-1]    
    Base = 1
    Int = 0
    for Byte in ByteArray:
        IntCurrentByte = ByteArrayToHex(ByteArray, MostSignificantByte=MostSignificantByte, Endian=Endian)
        Int += Base * IntCurrentByte
        Base *= 256
        
    return IntToHex(Int, Prefix=Prefix, Case=Case)
Bytes2Hex = BytesToHex = ByteArrayToHex

def Address(Var, DigitNum=16):
    return IntToHex(id(Var), Case="Upper", DigitNum=DigitNum)

def ByteToHex(Byte: bytes):
    
    return

def Bytes2Int(Bytes, LeastSignificantByte="Left", Endian=None):
    if Endian is not None:
        if Endian in ["Little", "little"]:
            byteorder = "little"
        else:
            byteorder = "big"
    else:
        if LeastSignificantByte in ["Left", "left"]:
            byteorder = "little"
        else:
            byteorder = "big"
    return int.from_bytes(Bytes, byteorder=byteorder, signed=False)
    
def HexStr2Bytes(HexStr):
    return bytes.fromhex(HexStr)

from collections import defaultdict
class OutPipeWriter:
    def __init__(self, OutPipe):
        self.OutPipe = OutPipe
        self.Indent = 0
        self.PrintCounterDict = defaultdict(lambda: 0)
        return
    def IncreaseIndent(self):
        self.Indent += 1
        return self
    def DecreaseIndent(self):
        if self.Indent > 0:
            self.Indent -= 1
        return self
    def PrintWithouthIndent(self, *List, **Dict):
        Result = PrintStrTo(self.OutPipe, *List, Indent=0, **Dict)
        return Result
    def PrintWithIncreaseIndent(self, *List, **Dict):
        Result = self.Print(*List, Indent=self.Indent + 1, **Dict)
        return Result
    def Print(self, *List, Indent=None, **Dict):
        if Indent is None:
            Indent = self.Indent
        Result = PrintStrTo(self.OutPipe, *List, Indent=Indent, **Dict)
        return Result
    print = Print
    def PrintEvery(self, Num, *List, Indent=None, **Dict):
        Count = self.PrintCounterDict[Num]
        Count += 1
        if Count >= Num:
            Result = self.Print(*List, Indent=Indent, **Dict)
            Count = 0
        else:
            Result = None
        self.PrintCounterDict[Num] = Count
        return Result
    def SetPrintCounterToZero(self):
        self.PrintCounterDict = defaultdict(lambda: 0)
    ResetPrintCounter = SetPrintCounterToZero

def GetLibOutPipeWriter():
    LibOutPipe = GetLibOutPipe()
    return OutPipeWriter(LibOutPipe)

def SetFileStrOut(FilePath):
    FilePath = DLUtils.StandardizeFilePath(FilePath)
    FileStrOut = open(FilePath, "w")
    return FileStrOut
    
def CloseFileStrOut(FileStrOut=None):
    FileStrOut.close()

def RedirectSysStdOutTo(Pipe):
    pass

def CheckIsStdOutExists():
    try:
        print("DLUtils: checking if stdout exists.", file=sys.__stdout__, flush=True)
        return True
    except Exception:
        return False

# def OutputTo(Pipe):
#     global OutPipe
#     global IndentLevel
#     global Write2StdOut, WriteBytes2StdOut, WriteStr2StdOut
#     # sys.stdout = Pipe
#     OutPipe = Pipe
#     if hasattr(OutPipe, "buffer"):
#         WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.buffer.write(Bytes)
#         WriteStr2StdOut = lambda StdOut, Str: StdOut.buffer.write(Str.encode("utf-8"))
#     else:
#         WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.write(Bytes)
#         WriteStr2StdOut = lambda StdOut, Str: StdOut.write(Str.decode("utf-8"))

def AddLibIndent():
    global LibIndent
    IndentLevel += 1
IncreaseLibIndentLevel = IncreaseLibIndent = AddLibIndentLevel = AddLibIndent

def DecreaseLibIndent():
    global LibIndent
    if LibIndent > 0:
        LibIndent -= 1
DecreaseLibIndentLevel = DecreaseLibIndent

def SetLibIndent(Indent: int):
    global LibIndent
    LibIndent = Indent
SetLibIndentLevel = SetLibIndent

def SetSysStdErr(Pipe):
    sys.stderr = Pipe

def GetCurrentSysStdOut():
    return sys.stdout
GetStdOut = GetCurrentSysStdOut

def PrintToStdErr(*List, **Dict):
    print(*List, file=sys.stderr, **Dict)
Print2StdErr = PrintToStdErr

def ResetLibOutPipe():
    global LibOutPipe
    LibOutPipe = sys.__stdout__
    # if hasattr(OutPipe, "buffer"):
    #     WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.buffer.write(Bytes)
    #     WriteStr2StdOut = lambda StdOut, Str: StdOut.buffer.write(Str.encode("utf-8"))
    # else:
    #     WriteBytes2StdOut = lambda StdOut, Bytes: StdOut.write(Bytes.decode("utf-8"))
    #     WriteStr2StdOut = lambda StdOut, Str: StdOut.write(Str)

def PrintStrToPipe(
        Pipe,
        Str: str,
        Indent=None,
        Flush=True,
        IndentStr="    " # "\t"
    ):
    if Pipe is None:
        Pipe = GetLibOutPipe()
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
            StrList[Index] = "".join([IndentStr for _ in range(Indent)] + [Line])
        Str = "\n".join(StrList)
        if IsEndWithNewLine:
            Str = Str + "\n"
    if hasattr(Pipe, "buffer"):
        Pipe.buffer.write(Str.encode("utf-8"))
    else:
        Pipe.write(Str)
    if Flush and hasattr(Pipe, "flush"):
        Pipe.flush()
PrintStr2Pipe = PrintStrToPipe

def PrintPIDTo(Pipe, Indent=None):
    Str = "PID: " + str(DLUtils.system.CurrentPID()) + "\n"
    PrintStrToPipe(Pipe, Str, Indent=Indent)

def PrintPIDToPipeAndStdOut(Pipe, Indent=None):
    PrintPIDTo(Pipe, Indent=Indent)
    PrintPIDTo(sys.__stdout__, Indent=Indent)

def PrintCurrentTimeTo(Pipe, Indent=None, Format=None, Prefix="Time: "):
    if Indent is not None:
        assert isinstance(Indent, int), Indent
    TimeStr = DLUtils.time.CurrentTimeStr(Format=Format) + "\n"
    if Prefix is not None:
        TimeStr = Prefix + TimeStr
    PrintStr2Pipe(Pipe, TimeStr, Indent=Indent)
PrintTimeStrTo = PrintTimeStr2 = PrintCurrentTimeTo
PrintCurrentTimeStrTo = PrintCurrentTimeTo = PrintCurrentTime2 = PrintCurrentTimeTo

def PrintCurrentTimeToPipeAndStdOut(Pipe, Indent=None, Format=None, Prefix="Time: "):
    PrintCurrentTimeTo(Pipe, Indent=Indent, Format=Format, Prefix=Prefix)
    PrintCurrentTimeTo(sys.__stdout__, Indent=Indent, Format=Format, Prefix=Prefix)

def PrintTimeStr(Indent=None, Format=None, Prefix="Time: "):
    global OutPipe
    if Indent is None:
        global IndentLevel
        Indent = IndentLevel
    PrintTimeStrTo(OutPipe, Indent=Indent, Format=Format, Prefix=Prefix)


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

def GetLibOutPipe():
    if not "LibOutPipe" in globals():
        global LibOutPipe
        LibOutPipe = sys.__stdout__

    return LibOutPipe

IsPrintBufInit = False

def GetPrintBuf():
    global PrintBuf
    global IsPrintBufInit
    if IsPrintBufInit:
        return PrintBuf
    else:
        IsPrintBufInit = True
        PrintBuf = StringIO()
        return PrintBuf

def PrintWithParam(*List, **Dict):
    # faster than reusing.
    # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
    PrintBuf = StringIO()
    PrintBuf.seek(0)
    PrintBuf.truncate(0)
    print(*List, **Dict, file=PrintBuf)
    Str = PrintBuf.getvalue()
    PrintBuf.flush()
    del PrintBuf
    return Str

def PrintStrToStdOut(*List, Indent=None, **Dict):
    return PrintStrTo(sys.__stdout__, *List, Indent=Indent)
PrintUTF8ToStdOut = PrintStrToStdOut

def PrintStr8ToStdErr(*List, Indent=None, **Dict):
    return PrintStrTo(sys.__stderr__, *List, Indent=Indent)
PrintUTF8ToStdErr = PrintStr8ToStdErr

def PrintStrTo(Pipe, *List, Indent=None, **Dict):
    Str = PrintWithParam(*List, **Dict)
    if Indent is None:
        Indent = 0
    PrintStr2Pipe(Pipe, Str, Indent=Indent)
PrintTo = PrintToPipe = PrintUTF8To = WriteTo = WriteStrTo = OutputTo = PrintStrTo

def PrintToPipeAndStdOut(Pipe, *List, Indent=None, **Dict):
    PrintUTF8ToStdOut(*List, Indent=Indent, **Dict)
    PrintUTF8To(Pipe, *List, Indent=Indent, **Dict)

def PrintToPipeAndStdErr(Pipe, *List, Indent=None, **Dict):
    PrintUTF8ToStdErr(*List, Indent=Indent, **Dict)
    PrintUTF8To(Pipe, *List, Indent=Indent, **Dict)

def PrintStrToLibOutPipe(*List, Indent=None, **Dict):
    LibOutPipe = GetLibOutPipe()
    Result = PrintStr2Pipe(LibOutPipe, *List, Indent=Indent, **Dict)
    return Result
Print = PrintStrToLibOutPipe
PrintToLibOutPipeUTF8 = PrintToLibOutPipe = PrintStrToLibOutPipe

def PrintWithTimeStr(*List, Encoding="utf-8", Indent=None, OutPipe=None, **Dict):
    Str = PrintWithParam(*List, **Dict)
    if Str.endswith("\n"):
        Str = Str[:-1] + " time: %s."%DLUtils.system.CurrentTimeStr() + "\n"
    else:
        Str = Str + " time: %s."%DLUtils.system.CurrentTimeStr()
    if OutPipe is None:
        OutPipe = GetLibOutPipe()
    Result = PrintStrTo(OutPipe, Indent=Indent)
    OutPipe.flush()
    return Result
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bitarray import bitarray
else:
    bitarray = DLUtils.LazyFromImport("bitarray", "bitarray")

def RemoveStartEndEmptySpaceChars(Str):
    Str = re.match(r"\s*([\S].*)", Str).group(1)
    Str = re.match(r"(.*[\S])\s*", Str).group(1)
    return Str
RemoveHeadTailWhiteChars = RemoveStartEndEmptySpaceChars
