import re
import string
import random
import DLUtils

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

def HasSuffix(Str, Suffix):
    MatchPattern = re.compile(r'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    return MatchResult is None

def RemoveSuffixIfExist(Str, Suffix):
    MatchPattern = re.compile(rf'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        return str(Str)
    else:
        return MatchResult.group(1)
    
RemoveStrSuffixIfExist = RemoveSuffixIfExist    

def RemoveSuffix(Str, Suffix, MustMatch=True):
    MatchPattern = re.compile(rf'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        if MustMatch:
            #raise Exception('%s does not have suffix %s'%(Str, Suffix))
            return None
        else:
            return Str
    else:
        return MatchResult.group(1)

def Bytes2Str(Bytes, Encoding="utf-8"):
    return Bytes.decode(Encoding)

def Str2Bytes(Str, Encoding="utf-8"):
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
    
    return "".join(DLUtils.math.RandomSelectRepeat(CharList, Length))

def Bytes2Hex(Bytes):
    return Bytes.hex()

def HexStr2Bytes(HexStr):
    return bytes.fromhex(HexStr)