import re

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

def RemoveSuffix(Str, Suffix, MustMatch=True):
    MatchPattern = re.compile(r'(.*)%s'%Suffix)
    MatchResult = MatchPattern.match(Str)
    if MatchResult is None:
        if MustMatch:
            #raise Exception('%s does not have suffix %s'%(Str, Suffix))
            return None
        else:
            return Str
    else:
        return MatchResult.group(1)

def Bytes2Str(Bytes, Format="utf-8"):
    return _str(Bytes, encoding = "utf-8")

def Str2Bytes(Str, Format="utf-8"):
    return Str.decode(Format)