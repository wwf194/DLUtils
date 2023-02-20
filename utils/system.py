from inspect import Traceback
import re
import sys
import time
import DLUtils

def ReportPyTorchInfo():
    import torch
    Report = ""
    if torch.cuda.is_available():
        Report += "Cuda is available"
    else:
        Report += "Cuda is unavailable"
    Report += "\n"
    Report += "Torch version:"+torch.__version__
    return Report

def GetSystemType():
    if "win" in sys.platform is not None:
        SystemType = 'Windows'
    elif "linux" in sys.platform is not None:
        SystemType = 'Linux'
    else:
        SystemType = 'Unknown'
    return SystemType
SystemType = SysType = GetSysType = GetSystemType

def ClassPathStr(Obj):
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    Class = Obj.__class__
    Module = Class.__module__
    if Module == 'builtins':
        return Module.__qualname__ # avoid outputs like 'builtins.str'
    _ClassPathStr = Module + '.' + Class.__qualname__
    return _ClassPathStr

import platform
 
def Time():
    return 

def GetSystemType2():
    SystemType = platform.system().lower()
    if SystemType in ['windows']:
        return "Windows"
    elif SystemType in ['linux']:
        return "Linux"
    else:
        raise Exception(SystemType)

def GetBytesInMemory(Obj):
    return sys.getsizeof(Obj)

def ReportMemoryOccupancy(Obj):
    ByteNum = GetBytesInMemory(Obj)
    return DLUtils.ByteNum2Str(Obj)



import subprocess
def RunPythonScript(FilePath, Args):
    ArgsList = ["python", FilePath, *Args]
    ArgsListStr = []
    for Arg in ArgsList:
        ArgsListStr.append(str(Arg))
    subprocess.call(ArgsListStr)
RunPythonFile = RunPythonScript

def GetTime(format="%Y-%m-%d %H:%M:%S", verbose=False):
    TimeStr = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(TimeStr)
    return TimeStr

import dateutil
def GetTimeDifferenceFromStr(TimeStr1, TimeStr2):
    Time1 = dateutil.parser.parse(TimeStr1)
    Time2 = dateutil.parser.parse(TimeStr2)

    TimeDiffSeconds = (Time2 - Time1).total_seconds()
    TimeDiffSeconds = round(TimeDiffSeconds)

    _Second = TimeDiffSeconds % 60
    Minute = TimeDiffSeconds // 60
    _Minute = Minute % 60
    Hour = Minute // 60
    TimeDiffStr = "%d:%02d:%02d"%(Hour, _Minute, _Second)
    return TimeDiffStr

import traceback
def Stack2File(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    traceback.print_exc(file=open(FilePath, "w"))