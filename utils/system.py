import re
import sys
import os
import time
import signal
import psutil
import datetime
import traceback
from inspect import Traceback
import DLUtils

def KillProcessbyPID(PID):
    os.kill(PID, signal.SIGTERM) #or signal.SIGKILL 
    # p = psutil.Process(pid)
    # p.terminate()  #or p.kill()

def ProcessExists(PID):
    return 

def GetCurrentProcessPID():
    return os.getpid()

CurrentProcessPID = GetCurrentProcessPID

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
import subprocess

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

def RunPythonScript(FilePath, Args):
    ArgsList = ["python", FilePath, *Args]
    ArgsListStr = []
    for Arg in ArgsList:
        ArgsListStr.append(str(Arg))
    subprocess.call(ArgsListStr)
RunPythonFile = RunPythonScript

def CurrentTimeStr(format="%Y-%m-%d %H:%M:%S", verbose=False):
    TimeStr = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(TimeStr)
    return TimeStr
GetCurrentTime = GetTime = CurrentTimeStr

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

def Stack2File(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    traceback.print_exc(file=open(FilePath, "w"))

def DateTimeObj2TimeStampFloat(DataTimeObj):
    # return time.mktime(DataTimeObj.timetuple())
    # TimeStamp = DataTimeObj.timestamp() # >= Python 3.3
    TimeDiff = DataTimeObj - TimeStampBase
    TimeStamp = TimeDiff.total_seconds()
    return TimeStamp
    
def DateTimeObj2TimeStampInt(DataTimeObj):
    return round(time.mktime(DataTimeObj.timetuple()))

def TimeStamp2DateTimeObj(TimeStamp):
    # TimeStamp: float or int. unit: second.
    # millisecond is supported, and will not be floored.
    DateTimeObj = TimeStampBase + datetime.timedelta(seconds=TimeStamp)
    return DateTimeObj
    # might throw error for out of range time stamp.
    # DateTimeObj = date.fromtimestamp(TimeStamp)

def GetCurrentTimeStampFloat():
    return DateTimeObj2TimeStampFloat(datetime.datetime.now())
GetCurrentTimeStamp = GetCurrentTimeStampFloat

def GetCurrentTimeStampInt():
    return round(DateTimeObj2TimeStampFloat(datetime.datetime.now()))

TimeStampBase = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

def Test():
    TimeFloat = DateTimeObj2TimeStampFloat(datetime(1800, 1, 19, 3, 15, 14, 200))
    print(TimeFloat) # 2147451300.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeFloat)