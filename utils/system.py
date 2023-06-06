# -*- coding: utf-8 -*-
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
    if psutil.pid_exists(PID):
        return True
    else:
        return False

def TeminateProcess(ExitCode):
    if IsWindowsSystem():
        os._exit(ExitCode)
    else:
        # sends a SIGINT to the main thread which raises a KeyboardInterrupt.
        # With that you have a proper cleanup. 
        os.kill(os.getpid(), signal.SIGINT)

import platform
def IsWindowsSystem():
    return platform.system() in ["Windows"]
IsWindows = IsWindowsSystem

ExistsProcess = ProcessExists

def GetCurrentProcessID():
    return os.getpid()

CurrentProcessID = CurrentProcessPID = GetCurrentProcessPID = GetCurrentProcessID

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

def TimeStr2Second(TimeStr):
    if isinstance(TimeStr, int):
        return TimeStr
    elif isinstance(TimeStr, float):
        return round(TimeStr)
    
    Pattern = r"(\d*\.\d*)([days|d|])"

    Result = re.match(Pattern, TimeStr)
    
    if Result is not None:
        NumStr = Result.group(1)
        UnitStr = Result.group(2)
        
        
    else:
        raise Exception()


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


def CurrentTimeStr(format="%Y-%m-%d %H-%M-%S", verbose=False):
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
GetCurrentTimeStamp = GetCurrentTimeStamp = GetCurrentTimeStampFloat

def GetCurrentTimeStampInt():
    return round(GetCurrentTimeStampFloat())
CurrentTimeStampInt = GetCurrentTimeStampInt

def GetCurrentTimeStampInt():
    return round(DateTimeObj2TimeStampFloat(datetime.datetime.now()))

TimeStampBase = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

def Test():
    TimeFloat = DateTimeObj2TimeStampFloat(datetime(1800, 1, 19, 3, 15, 14, 200))
    print(TimeFloat) # 2147451300.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeFloat)

import chardet
import locale
def RunPythonScript(FilePath, ArgList=[]):
    FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    # os.system('chcp 65001')
    StrList = ["python -u"]
    StrList.append("\"" + FilePath + "\"")
    StrList += ArgList
    Command = " ".join(StrList)
    ExitCode = 0
    try:
        # # unrecognized encoding problem
        # OutBytes = subprocess.check_output(
        #     Command, shell=True, stderr=subprocess.STDOUT
        # )
        Result = subprocess.Popen(Command, stdout=subprocess.PIPE, shell=True)
        # (OutBytes, ErrBytes) = Result.communicate()
        OutBytes = Result.stdout.read()
        # OutBytes = OutStr.encode("utf-8")
        
        # ExitCode = os.system(Command)
        # OutBytes = "None".encode("utf-8")     
    except subprocess.CalledProcessError as grepexc:                                                                                                   
        ExitCode = grepexc.returncode
        # grepexc.output
    # print(OutBytes.hex())
    try:
        # print("try decoding with utf-8")
        OutStr = OutBytes.decode("utf-8", errors="replace")
        # Encoding = chardet.detect(OutBytes)['encoding']
        # print("Encoding:", Encoding)
        # OutStr = OutBytes.decode(Encoding)
    except Exception:
        print("error in decoding OutBytes with %s."%Encode)
        try:
            Encode = locale.getdefaultlocale()[1]
            print("decoding with %s"%Encode)
            OutStr = OutBytes.decode(Encode)
        except Exception:
            OutStr = OutBytes.hex()
    return OutStr

RunPythonFile = RunPythonScript


import traceback
def PrintErrorStack():
    DLUtils.print(traceback.format_exc())