# -*- coding: utf-8 -*-
import re
import sys
import os
import time
import signal
import warnings
import traceback
from inspect import Traceback
import DLUtils

def KillProcessbyPID(PID):
    os.kill(PID, signal.SIGTERM) #or signal.SIGKILL 
    # p = psutil.Process(pid)
    # p.terminate()  #or p.kill()
try:
    import psutil
    def ProcessExists(PID):
        if psutil.pid_exists(PID):
            return True
        else:
            return False
    ExistsProcess = ProcessExists
except Exception:
    warnings.warn("lib psutil not found.")

KillProcess = KillProcessbyID = KillProcessbyPID

def TeminateProcess(ExitCode):
    if IsWindowsSystem():
        os._exit(ExitCode)
    else:
        # sends a SIGINT to the main thread which raises a KeyboardInterrupt.
        # With that you have a proper cleanup. 
        os.kill(os.getpid(), signal.SIGINT)


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

import platform
def IsWindowsSystem():
    return platform.system() in ["Windows"]
IsWindows = IsWindowsSystem


def GetSystemType():
    if "win" in sys.platform is not None:
        SystemType = 'Windows'
    elif "linux" in sys.platform is not None:
        SystemType = 'Linux'
    else:
        SystemType = 'Unknown'
    return SystemType
SystemType = SysType = GetSysType = GetSystemType
SystemType = GetSystemType()

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

from ._time import TimeStamp2DateTimeObj, TimeStr2Second, DateTimeObj2TimeStampFloat
from ._time import CurrentTimeStampInt, CurrentTimeStr



def Stack2File(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    traceback.print_exc(file=open(FilePath, "w"))

def Test():
    TimeFloat = DateTimeObj2TimeStampFloat(datetime(1800, 1, 19, 3, 15, 14, 200))
    print(TimeFloat) # 2147451300.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeFloat)
import locale
try:
    import chardet
    def RunPythonScript(FilePath, ArgList=[], PassPID=True):
        FilePath = DLUtils.file.StandardizeFilePath(FilePath)
        # os.system('chcp 65001')
        StrList = ["python -u"]
        StrList.append("\"" + FilePath + "\"")
        StrList += ArgList
        if PassPID:
            StrList.append("--parent-pid")
            StrList.append("%d"%DLUtils.system.CurrentProcessID())
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
except Exception:
    warnings.warn("lib chardet not found")

import traceback
def PrintErrorStack():
    DLUtils.print(traceback.format_exc())
    
def ExcStack2File(File):
    traceback.format_exc()
    
def Print2StdErr(Str):
    print(Str, file=sys.stderr)
    
def CloseWindow(*List, **Dict):
    if SystemType in ["Windows"]:
        DLUtils.backend.win.CloseWindow(*List, **Dict)
    else:
        raise Exception()