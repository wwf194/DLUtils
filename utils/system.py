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
IsWin = IsWindows = IsWindowsSystem

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


try:
    from ._time import TimeStamp2DateTimeObj, TimeStr2Second, DateTimeObj2TimeStampFloat
    from ._time import CurrentTimeStampInt, CurrentTimeStr
except Exception:
    pass

def Stack2File(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    traceback.print_exc(file=open(FilePath, "w"))

def Test():
    TimeFloat = DateTimeObj2TimeStampFloat(datetime(1800, 1, 19, 3, 15, 14, 200))
    print(TimeFloat) # 2147451300.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeFloat)
import locale
import _thread
try:
    import chardet
    def RunPythonScript(FilePath, ArgList=[], PassPID=True,
            Async=True, # blocking / synchronous
            KillChildOnParentExit=False, GetResult=True,
            Method="os.popen"
        ):
        FilePath = DLUtils.file.StandardizeFilePath(FilePath)
        # os.system('chcp 65001')
        StrList = ["python -u"]
            # win: pythonw does not start the process.
        StrList.append("\"" + FilePath + "\"")
        StrList += ArgList
        if PassPID:
            StrList.append("--parent-pid")
            StrList.append("%d"%DLUtils.system.CurrentProcessID())
        Command = " ".join(StrList)
        ExitCode = 0
            # # unrecognized encoding problem
            # OutBytes = subprocess.check_output(
            #     Command, shell=True, stderr=subprocess.STDOUT
            # )
        if KillChildOnParentExit:
            if Async:
                if Method in ["os.popen", "os"]:
                    Result = os.popen("start /b cmd /c %s"%Command) # this is async.
                    print("after os.popen")
                    return True
                else:
                    try:
                        Result = subprocess.Popen(Command, stdout=subprocess.PIPE,
                            shell=True,
                            # preexec_fn=os.setpgrp # not supported on Windows
                        ) # run aynchronously
                        # (OutBytes, ErrBytes) = Result.communicate()

                    except subprocess.CalledProcessError as grepexc:                                                                                                   
                        ExitCode = grepexc.returncode
                        # grepexc.output
                    OutBytes = Result.stdout.read()
                    # OutBytes = OutStr.encode("utf-8")
                    # ExitCode = os.system(Command)
                    # OutBytes = "None".encode("utf-8")
                    
                    if GetResult:
                        
                        # if parent exit, child will also exit.
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
                                DLUtils.print("error ")
                                OutStr = OutBytes.hex()
                        return OutStr
            else: # blocking / synchronous
                # no window created
                    # win. parent .py script from .bat.
                    # win. parent .py script from cmd.
                Result = os.popen("start /b cmd /c %s"%Command) # this is async.
                # Result = os.popen(Command)
                # OutStr = Result.read()
                OutBytes = Result.buffer.read() # blocking. return after child process exit.
                OutStr = OutBytes.decode("utf-8")
                a = 1
        else:
            if Async: 
                # def ExecuteCommand(_Command):
                #     Result = subprocess.run(_Command, stdout=subprocess.PIPE) # subprocess.run is blocking.
                # try:
                #     _thread.start_new_thread(ExecuteCommand, 
                #         (Command,) # ArgList. must be tuple. (xxx) is not a tuple. (xxx,) is a tuple.
                #     )
                # except:
                #     return False
                # os.system("start /min %s"%Command)
                try:
                    Command = "".join(["start /b cmd /c"] + StrList)
                    DLUtils.print(Command)
                    Result = os.system(Command)
                        # call from console. no window.
                        # call from .bat. window created.
                except Exception:
                    DLUtils.print("error.")
                    DLUtils.system.PrintErrorStack()
            else:
                raise NotImplementedError()


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

def GetCmdArgList():
    return sys.argv
GetCmdArgList = CommandLineArgList = GetComamndLineArgList = GetCmdArgList()

def NewCmdArg(**Dict):
    NameList = []
    for Key in ["CmdName", "Name", "NameList", "NamePattern"]:
        NameList += DLUtils.ToList(Dict.get(Key, []))
    assert len(NameList) > 0
    
    NumStr = Dict.get("Num", "0~")
    if isinstance(NumStr, int):
        ValueNum = NumStr
    elif isinstance(NumStr, list):
        # ?
        pass
    else:
        if NumStr in ["0~1", "?", "01", "AtMostOne"]:
            ValueNum = "?"
        elif NumStr in ["1~", "+", "AtLeastOne"]:
            ValueNum = "+"
        elif NumStr in ["0~", "*", "Any"]:
            ValueNum = "*"
        else:
            raise Exception()
    
    Type = Dict.get("Type", str)
    
    PythonName = Dict.get("PythonName")
    if PythonName is None:
        PythonName = NameList[0]
    
    if isinstance(Type, type):
        pass
    elif isinstance(Type, str):
        if Type in ["str", "string"]:
            Type = str
        elif Type in ["int", "integer"]:
            Type = int
        else:
            Type = type(Type)
    else:
        DLUtils.ToList()

    ValueDefault = DLUtils.GetFromKeyList(Dict, "Default", "ValueDefault")
    return DLUtils.Dict(
        CmdName=NameList,
        ValueNum=ValueNum,
        ValueDefault=ValueDefault,
        Type=Type,
        PythonName=PythonName
    )

def ParseCmdArg(*CmdArgItem):
    import argparse
    parser = argparse.ArgumentParser()
    for Item in CmdArgItem:
        parser.add_argument(
            *Item.CmdName, #"-pp", "--parent-pid", "-parent-pid",
            dest=Item.PythonName,
            nargs=Item.ValueNum, 
            type=Item.Type, default=-1
        )
    CmdArg = parser.parse_args()
    return CmdArg


