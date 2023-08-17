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

CurrentPID = CurrentProcessID = CurrentProcessPID = GetCurrentProcessPID = GetCurrentProcessID

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
            Async=False, # blocking / synchronous
            KillChildOnParentExit=True, GetResult=False,
            Method="subprocess", StandardizeFilePath=False
        ):
        if StandardizeFilePath:
            FilePath = DLUtils.file.StandardizeFilePath(FilePath)
        # os.system('chcp 65001')
        StrList = ["python.exe", "-u"]
            # win: pythonw does not start the process.
        StrList.append("\"" + FilePath + "\"")
        StrList += ArgList
        if PassPID:
            PIDList = ["--parent-pid", "%d"%DLUtils.system.CurrentProcessID()]
            StrList += PIDList
        else:
            PIDList = []
        Command = " ".join(StrList)
        CommandList = ["python.exe", "-u", FilePath] + PIDList
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
                        Result = subprocess.Popen(
                            Command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True,
                            # preexec_fn=os.setpgrp # Linxu only.
                        ) # run aynchronously
                        # (OutBytes, ErrBytes) = Result.communicate()

                    except subprocess.CalledProcessError as grepexc:                                                                                                   
                        ExitCode = grepexc.returncode
                        # grepexc.output
                    # OutBytes = Result.stdout.read()
                    # OutBytes = OutStr.encode("utf-8")
                    # ExitCode = os.system(Command)
                    # OutBytes = "None".encode("utf-8")
                    
                    if GetResult:
                        Encode = "utf-8"
                        # if parent exit, child will also exit.
                        try:
                            # print("try decoding with utf-8")
                            OutStr = OutBytes.decode(Encode, errors="replace")
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
                Result = subprocess.run(
                    Command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    # preexec_fn=os.setpgrp # Linxu only.
                ) # run synchronously
                # Result = os.popen("start /b cmd /c %s"%Command) # this is async.
                # OutStr = Result.read()
                # OutBytes = Result.buffer.read() # blocking. return after child process exit.
                # OutStr = OutBytes.decode("utf-8")
                a = 1
        else: # KillChildOnParentExit=False. child continues to run when parent exit. 
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
                    CommandList = [
                        # "cmd", "/c"
                        # "start", "/b",
                    ] + CommandList
                    CommandStr = " ".join(
                        [
                            "start", "/b",
                            "cmd", "/c"
                        ] + StrList
                    )
                    
                    # os.spawnl(os.P_NOWAIT, *CommandList)
                    # print("after os.spawnl")
                    # DLUtils.print(CommandStr)
                    # Result = os.system(CommandStr)
                    # print("after os.system(...)")
                        # call from console. no window.
                        # call from .bat. window created.
                        # call from .bat
                            # child std out --> parent std out.
                            # child killed if parent exit.

                    # p = subprocess.Popen(CommandList,
                    #     # start_new_session=True,
                    #     stdout=subprocess.PIPE, # child stdout to another p.stdout.
                    #     stderr=subprocess.PIPE, # child stderr to another p.stderr.
                    #     # creationflags= subprocess.CREATE_NEW_CONSOLE
                    #     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    #     # shell=True
                    # )
                    # p = subprocess.run(CommandList, # blocking
                    #     # start_new_session=True,
                    #     stdout=subprocess.PIPE, # child stdout to another p.stdout.
                    #     stderr=subprocess.PIPE, # child stderr to another p.stderr.
                    #     # creationflags= subprocess.CREATE_NEW_CONSOLE
                    #     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    #     # shell=True
                    # )
                    # p = subprocess.call(
                    #     CommandList, shell=True
                    # )
                    # Popen("start ...") fails.
                        # start is cmd built-in command.
                    # print("after subprocess.Popen")

                except Exception:
                    DLUtils.print("error.")
                    DLUtils.system.PrintErrorStack()
                    a = 1
            else:
                raise NotImplementedError()
    RunPythonFile = RunPythonScript
except Exception:
    warnings.warn("lib chardet not found")

import traceback
def PrintErrorStackTo(Pipe, Indent=None):
    DLUtils.PrintUTF8To(Pipe, traceback.format_exc(), Indent=Indent)

PrintErrorStack2 = PrintErrorStackTo

def PrintErrorStack():
    print(traceback.format_exc())

def ErrorStackStr():
    return traceback.format_exc()

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


def ListNetworkInterface():
    psutil.net_if_stats()
    raise NotImplementedError()

