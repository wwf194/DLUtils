import re
import sys
import os
import time
import signal
import warnings
import traceback
from inspect import Traceback
import DLUtils

try:
    import psutil
    def ProcessExists(PID):
        if psutil.pid_exists(PID):
            return True
        else:
            return False
    def ProcessStartTime(PID, ReturnType="UnixTimeStamp"):
        assert ExistsProcess(PID)
        ProcessObj = psutil.Process(PID)
        TimeStamp = ProcessObj.create_time()
        if ReturnType in ["ReturnType"]:
            return TimeStamp
        else:
            return DLUtils.time.TimeStamp2Str(TimeStamp)

    ExistsProcess = ProcessExists
    def ListNetworkInterface():
        psutil.net_if_stats()
        raise NotImplementedError()
except Exception:
    warnings.warn("package psutil not found.")
    
def KillProcessbyPID(PID):
    os.kill(PID, signal.SIGTERM) #or signal.SIGKILL 
    # p = psutil.Process(pid)
    # p.terminate()  #or p.kill()
KillProcess = KillProcessbyID = KillProcessbyPID

def TeminateProcess(ExitCode):
    if IsWindowsSystem():
        os._exit(ExitCode)
    else:
        # sends a SIGINT to the main thread which raises a KeyboardInterrupt.
        # With that you have a proper cleanup. 
        os.kill(os.getpid(), signal.SIGINT)

def SelfPID():
    return os.getpid()
CurrentPID = CurrentProcessID = CurrentProcessPID = GetCurrentProcessPID = GetCurrentProcessID = SelfPID

def ParentPID():
    return os.getppid()
GetParentPID = ParentPID()

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

from ._time import TimeStamp2DateTimeObj, TimeStr2Second, DateTimeObj2TimeStampFloat
from ._time import CurrentTimeStampInt, GetCurrentTimeStampInt, CurrentTimeStr

def Stack2File(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    traceback.print_exc(file=open(FilePath, "w"))

import locale
from subprocess import PIPE, Popen
from threading  import Thread
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.x
class _ProcessObj:
    def __init__(self, Process:subprocess.Popen, Block=False, Daemon=False) -> None:
        self.Process = Process
        self._ExitCode = None
        self._HasFinished = False
        import sys
        ON_POSIX = 'posix' in sys.builtin_module_names
        if not Block and not Daemon:
            def enqueue_output(out, queue):
                for line in iter(out.readline, b''):
                    queue.put(line)
                out.close()
            q = Queue()
            t = Thread(target=enqueue_output, args=(self.Process.stdout, q))
            t.daemon = True # thread dies with the program
            t.start()
            self.t = t
            self.q = q
            self.StdOutStrList = []
            self.StdOutBytesList = []
        self._Block = Block
    def SetAttr(self, **Dict):
        for Key, Value in Dict.items():
            if Key in ["ReturnCode", "ExitCode"]:
                self.SetExitCode(Value)
            elif Key in ["StdOutBytes"]:
                self.SetStdOutBytes(Value)
            elif Key in ["StdErrBytes"]:
                self.SetStdErrBytes(Value)
            elif Key in ["HasFinished"]:
                self.SetFinished(Value)
            else:
                raise Exception()
        return self
    def StdOutBytes(self) -> bytes:
        if self._HasFinished:
            if(len(self.StdOutBytesList) == 0):
                return b''
            else:
                return b''.join(self.StdOutBytesList)
        else: # StdOut so far
            raise NotImplementedError()
    def StdOut(self):
        if hasattr(self, "_StdOutBytes"):
            self._StdOutStr = self._StdOutBytes.decode("utf-8")
            return self._StdOutStr
        else:
            try:
                self._StdOutBytes = self.Process.stdout.read()
                self._StdOutStr = self._StdOutBytes.decode("utf-8")
                return self._StdOutStr
            except Exception:
                pass
        if self._HasFinished:
            return "\n".join(self.StdOutStrList)
        else:
            # read line without blocking
            LineStrList = []
            try:
                while True:
                    Line = self.q.get_nowait() # or q.get(timeout=.1)
                    LineStr = Line.decode("utf-8")
                    self.StdOutBytesList.append(Line)
                    self.StdOutStrList.append(LineStr)
                    LineStrList.append(LineStr)
            except Empty:
                pass
            if len(self.StdOutStrList) == 0:
                return ""
            else:
                return "".join(self.StdOutStrList)
            # return Line.decode("utf-8")
    StdOutStr = StdOut
    def StdErr(self):
        if hasattr(self, "_StdErrBytes"):
            return self._StdErrBytes.decode("utf-8")
        raise NotImplementedError()
    StdErrStr = StdErr
    def SetStdErrBytes(self, _StdErrBytes: bytes):
        self._StdErrBytes = _StdErrBytes
        return self
    def SetStdOutBytes(self, _StdOutBytes: bytes):
        self._StdOutBytes = _StdOutBytes
        return self
    def SetExitCode(self, _ExitCode: int):
        self._ExitCode = _ExitCode
        return self
    def SetFinished(self, _HasFinished):
        assert isinstance(_HasFinished, bool)
        self._HasFinished = _HasFinished
        return self
    def HasFinished(self):
        if not self._HasFinished:
            ExitCode = self.Process.poll()
            if ExitCode is not None:
                self._ExitCode = ExitCode
                self._HasFinished = True
            else:
                self._HasFinished = False
        else:
            self._HasFinished = True
        return self._HasFinished
    def ExitCode(self):
        assert self.HasFinished()
        return self._ExitCode
    def PID(self):
        return self.Process.pid
    def Wait(self):
        if self._HasFinished:
            return self
        else:
            # self.Process.wait()
            while(True):
                if self.Process.poll() is not None:
                    return self
                time.sleep(0.1)

def RunPythonScript(FilePath, ArgList=[], PassPID=True,
        Async=None, Block=None,# blocking / synchronous
        Daemon=None, RunInBackGround=None,
        ChildDiesWithParent=None, KillChildOnParentExit=None,
        Method="subprocess", StandardizeFilePath=False, **Dict
    ):
    if Async is not None:
        assert Block is None or Block == True
        Block = not Async
    else:
        if Block is None:
            Block = True
        else:
            assert isinstance(Block, bool)

    if Block is None:
        Block = False
    else:
        assert isinstance(Block, bool)
    
    NonDaemon = DLUtils.GetFirstNotNoneValue(ChildDiesWithParent, KillChildOnParentExit)
    if NonDaemon is not None:
        assert isinstance(NonDaemon, bool)
        Daemon = not NonDaemon
    else:
        Daemon = DLUtils.GetFirstNotNoneValue(RunInBackGround, Daemon)
        if Daemon is None:
            Daemon = False
        else:
            assert isinstance(Daemon, bool)

    if StandardizeFilePath:
        FilePath = DLUtils.file.StandardizeFilePath(FilePath)
    # os.system('chcp 65001')
    if IsWindowsSystem():
        CommandList = ["python.exe", "-u"]
        # win: pythonw does not start the process.
    else:
        CommandList = ["python", "-u"]
    CommandList.append("\"" + FilePath + "\"")
    CommandList += ArgList
    if PassPID:
        PIDList = ["--parent-pid", "%d"%DLUtils.system.CurrentProcessID()]
        CommandList += PIDList
    else:
        PIDList = []
    Command = " ".join(CommandList)
    # if IsWindowsSystem():
    #     CommandList = ["python.exe", "-u", FilePath] + PIDList
    # else:
    #     CommandList = ["python", "-u", FilePath] + PIDList
    ExitCode = 0
        # # unrecognized encoding problem
        # OutBytes = subprocess.check_output(
        #     Command, shell=True, stderr=subprocess.STDOUT
        # )
    print("RunPythonSciprt:(%s). Block=%s. Daemon=%s"%(FilePath, Block, Daemon))
    if not Daemon: # child dies with parent
        if not Block: # non-blocking / asynchronous. kill child when parent exit.
            if Method in ["os.popen", "os"]:
                Process = os.popen("start /b cmd /c %s"%Command) # this is async.
                print("after os.popen")
                return True
            else: # subprocess.Popen
                try:
                    ON_POSIX = 'posix' in sys.builtin_module_names
                    Process = subprocess.Popen(
                        CommandList, # shell=False requires passing list rather than string.
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=False, # if True, Popen.pid will be the shell procee pid.
                        bufsize=1,
                        close_fds=ON_POSIX
                        # preexec_fn=os.setpgrp # Linxu only.
                    ) # run aynchronously
                    # (OutBytes, ErrBytes) = Result.communicate()
                    return _ProcessObj(Process, Block=Block, Daemon=Daemon)
                except subprocess.CalledProcessError as grepexc:                                                                                                   
                    ExitCode = grepexc.returncode
                    # grepexc.output
                # OutBytes = Result.stdout.read()
                # OutBytes = OutStr.encode("utf-8")
                # ExitCode = os.system(Command)
                # OutBytes = "None".encode("utf-8")
        else: # blocking / synchronous. kill child when parent exit.
            # no window created
                # win. parent .py script from .bat.
                # win. parent .py script from cmd.
            Process = subprocess.Popen(
                Command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # stdout=f, stderr=e,
                shell=True,
                # preexec_fn=os.setpgrp # Linxu only.
            ) # non-blocking
            StdOutBytes, StdErrBytes = Process.communicate() # blocking. return when child process return.
            ReturnCode = Process.returncode
            return _ProcessObj(
                Process=Process, Block=Block, Daemon=Daemon
            ).SetAttr(
                ReturnCode=ReturnCode,
                StdOutBytes=StdOutBytes,
                StdErrBytes=StdErrBytes,
                HasFinished=True
            )
            # Result = subprocess.run(
            #     Command,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     shell=True,
            #     # preexec_fn=os.setpgrp # Linxu only.
            # ) # run synchronously
            
            # Result = os.popen("start /b cmd /c %s"%Command) # this is async.
            # OutStr = Result.read()
            # OutBytes = Result.buffer.read() # blocking. return after child process exit.
            # OutStr = OutBytes.decode("utf-8")
            a = 1
    else: # KillChildOnParentExit=False. child continues to run when parent exit. 
        if not Block: 
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
                CommandList = ["python", FilePath]
                CommandStr = " ".join(
                    [
                        "start", "/b",
                        "cmd", "/c"
                    ] + StrList
                )
                CommandStr = " ".join(CommandList)
                print("CommandStr: %s"%CommandStr)
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

                devnull = open(os.devnull, 'wb')
                
                Process = subprocess.Popen(["python", 
                        # "\"" + FilePath + "\"",
                        FilePath,
                        "--disable-stdout"
                    ],
                    # start_new_session=True,
                    # stdout=devnull, stderr=devnull,
                    stdout=subprocess.PIPE, # child stdout to another p.stdout.
                    stderr=subprocess.PIPE, # child stderr to another p.stderr.
                    # creationflags= subprocess.CREATE_NEW_CONSOLE
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    shell=False
                )
                # still non-daemon if in VSCode debug mode.
                return _ProcessObj(
                    Process=Process, Daemon=Daemon, Block=Block
                )
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
        else:
            try:
                CommandList = ["python", 
                    # "\"" + FilePath + "\"",
                    FilePath,
                    "--disable-stdout"
                ]
                devnull = open(os.devnull, 'wb')
                Process = subprocess.Popen(["python", 
                        # "\"" + FilePath + "\"",
                        FilePath,
                        "--disable-stdout"
                    ],
                    # start_new_session=True,
                    # stdout=devnull, stderr=devnull,
                    stdout=subprocess.PIPE, # child stdout to another p.stdout.
                    stderr=subprocess.PIPE, # child stderr to another p.stderr.
                    # creationflags= subprocess.CREATE_NEW_CONSOLE
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    shell=False
                )
                # still non-daemon if in VSCode debug mode.
                return _ProcessObj(
                    Process=Process, Daemon=Daemon, Block=Block
                ).Wait()
            except Exception:
                DLUtils.print("error.")
                DLUtils.system.PrintErrorStack()

RunPythonFile = RunPythonScript

def PrintErrorStackTo(Pipe, Indent=None):
    DLUtils.PrintUTF8To(Pipe, traceback.format_exc(), Indent=Indent)
PrintErrorStack2 = PrintErrorStackTo

def PrintErrorStackWithInfoTo(Pipe, Indent=None):
    if Pipe is None:
        Pipe = DLUtils.GetOutPipe()
    if Indent is None:
        Indent = 0
    DLUtils.PrintTimeStrTo(Pipe, Indent=Indent)
    DLUtils.PrintPIDTo(Pipe, Indent=Indent + 1)
    DLUtils.PrintErrorStackTo(Pipe, Indent=Indent + 1)
PrintErrorStackWithInfo2 = PrintErrorStackWithInfoTo

def PrintErrorStack():
    print(traceback.format_exc())

def ErrorStackStr():
    return traceback.format_exc()
ErrorStack = ErrorStackStr

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

import datetime
def Test():
    TimeFloat = DateTimeObj2TimeStampFloat(datetime.datetime(1800, 1, 19, 3, 15, 14, 200))
    print(TimeFloat) # 2147451300.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeFloat)

def IsPython3():
    return sys.version_info.major == 3