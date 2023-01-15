import os
import re
import sys
import functools
import threading
import time
import warnings
import pickle
import random

from typing import Iterable, List
#import pynvml
#from pynvml.nvml import nvmlDeviceOnSameBoard

#import timeout_decorator
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib import pyplot as plt
# from inspect import getframeinfo, stack
from DLUtils.attr import *
from DLUtils.file import *

#import utils.param as utils
import DLUtils.utils.json as json
import DLUtils.utils.math as math

import argparse
import traceback

def EmptyObj():
    return types.SimpleNamespace()
GenerateEmptyObj = EmptyObj

def JsonFile2ParamObj(FilePath):
    JsonDict = JsonFile2JsonDict(FilePath)
    Obj = utils.JsonStyleObj2Param(JsonDict)
    return Obj

def ParseCmdArgs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("task", nargs="?", default="DoTasksFromFile")
    parser.add_argument("-t", "--task", dest="task", nargs="?", default="CopyProject2DirAndRun")
    parser.add_argument("-t2", "--task2", dest="task2", default="DoTasksFromFile")
    parser.add_argument("-id", "--IsDebug", dest="IsDebug", default=True)

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
    # parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-2021-10-16-16:04:16/")
    parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")
    parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")
    # parser.add_argument("-tn", "--TaskName", dest="TaskName", default="AddAnalysis")

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-ms", "--MainScript", dest="MainScript", default="main.py")
    CmdArgs = parser.parse_args()
    return Namespace2PyObj(CmdArgs) # CmdArgs is of type namespace

def Namespace2PyObj(Namespace):
    return DLUtils.json.JsonObj2PyObj(Namespace2Dict(Namespace))

def Namespace2Dict(Namespace):
    return vars(Namespace)

def Dict2Namespace(Dict):
    return argparse.Namespace(Dict)

def ParseTaskName(task):
    if task in ["CleanLog", "CleanLog", "cleanlog"]:
        task = "CleanLog"
    elif task in ["DoTasksFromFile"]:
        task = "DoTasksFromFile"
    elif task in ["CopyProject2DirAndRun", "CopyProject2FolderAndRun", "CPFR"]:
        task = "CopyProject2FolderAndRun"
    else:
        pass
    return task

def Main(**kw):
    GlobalParam = DLUtils.GetGlobalParam()
    SetAttrs(GlobalParam, "time.StartTime", value=DLUtils.system.GetTime())
    
    #try:
    CmdArgs = kw.get("CmdArgs")
    if CmdArgs is None:
        CmdArgs = ParseCmdArgs()
    else:
        CmdArgs = Namespace2PyObj(CmdArgs)
    TaskFilePath = CmdArgs.TaskFile # All sciprt loads a task file, and keep doing tasks in it.
    CmdArgs.task = ParseTaskName(CmdArgs.task)
    task = CmdArgs.task
    if task in ["CleanLog"]:
        CleanLog()
    elif task in ["CleanFigure"]:
        CleanFigure()
    elif task in ["DoTasksFromFile"]:
        TaskObj = DLUtils.LoadTaskFile(TaskFilePath)
        Tasks = getattr(TaskObj, CmdArgs.TaskName)
        if not CmdArgs.IsDebug:
            try: # catch all unhandled exceptions
                DLUtils.DoTasks(Tasks, ObjRoot=DLUtils.GetGlobalParam())
            except Exception:
                DLUtils.Log(traceback.format_exc())
                raise Exception()
        else:
            DLUtils.DoTasks(Tasks, ObjRoot=DLUtils.GetGlobalParam())
    elif task in ["TotalLines"]:
        DLUtils.CalculateGitProjectTotalLines()
    elif task in ["QuickScript"]:
        QuickScript = kw.get("QuickScript")
        QuickScript(CmdArgs)
    elif task in ["CopyProject2FolderAndRun"]:
        GlobalParam = DLUtils.GetGlobalParam()
        CopyFilesAndDirs2DestDir(
            GlobalParam.config.Project.Files, 
            "./", 
            DLUtils.GetMainSaveDir() + "src/"
        )
        CmdArgs.task = CmdArgs.task2
        delattr(CmdArgs, "task2")
        DLUtils.system.RunPythonScript(
            DLUtils.GetMainSaveDir() + "src/" + CmdArgs.MainScript,
            ParsedArgs2CmdArgs(CmdArgs)
        )
    elif task in ["TotalLines"]:
        DLUtils.CalculateGitProjectTotalLines()
    else:
        raise Exception("Inavlid Task: %s"%CmdArgs.task)

    GlobalParam = DLUtils.GetGlobalParam()
    
    SetAttrs(GlobalParam, "time.EndTime", value=DLUtils.system.GetTime())
    DurationTime = DLUtils.system.GetTimeDifferenceFromStr(GlobalParam.time.StartTime, GlobalParam.time.EndTime)
    SetAttrs(GlobalParam, "time.DurationTime", value=DurationTime)
    _StartEndTime2File()

def _StartEndTime2File():
    GlobalParam = DLUtils.GetGlobalParam()
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-0-Time-Start:%s"%GlobalParam.time.EndTime)
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-1-Time- End :%s"%GlobalParam.time.StartTime)
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-2-Time-Duration:%s"%GlobalParam.time.DurationTime)

def ParsedArgs2CmdArgs(ParsedArgs, Exceptions=[]):
    CmdArgsList = []
    for Name, Value in ListAttrsAndValues(ParsedArgs, Exceptions=Exceptions):
        CmdArgsList.append("--%s"%Name)
        CmdArgsList.append(Value)
    return CmdArgsList

def CopyProjectFolder2Dir(DestDir):
    EnsureDir(DestDir)
    DLUtils.file.CopyFolder2DestDir("./", DestDir)
    return

def CopyProjectFolderAndRunSameCommand(Dir):
    CopyProjectFolder2Dir(Dir)

def CleanLog():
    DLUtils.file.RemoveAllFilesAndDirsUnderDir("./log/")

def CleanFigure():
    DLUtils.file.RemoveMatchedFiles("./", r".*\.png")

def ParseTaskList(TaskList, InPlace=True, **kw):
    TaskListParsed = []
    for Index, Task in enumerate(TaskList):
        if isinstance(Task, str):
            TaskParsed = DLUtils.PyObj({
                "Type": Task,
                "Args": {}
            })
            if InPlace:
                TaskList[Index] = TaskParsed
            else:
                TaskListParsed.append(TaskParsed)
        elif DLUtils.IsDictLikePyObj(Task):
            if hasattr(Task, "Type") and hasattr(Task, "Args"):
                if InPlace:
                    pass
                else:
                    TaskListParsed.append(Task)
            else:
                for key, value in ListAttrsAndValues(Task):
                    TaskParsed = DLUtils.PyObj({
                        "Type": key, "Args": value
                    })

                    if InPlace:
                        TaskList[Index] = TaskParsed
                    else:
                        TaskListParsed.append(TaskParsed)
        elif DLUtils.IsListLikePyObj(Task) or isinstance(Task, list) or isinstance(Task, tuple):
            TaskParsed = DLUtils.PyObj({
                "Type": Task[0],
                "Args": Task[1]
            })
            if InPlace:
                TaskList[Index] = TaskParsed
            else:
                TaskListParsed.append(TaskParsed)
        else:
            raise Exception(type(Task))
        
    if InPlace:
        return TaskList
    else:
        return DLUtils.PyObj(TaskListParsed)

def ParseTaskObj(TaskObj, Save=True, **kw):
    kw.setdefault("ObjRoot", DLUtils.GetGlobalParam())
    if isinstance(TaskObj, str):
        TaskObj = DLUtils.parse.ResolveStr(TaskObj, **kw)
    if DLUtils.IsDictLikePyObj(TaskObj):
        if hasattr(TaskObj, "__Tasks__"):
            TaskObj.__Tasks__ = ParseTaskList(TaskObj.__Tasks__, **kw)
            TaskList = TaskObj.__Tasks__
        else:
            TaskObj.__Tasks__ = ParseTaskList(ListAttrsAndValues(TaskObj), InPlace=False, **kw)
            for Attr, Value in ListAttrsAndValues(TaskObj, Exceptions=["__Tasks__"]):
                delattr(TaskObj, Attr)
            TaskList = TaskObj.__Tasks__
    elif DLUtils.IsListLikePyObj(TaskObj):
        TaskObj.__Tasks__ = ParseTaskList(TaskObj, InPlace=False, **kw)
        delattr(TaskObj, "__value__")
        TaskList = TaskObj.__Tasks__
    else:
        raise Exception()
    TaskObj.SetResolveBase(True)
    TaskList.SetResolveBase(False)
    for Index, Task in enumerate(TaskList):
        Task.SetResolveBase() # So that "&" in each Task resolves to the task object it is inside.

    if Save:
        DLUtils.json.PyObj2JsonFile(TaskList, DLUtils.GetMainSaveDir() + "task_loaded.jsonc")
    DLUtils.parse.ParsePyObjStatic(TaskObj, ObjCurrent=TaskList, ObjRoot=DLUtils.GetGlobalParam(), InPlace=True)
    if Save:
        DLUtils.json.PyObj2JsonFile(TaskList, DLUtils.GetMainSaveDir() + "task_parsed.jsonc")
    return TaskObj


def DoTasks(Tasks, **kw):
    if not kw.get("DoNotChangeObjCurrent"):
        kw["ObjCurrent"] = Tasks
    if isinstance(Tasks, str) and "&" in Tasks:
        Tasks = DLUtils.parse.ResolveStr(Tasks, **kw)
    Tasks = DLUtils.ParseTaskObj(Tasks)

    In = kw.get("In")
    if In is not None:
        Tasks.cache.In = DLUtils.PyObj(In)
    for Index, Task in enumerate(Tasks.__Tasks__):
        if not kw.get("DoNotChangeObjCurrent"):
            kw["ObjCurrent"] = Task
        #DLUtils.EnsureAttrs(Task, "Args", default={})
        DLUtils.DoTask(Task, **kw)

def DoTask(Task, **kw):
    ObjRoot = kw.setdefault("ObjRoot", None)
    ObjCurrent = kw.setdefault("ObjCurrent", None)
    #Task = DLUtils.parse.ParsePyObjDynamic(Task, RaiseFailedParse=False, InPlace=False, **kw)
    TaskType = Task.Type
    TaskArgs = Task.Args
    if isinstance(TaskArgs, str) and "&" in TaskArgs:
        TaskArgs = DLUtils.parse.ResolveStr(TaskArgs, kw)
    if TaskType in ["BuildObjFromParam", "BuildObjectFromParam"]:
        BuildObjFromParam(TaskArgs, **kw)
    elif TaskType in ["FunctionCall"]:
        DLUtils.CallFunctions(TaskArgs, **kw)
    elif TaskType in ["CallGraph"]:
        if hasattr(TaskArgs, "Router"):
            Router = TaskArgs.Router
        else:
            Router = TaskArgs
        if isinstance(Router, str):
            Router = DLUtils.parse.ResolveStr(Router)
        # Require that router is already parsed.
        #RouterParsed = DLUtils.router.ParseRouterStaticAndDynamic(Router, ObjRefList=[Router], **kw)
        InParsed = DLUtils.parse.ParsePyObjDynamic(TaskArgs.In, RaiseFailedParse=True, InPlace=False, **kw)
        #InParsed = DLUtils.parse.ParsePyObjDynamic(Router, RaiseFailedParse=True, InPlace=False, **kw)
        DLUtils.CallGraph(Router, InParsed)
    elif TaskType in ["RemoveObj"]:
        RemoveObj(TaskArgs, **kw)
    elif TaskType in ["LoadObjFromFile"]:
        LoadObjFromFile(TaskArgs, **kw)
    elif TaskType in ["LoadObj"]:
        DLUtils.LoadObj(TaskArgs, **kw)
    # elif TaskType in ["AddLibraryPath"]:
    #     AddLibraryPath(TaskArgs)
    elif TaskType in ["LoadJsonFile"]:
        LoadJsonFile(TaskArgs)
    elif TaskType in ["LoadParamFile"]:
        DLUtils.LoadParamFromFile(TaskArgs, ObjRoot=DLUtils.GetGlobalParam())
    elif TaskType in ["ParseParam", "ParseParamStatic"]:
        DLUtils.parse.ParseParamStatic(TaskArgs)
    elif TaskType in ["ParseParamDynamic"]:
        DLUtils.parse.ParseParamDynamic(TaskArgs)
    elif TaskType in ["BuildObj"]:
        DLUtils.BuildObj(TaskArgs, **kw)
    elif TaskType in ["BuildObjFromFile", "BuildObjectFromFile"]:
        DLUtils.BuildObjFromFile(TaskArgs, ObjRoot=DLUtils.GetGlobalParam())
    elif TaskType in ["BuildObjFromParam", "BuildObjectFromParam"]:
        DLUtils.BuildObjFromParam(TaskArgs, ObjRoot=DLUtils.GetGlobalParam())
    elif TaskType in ["SetTensorLocation"]:
        SetTensorLocation(TaskArgs)
    elif TaskType in ["Train"]:
        DLUtils.train.Train(
            TaskArgs,
            ObjRoot=DLUtils.GetGlobalParam(),
            Logger=DLUtils.GetDataLogger()
        )
    elif TaskType in ["DoTasks"]:
        _TaskList = DLUtils.ParseTaskObj(TaskArgs, ObjRoot=DLUtils.GetGlobalParam())
        DoTasks(_TaskList, **kw)
    elif TaskType in ["SaveObj"]:
        DLUtils.SaveObj(TaskArgs, ObjRoot=DLUtils.GetGlobalParam())

    else:
        DLUtils.AddWarning("Unknown Task.Type: %s"%TaskType)
        raise Exception(TaskType)

def GetTensorLocation(Method="auto"):
    if Method in ["Auto", "auto"]:
        Location = DLUtils.GetGPUWithLargestUseableMemory()
    else:
        raise Exception()
    return Location

def BuildObjFromParam(Args, **kw):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list):
        for Arg in Args:
            _BuildObjFromParam(Arg, **kw)
    elif isinstance(Args, DLUtils.PyObj):
        _BuildObjFromParam(Args, **kw)
    else:
        raise Exception()

def _BuildObjFromParam(Args, **kw):
    ParamPathList = DLUtils.ToList(Args.ParamPath)
    ModulePathList = DLUtils.ToList(Args.ModulePath)
    MountPathList = DLUtils.ToList(Args.MountPath)

    for ModulePath, ParamPath, MountPath, in zip(ModulePathList, ParamPathList, MountPathList):        
        param = DLUtils.parse.ResolveStr(ParamPath, kw)
        #Class = eval(ModulePath)
        #Obj = Class(param)
        Class = DLUtils.parse.ParseClass(ModulePath)
        Obj = Class(param)
        # Module = DLUtils.ImportModule(ModulePath)
        # Obj = Module.__MainClass__(param)

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def BuildObjFromFile(Args, **kw):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, list):
        for Arg in Args:
            _BuildObjFromFile(Arg, **kw)
    elif isinstance(Args, DLUtils.PyObj):
        _BuildObjFromFile(Args, **kw)
    else:
        raise Exception()

def _BuildObjFromFile(Args, **kw):
    ParamFilePathList = DLUtils.ToList(Args.ParamFilePath)
    ModulePathList = DLUtils.ToList(Args.ModulePath)
    MountPathList = DLUtils.ToList(Args.MountPath)

    for ModulePath, ParamFilePath, MountPath, in zip(ModulePathList, ParamFilePathList, MountPathList):        
        param = DLUtils.json.JsonFile2PyObj(ParamFilePath)
        Class = DLUtils.parse.ParseClass(ModulePath)
        Obj = Class(param)

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def BuildObj(Args, **kw):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list) or DLUtils.IsListLikePyObj(Args):
        for Arg in Args:
            _BuildObj(Arg, **kw)
    elif isinstance(Args, DLUtils.PyObj):
        _BuildObj(Args, **kw)
    else:
        raise Exception()

def _BuildObj(Args, **kw):
    ModulePathList = DLUtils.ToList(Args.ModulePath)
    MountPathList = DLUtils.ToList(Args.MountPath)

    for ModulePath, MountPath in zip(ModulePathList, MountPathList):
        Class = DLUtils.parse.ParseClass(ModulePath)
        Obj = Class()

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def RemoveObj(Args, **kw):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list):
        for Arg in Args:
            _RemoveObj(Arg, **kw)
    elif isinstance(Args, DLUtils.PyObj):
        _RemoveObj(Args, **kw)
    else:
        raise Exception()

def _RemoveObj(Args, **kw):
    MountPathList = DLUtils.ToList(Args.MountPath)

    for MountPath in MountPathList:
        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")
        MountPathList = MountPath.split(".")
        RemoveAttrs(eval(MountPathList[0]), MountPathList[1:])

def SaveObj(Args, **kw):
    SaveObjList = DLUtils.ToList(Args.SaveObj)
    SaveDirList = DLUtils.ToList(Args.SaveDir)

    for SaveObj, SaveDir in zip(SaveObjList, SaveDirList):
        if SaveDir in ["auto", "Auto"]:
            SaveDir = DLUtils.GetMainSaveDirForModule()
        Obj = DLUtils.parse.ResolveStr(SaveObj, **kw)
        Obj.Save(SaveDir)

def LoadObj(Args, **kw):
    SourcePathList = DLUtils.ToList(Args.SourcePath)
    MountPathList = DLUtils.ToList(Args.MountPath)

    for SourcePath, MountPath in zip(SourcePathList, MountPathList):
        Obj = DLUtils.parse.ResolveStr(SourcePath, **kw)
        MountObj(MountPath, Obj, **kw)

def LoadObjFromFile(Args, **kw):
    SaveNameList = DLUtils.ToList(Args.SaveName)
    MountPathList = DLUtils.ToList(Args.MountPath)
    SaveDirList = DLUtils.ToList(Args.SaveDir)

    SaveDirParsedList = []
    for SaveDir in SaveDirList:
        SaveDirParsedList.append(DLUtils.parse.ResolveStr(SaveDir, **kw))

    for SaveName, SaveDir, MountPath in zip(SaveNameList, SaveDirParsedList, MountPathList):
        ParamPath = SaveDir + SaveName + ".param.jsonc"
        assert DLUtils.FileExists(ParamPath)
        param = DLUtils.json.JsonFile2PyObj(ParamPath)
        DataPath = SaveDir + SaveName + ".data"
        if DLUtils.FileExists(DataPath):
            data = DLUtils.json.DataFile2PyObj(DataPath)
        else:
            data = DLUtils.EmptyPyObj()
        Class = DLUtils.parse.ParseClass(param.ClassPath)
        Obj = Class(param, data, LoadDir=SaveDir)
        MountObj(MountPath, Obj, **kw)

def LoadTaskFile(FilePath="./task.jsonc"):
    TaskObj = DLUtils.json.JsonFile2PyObj(FilePath)
    return TaskObj

def LoadJsonFile(Args):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, dict):
        _LoadJsonFile(DLUtils.json.JsonObj2PyObj(Args))
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadJsonFile(Arg)
    elif isinstance(Args, DLUtils.PyObj):
        _LoadJsonFile(Args)
    else:
        raise Exception()

def _LoadJsonFile(Args, **kw):
    Obj = DLUtils.json.JsonFile2PyObj(Args.FilePath)
    MountObj(Args.MountPath, Obj, **kw)

def SaveObj(Args):
    Obj = DLUtils.parse.ResolveStr(Args.MountPath, ObjRoot=DLUtils.GetGlobalParam()),
    Obj.Save(SaveDir=Args.SaveDir)

def IsClassInstance(Obj):
    # It seems that in Python, all variables are instances of some class.
    return

import types
def IsFunction(Obj):
    return isinstance(Obj, types.FunctionType) \
        or isinstance(Obj, types.BuiltinFunctionType)

from collections.abc import Iterable   # import directly from collections for Python < 3.3
def IsIterable(Obj):
    if isinstance(Obj, Iterable):
        return True
    else:
        return False
def IsListLike(List):
    if isinstance(List, list):
        return True
    elif isinstance(List, DLUtils.PyObj) and List.IsListLike():
        return True
    else:
        return False

def RemoveStartEndEmptySpaceChars(Str):
    Str = re.match(r"\s*([\S].*)", Str).group(1)
    Str = re.match(r"(.*[\S])\s*", Str).group(1)
    return Str

RemoveHeadTailWhiteChars = RemoveStartEndEmptySpaceChars

def RemoveWhiteChars(Str):
    Str = re.sub(r"\s+", "", Str)
    return Str

def TensorType(data):
    return data.dtype

def NpArrayType(data):
    if not isinstance(data, np.ndarray):
        return "Not an np.ndarray, but %s"%type(data)
    return data.dtype

def List2NpArray(data):
    return np.array(data)

def Dict2GivenType(Dict, Type):
    if Type in ["PyObj"]:
        return DLUtils.PyObj(Dict)
    elif Type in ["Dict"]:
        return Dict
    else:
        raise Exception(Type)

def ToSaveFormat(Data):
    if isinstance(Data, torch.Tensor):
        return ToNpArray(Data)
    else:
        return Data

def ToRunFormat(Data):
    if isinstance(Data, np.ndarray):
        return ToTorchTensor(Data)
    else:
        return Data

def ToNpArray(data, DataType=np.float32):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data, dtype=DataType)
    elif isinstance(data, torch.Tensor):
        return Tensor2NpArray(data)
    elif isinstance(data, float):
        return np.asarray([data],dtype=DataType)
    else:
        raise Exception(type(data))

def ToNpArrayOrNum(data, DataType=np.float32):
    if isinstance(data, float):
        return data
    if isinstance(data, int):
        return data
    data = ToNpArray(data)
    if data.size == 0: # empty array
        return None
    elif data.size == 1: # single element array
        return data.reshape(1)[0]
    else:
        return data

def ToNpArrayIfIsTensor(data):
    if isinstance(data, torch.Tensor):
        return DLUtils.ToNpArray(data), False
    else:
        return data, True

def ToPyObj(Obj):
    if isinstance(Obj, DLUtils.json.PyObj):
        return Obj
    else:
        return DLUtils.PyObj(Obj)

def ToTrainableTorchTensor(data):
    if isinstance(data, np.ndarray):
        return NpArray2Tensor(data, RequiresGrad=True)
    elif isinstance(data, list):
        return NpArray2Tensor(List2NpArray(data), RequiresGrad=True)
    elif isinstance(data, torch.Tensor):
        data.requires_grad = True
        return data
    else:
        raise Exception(type(data))

def ToTorchTensor(data):
    if isinstance(data, np.ndarray):
        return NpArray2Tensor(data)
    elif isinstance(data, list):
        return NpArray2Tensor(List2NpArray(data))
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise Exception(type(data))

def ToTorchTensorOrNum(data):
    if isinstance(data, float):
        return data
    elif isinstance(data, int):
        return data
    else:
        return ToTorchTensor(data)

def _1DTo2D(data):
    # turn 1-D data to 2-D data for visualization
    DimensionNum = DLUtils.GetDimensionNum(data)
    assert DimensionNum == 1, DimensionNum

    dataNum = data.shape[0]
    RowNum, ColNum = DLUtils.plot.ParseRowColNum(dataNum)
    mask = np.ones((RowNum, ColNum), dtype=np.bool8)

    maskNum = RowNum * ColNum - dataNum
    RowIndex, ColIndex = RowNum - 1, ColNum - 1 # Start from point at right bottom.
    
    for Index in range(maskNum):
        mask[RowIndex, ColIndex] = False
        ColIndex -= 1
    if maskNum > 0:
        dataEnd = np.zeros(maskNum,dtype=data.dtype)
        #dataEnd[:] = np.nan
        data = np.concatenate([data, dataEnd])
    data = data.reshape((RowNum, ColNum))
    return data, mask

def FlattenNpArray(data):
    return data.flatten()

def EnsureFlatNp(data):
    return data.flatten()

EnsureFlat = EnsureFlatNp

def NpArray2Tensor(data, Location="cpu", DataType=torch.float32, RequiresGrad=False):
    data = torch.from_numpy(data)
    data = Tensor2GivenDataType(data, DataType)
    data = data.to(Location)
    data.requires_grad = RequiresGrad
    return data

def NpArray2List(data):
    return data.tolist()

def NpArray2Str(data):
    return np.array2string(data)

def ToStandardizeTorchDataType(DataType):
    if DataType in ["Float", "float"]:
        return torch.float32
    elif DataType in ["Double", "double"]:
        return torch.float64

def ToGivenDataTypeTorch(data, DataType=torch.float32):
    if data.dtype==DataType:
        return data
    else:
        return data.to(DataType)
Tensor2GivenDataType = ToGivenDataTypeTorch

def DeleteKeysIfExist(Dict, Keys):
    for Key in Keys:
        if Key in Dict:
            Dict.pop(Key)
    return Dict

def ParseDataTypeNp(DataType):
    if isinstance(DataType, str):
        # if DataType in ["np.float32"]:
        #     return np.float32
        # elif DataType in ["np.int8"]:
        #     return np.int8
        # else:
        #     raise Exception(DataType)
        #     # To Be Implemented
        return eval(DataType)
    else:
        return DataType

def ToGivenDataTypeNp(data, DataType):
    DataType = DLUtils.ParseDataTypeNp(DataType)
    return data.astype(DataType)

def TorchTensor2NpArray(data):
    data = data.detach().cpu().numpy()
    return data # data.grad will be lost.
Tensor2NpArray = TorchTensor2NpArray

def Tensor2Str(data):
    return NpArray2Str(Tensor2NpArray(data))

def Tensor2File(data, SavePath):
    EnsureFileDir(SavePath)
    np.savetxt(SavePath, DLUtils.Tensor2NpArray(data))

def Tensor2NumpyOrFloat(data):
    try:
        _data = data.item()
        return _data
    except Exception:
        pass
    data = data.detach().cpu().numpy()
    return data

def List2NpArray(data, Type=None):
    if Type is not None:
        return np.array(data, dtype=Type)
    else:
        return np.array(data)

def ToList(Obj):
    if isinstance(Obj, list):
        return Obj
    elif isinstance(Obj, np.ndarray):
        return Obj.tolist()
    elif isinstance(Obj, torch.Tensor):
        return NpArray2List(Tensor2NpArray(Obj))
    elif DLUtils.IsListLikePyObj(Obj):
        return Obj.ToList()
    elif isinstance(Obj, dict) or DLUtils.IsDictLikePyObj(Obj):
        raise Exception()
    else:
        return [Obj]

def ToDict(Obj):
    if isinstance(Obj, dict):
        return dict(Obj)
    elif isinstance(Obj, DLUtils.PyObj):
        return Obj.ToDict()
    else:
        raise Exception(type(Obj))

import functools
def SortListByCmpMethod(List, CmpMethod):
    # Python3 no longer supports list.sort(cmp=...)
    List.sort(key=functools.cmp_to_key(CmpMethod))

# def GetFunction(FunctionName, ObjRoot=None, ObjCurrent=None, **kw):
#     return eval(FunctionName.replace("&^", "ObjRoot.").replace("&", "ObjCurrent"))

def ContainAtLeastOne(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args] 
    for Item in Items:
        if Item in List:
            return True
    return False

def ContainAll(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args]   
    for Item in Items:
        if Item not in List:
            return False
    return True

import timeout_decorator

def CallFunctionWithTimeLimit(TimeLimit, Function, *Args, **ArgsKw):
    # TimeLimit: in seconds.
    event = threading.Event()

    FunctionThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, Function, *Args), kwargs=ArgsKw)
    FunctionThread.setDaemon(True)
    FunctionThread.start()

    TimerThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, ReturnInGivenTime, TimeLimit))
    TimerThread.setDaemon(True)
    # So that this thread will be forced to terminate with the thread calling this function.
    # Which does not satisfy requirement. We need this thread to terminate when this function returns.
    TimerThread.start()
    event.wait()
    return 

def NotifyWhenFunctionReturn(event, Function, *Args, **ArgsKw):
    Function(*Args, **ArgsKw)
    event.set()

def ReturnInGivenTime(TimeLimit, Verbose=True):
    # TimeLimit: float or int. In Seconds.
    if Verbose:
        DLUtils.Log("Start counding down. TimeLimit=%d."%TimeLimit)
    time.sleep(TimeLimit)
    if Verbose:
        DLUtils.Log("TimeLimit reached. TimeLimit=%d."%TimeLimit)
    return

def GetGPUWithLargestUseableMemory(TimeLimit=10, Default='cuda:0'):
    # GPU = [Default]
    # CallFunctionWithTimeLimit(TimeLimit, __GetGPUWithLargestUseableMemory, GPU)
    # return GPU[0]
    return _GetGPUWithLargestUseableMemory()

def __GetGPUWithLargestUseableMemory(List):
    GPU= _GetGPUWithLargestUseableMemory()
    List[0] = GPU
    DLUtils.Log("Selected GPU: %s"%List[0])

def _GetGPUWithLargestUseableMemory(Verbose=True): # return torch.device with largest available gpu memory.
    try:
        import pynvml
        pynvml.nvmlInit()
        GPUNum = pynvml.nvmlDeviceGetCount()
        GPUUseableMemory = []
        for GPUIndex in range(GPUNum):
            Handle = pynvml.nvmlDeviceGetHandleByIndex(GPUIndex) # sometimes stuck here.
            MemoryInfo = pynvml.nvmlDeviceGetMemoryInfo(Handle)
            GPUUseableMemory.append(MemoryInfo.free)
        GPUUseableMemory = np.array(GPUUseableMemory, dtype=np.int64)
        GPUWithLargestUseableMemoryIndex = np.argmax(GPUUseableMemory)    
        if Verbose:
            DLUtils.Log("Useable GPU Num: %d"%GPUNum)
            report = "Useable GPU Memory: "
            for GPUIndex in range(GPUNum):
                report += "GPU%d: %.2fGB "%(GPUIndex, GPUUseableMemory[GPUIndex] * 1.0 / 1024 ** 3)
            DLUtils.Log(report)
        return 'cuda:%d'%(GPUWithLargestUseableMemoryIndex)
    except Exception:
        return "cuda:0"

def split_batch(data, batch_size): #data:(batch_size, image_size)
    sample_num = data.size(0)
    batch_sizes = [batch_size for _ in range(sample_num // batch_size)]
    if not sample_num % batch_size==0:
        batch_sizes.apend(sample_num % batch_size)
    return torch.split(data, section=batch_sizes, dim=0)

def cat_batch(dataloader): #data:(batch_num, batch_size, image_size)
    if not isinstance(dataloader, list):
        dataloader = list(dataloader)
    return torch.cat(dataloader, dim=0)



def import_file(file_from_sys_path):
    if not os.path.isfile(file_from_sys_path):
        raise Exception("%s is not a file."%file_from_sys_path)
    if file_from_sys_path.startswith("/"):
        raise Exception("import_file: file_from_sys_path must not be absolute path.")
    if file_from_sys_path.startswith("./"):
        module_path = file_from_sys_path.lstrip("./")
    module_path = module_path.replace("/", ".")
    return importlib.ImportModule(module_path)

def CopyDict(Dict):
    return dict(Dict)

def GetItemsfrom_dict(dict_, keys):
    items = []
    for name in keys:
        items.append(dict_[name])
    if len(items) == 1:
        return items[0]
    else:
        return tuple(items)   

def write_dict_info(dict_, save_path='./', save_name='dict info.txt'): # write readable dict info into file.
    values_remained = []
    with open(save_path+save_name, 'w') as f:
        for key in dict_.keys():
            value = dict_[value]
            if isinstance(value, str) or isinstance(value, int):
                f.write('%s: %s'%(str(key), str(value)))
            else:
                values_remained.append([key, value])

def GetNonLinearMethodModule(Name):
    if Name in ['relu']:
        return nn.ReLU()
    elif Name in ['tanh']:
        return nn.Tanh()
    elif Name in ['softplus']:
        return nn.Softplus()
    elif Name in ['sigmoid']:
        return nn.Sigmoid()
    else:
        raise Exception(Name)

def trunc_prefix(string, prefix):
    if(string[0:len(prefix)]==prefix):
        return string[len(prefix):len(string)]
    else:
        return string

def update_key(dict_0, dict_1, prefix='', strip=False, strip_only=True, exempt=[]):
    if not strip:
        for key in dict_1.keys():
            dict_0[prefix+key]=dict_1[key]
    else:
        for key in dict_1.keys():
            trunc_key=trunc_prefix(key, prefix)
            if strip_only:
                if(trunc_key!=key or key in exempt):
                    dict_0[trunc_key]=dict_1[key]
            else:
                dict_0[trunc_key]=dict_1[key]

def set_instance_attr(self, dict_, keys=None, exception=[]):
    if keys is None: # set all keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    setattr(self, key, value)

set_instance_variable = set_instance_attr

def set_dict_variable(dict_1, dict_0, keys=None, exception=['self']): # dict_1: target. dict_0: source.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
        
def set_instance_variable_and_dict(self, dict_1, dict_0, keys=None, exception=['self']): # dict_0: source. dict_1: target dict. self: target class object.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
                    setattr(self, key, value)
                
def set_default_attr(self, key, value):
    if self.__dict__.get(key) is None:
        setattr(self, key, value)

set_dict_and_instance_variable = set_class_variable_and_dict = set_instance_variable_and_dict


def load_param(dict_, exception=[], default_exception=['kw', 'param', 'key', 'item'], use_default_exception=True):
    param = Param()
    for key, item in dict_.items():
        if key not in exception:
            if use_default_exception:
                if key not in default_exception:
                    setattr(param, key, item)
            else:
                setattr(param, key, item)
    return param

def print_dict(dict_):
    Str = ""
    
    for key, items in dict_.items():
        Str('%s=%s'%(str(key), str(items)), end=' ')
    print('\n')

def GetLastestModel(model_prefix, base_dir='./', is_dir=True):
    # search for directory or file of most recently saved models(model with biggest epoch index)
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d*)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if(max_epoch is None):
                    max_epoch = epoch_num
                else:
                    if(max_epoch < epoch_num):
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + str(max_epoch) + '/'
    else:
        return "error"

def standardize_suffix(suffix):
    pattern = re.compile(r'\.?(\w+)')
    result = pattern.match(suffix)
    if result is None:
        raise Exception('check_suffix: %s is illegal suffix.'%suffix)
    else:
        suffix = result.group(1)
    return suffix

def EnsureSuffix(name, suffix):
    if not suffix.startswith("."):
        suffix = "." + suffix
    if name.endswith(suffix):
        return suffix
    else:
        return name + suffix

def check_suffix(name, suffix=None, is_path=True):
    # check whether given file name has suffix. If true, check whether it's legal. If false, add given suffix to it.
    if suffix is not None:
        if isinstance(suffix, str):
            suffix = standardize_suffix(suffix)
        elif isinstance(suffix, list):
            for i, suf_ in enumerate(suffix):
                suffix[i] = standardize_suffix(suf_)
            if len(suffix)==0:
                suffix = None
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))      

    pattern = re.compile(r'(.*)\.(\w+)')
    result = pattern.match(name)
    if result is not None: # match succeeded
        name = result.group(1)
        suf = result.group(2)
        if suffix is None:
            return name + '.' + suf
        elif isinstance(suffix, str):
            if name==suffix:
                return name
            else:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix))
                return name + '.' + suffix
        elif isinstance(suffix, list):
            sig = False
            for suf_ in suffix:
                if suf==suf_:
                    sig = True
                    return name
            if not sig:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix[0]))
                return name + '.' + suffix[0]                
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
    else: # fail to match
        if suffix is None:
            raise Exception('check_suffix: %s does not have suffix.'%name)
        else:
            if isinstance(suffix, str):
                suf_ = suffix
            elif isinstance(suffix, str):
                suf_ = suffix[0]
            else:
                raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
            warnings.warn('check_suffix: no suffix found in %s. adding suffix %s.'%(name, suffix))            
            return name + '.' + suf_

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

def scan_files(path, pattern, ignore_folder=True, raise_not_found_error=False):
    if not path.endswith('/'):
        path.append('/')
    files_path = os.listdir(path)
    matched_files = []
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for file_name in files_path:
        #print(file_name)
        if pattern.match(file_name) is not None:
            if os.path.isdir(path + file_name):
                if ignore_folder:
                    matched_files.append(file_name)
                else:
                    warnings.warn('%s is a folder, and will be ignored.'%(path + file))
            else:
                matched_files.append(file_name)
    
    if raise_not_found_error:
        if len(matched_files)==0:
            raise Exception('scan_files: cannot find any files that match pattern %s'%pattern)

    return matched_files

def copy_files(file_list, SourceDir='./', TargetDir=None, sys_type='linux'):
    if not SourceDir.endswith('/'):
        SourceDir += '/'

    if not TargetDir.endswith('/'):
        TargetDir += '/'

    EnsurePath(TargetDir)

    '''
    if subpath is not None:
        if not subpath.endswith('/'):
             subpath += '/'
        path += subpath
    EnsurePath(path)
    '''
    #print(TargetDir)
    if sys_type in ['linux']:
        for file in file_list:
            file = file.lstrip('./')
            file = file.lstrip('/')
            #print(path)
            #print(file)
            #shutil.copy2(file, dest + file)
            #print(SourceDir + file)
            #print(TargetDir + file)
            EnsurePath(os.path.dirname(TargetDir + file))
            if os.path.exists(TargetDir + file):
                os.system('rm -r %s'%(TargetDir + file))
            #print('cp -r %s %s'%(file_path + file, path + file))
            os.system('cp -r %s %s'%(SourceDir + file, TargetDir + file))
    elif sys_type in ['windows']:
        # to be implemented 
        pass
    else:
        raise Exception('copy_files: Invalid sys_type: '%str(sys_type))


def TargetDir_module(path):
    path = path.lstrip('./')
    path = path.lstrip('/')
    if not path.endswith('/'):
        path += '/'
    path =  path.replace('/','.')
    return path


def GetAllMethodsOfModule(ModulePath):
    from inspect import getmembers, isfunction
    Module = ImportModule(ModulePath)
    return getmembers(Module, isfunction)

ListAllMethodsOfModule = GetAllMethodsOfModule

# GlobalParam = DLUtils.json.JsonObj2PyObj({
#     "Logger": None
# })

def RandomSelect(List, SelectNum):
    if isinstance(List, int):
        Num = List
        List = range(Num)
    else:
        Num = DLUtils.GetLength(List)

    if Num > SelectNum:
        return random.sample(List, SelectNum)
    else:
        return List

def RandomIntInRange(Left, Right, IncludeRight=False):
    if not IncludeRight:
        Right -= 1
    #assert Left <= Right 
    return random.randint(Left, Right)

def MultipleRandomIntInRange(Left, Right, Num, IncludeRight=False):
    if not IncludeRight:
        Right += 1
    return RandomSelect(range(Left, Right), Num)

def RandomOrder(List):
    if isinstance(List, range):
        List = list(List)
    random.shuffle(List) # InPlace operation
    return List
def GetLength(Obj):
    if DLUtils.IsIterable(Obj):
        return len(Obj)
    else:
        raise Exception()

import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

def CalculateGitProjectTotalLines(Verbose=False):
    # runcmd(
    #     "git log  --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2 } END { printf \"added lines: %s, removed lines: %s, total lines: %s\n\", add, subs, loc }'"
    # )
    # GitCommand = 'git log  --pretty=tformat: --numstat | awk "{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\n", add, subs, loc }"'
    # report = os.system(GitCommand)
    # if Verbose:
    #     DLUtils.Log(report)
    # return report
    import os
    GitCommand = 'git log  --pretty=tformat: --numstat | awk \'{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\\n", add, subs, loc }\''
    report = os.system(GitCommand)

def GetDimensionNum(data):
    if isinstance(data, torch.Tensor):
        return len(list(data.size()))
    elif isinstance(data, np.ndarray):
        return len(data.shape)
    else:
        raise Exception(type(data))

def ToLowerStr(Str):
    return Str.lower()

from DLUtils.file import Str2File

def GetSavePathFromName(Name, Suffix=""):
    if not Suffix.startswith("."):
        Suffix = "." + Suffix
    FilePath = DLUtils.GetMainSaveDir() + Name + Suffix
    FilePath = DLUtils.file.RenameIfFileExists(FilePath)
    return FilePath


def Float2StrDisplay(Float):
    if np.isinf(Float):
        return "inf"
    if np.isneginf(Float):
        return "-inf"
    if np.isnan(Float):
        return "NaN"

    if Float==0.0:
        return "0.0"

    Positive = Float < 0.0
    if not Positive:
        Float = - Float
        Sign = - 1.0
    else:
        Sign = 1.0

    Base, Exp = DLUtils.math.Float2BaseAndExponent(Float)
    TicksStr = []
    if 1 <= Exp <= 2:
        FloatStr = str(int(Float))
    elif Exp == 0:
        FloatStr = '%.1f'%Float
    elif Exp == -1:
        FloatStr = '%.2f'%Float
    elif Exp == -2:
        FloatStr = '%.3f'%Float
    else:
        FloatStr = '%.2e'%Float
    return FloatStr * Sign

def Floats2StrDisplay(Floats):
    Floats = ToNpArray(Floats)
    Base, Exp = DLUtils.math.FloatsBaseAndExponent(Floats)

def Floats2StrWithEqualLength(Floats):
    Floats = DLUtils.ToNpArray(Floats)
    Base, Exp = DLUtils.math.Floats2BaseAndExponent(Floats)
    # to be implemented

def MountObj(MountPath, Obj, **kw):
    ObjRoot = kw.get("ObjRoot")
    ObjCurrent = kw.get("ObjCurrent")

    MountPath = MountPath.replace("/&", "&")
    MountPath = MountPath.replace("&^", "ObjRoot.")
    MountPath = MountPath.replace("&", "ObjCurrent.")
    MountPath = MountPath.split(".")
    SetAttrs(eval(MountPath[0]), MountPath[1:], value=Obj)

def MountDictOnObj(Obj, Dict):
    Obj.__dict__.update(Dict)

ExternalMethods = None
ExternalClasses = None
def InitExternalMethods():
    global ExternalMethods, ExternalClasses
    ExternalMethods = DLUtils.utils.EmptyPyObj()
    ExternalClasses = DLUtils.utils.EmptyPyObj()

def RegisterExternalMethods(Name, Method):
    setattr(ExternalMethods, Name, Method)

def RegisterExternalClasses(Name, Class):
    setattr(ExternalClasses, Name, Class)

def Bytes2Str(Bytes, Format="utf-8"):
    return str(Bytes, encoding = "utf-8")

def Str2Bytes(Str, Format="utf-8"):
    return Str.decode(Format)

def Unzip(Lists):
    return zip(*Lists)

def Zip(*Lists):
    return zip(*Lists)

def EnsurePyObj(Obj):
    if isinstance(Obj, argparse.Namespace):
        return Namespace2PyObj(Obj)
    elif isinstance(Obj, dict) or isinstance(Obj, list):
        return DLUtils.PyObj(Obj)
    else:
        raise Exception(type(Obj))

from collections import defaultdict
def CreateDefaultDict(GetDefaultMethod):
    return defaultdict(GetDefaultMethod)
GetDefaultDict = CreateDefaultDict

from DLUtils.format import *

# SystemType = DLUtils.system.GetSystemType()
# def GetSystemType():
#     return SystemType

def RandomImage(Height=512, Width=512, ChannelNum=3, 
        BatchNum=10, DataType="TorchTensor"):
    return

def IterableKeyToElement(Dict):
    for Key, Value in dict(Dict).items():
        if isinstance(Key, tuple) or isinstance(Key, set):
            for _Key in Key:
                Dict[_Key] = Value
            Dict.pop(Key)
    return Dict