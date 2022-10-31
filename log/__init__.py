import DLUtils

import numpy as np
import matplotlib as mpl
import pandas as pd

from matplotlib import pyplot as plt
from pydblite import Base
from collections import defaultdict

import time
import logging
import json5
from inspect import getframeinfo, stack

import DLUtils
from DLUtils.attr import *

import DLUtils.log.AbstractLog as AbstractLog
from DLUtils.log.AbstractLog import \
    AbstractLogAlongEpochBatchTrain, AbstractLogAlongBatch, AbstractModuleAlongEpochBatchTrain

def SetMethodForLogClass(Class, **kw):
    SaveDataOnly = kw.setdefault("SaveDataOnly", False)
    MountLocation = kw.setdefault("MountLocation", "data")
    if SaveDataOnly:
        if not hasattr(Class, "ToFile"):
            Class.ToFile = AbstractLogAlongEpochBatchTrain.ToFileDataOnly
        if not hasattr(Class, "FromFile"):
            Class.FromFile = AbstractLogAlongEpochBatchTrain.FromFileDataOnly
    else:
        if not hasattr(Class, "ToFile"):
            Class.ToFile = AbstractLogAlongEpochBatchTrain.ToFile
        if not hasattr(Class, "FromFile"):
            Class.FromFile = AbstractLogAlongEpochBatchTrain.FromFile
    # if MountLocation in ["Data", "data"]:
    #     Class.SetEpochBatchIndex = SetEpochBatchIndexForModuleData
    # elif MountLocation in ["Cache", "cache"]:
    #     Class.SetEpochBatchIndex = SetEpochBatchIndexForModuleCache
    SetEpochBatchMethodForModule(Class, **kw)

class DataLog:
    def __init__(self, IsRoot=False):
        if IsRoot:
            self.tables = {}
        param = self.param = DLUtils.EmptyPyObj()
        cache = self.cache = DLUtils.EmptyPyObj()
        cache.LocalColumn = {}
        param.LocalColumnNames = cache.LocalColumn.keys()
        self.HasParent = False
        return
    def SetParent(self, log, prefix=""):
        self.parent = log
        self.parentPrefix = prefix
        self.HasParent = True
        self.IsRoot = False
        return self
    def GetParent(self):
        return self.parent
    def SetParentPrefix(self, prefix):
        if not self.HasParent:
            raise Exception()
        self.parentPrefix = prefix
    def SetLocal(self, Name, Value):
        param = self.param
        cache = self.cache
        cache.LocalColumn[Name] = Value
        param.LocalColumnNames = cache.LocalColumn.keys()
        return self
    def CreateTable(self, TableName, ColumnNames, SavePath):
        param = self.param
        if self.HasParent:
            table = self.parent.CreateTable(self.parentPrefix + TableName, [*ColumnNames, *param.LocalColumnNames], SavePath)
        else:
            if hasattr(self.tables, TableName):
                DLUtils.AddWarning("Table with name: %s already exists."%TableName)
            DLUtils.EnsureFileDir(SavePath)
            DLUtils.EnsureFileDir(SavePath)
            table = Base(SavePath)
            table.create(*ColumnNames)
            self.tables[TableName] = table
        return table     
    def GetTable(self, TableName):
        table = self.tables.get(TableName)
        # if table is None:
        #     raise Exception("No such table: %s"%TableName)
        return table
    def HasTable(self, TableName):
        return self.tables.get(TableName) is None
    def CreateIndex(self, TableName, IndexColumn):
        table = self.GetTable(TableName)
        table.create_index(IndexColumn)
    def AddRecord(self, TableName, ColumnValues, AddLocalColumn=True):
        param = self.param
        cache = self.cache
        if self.HasParent:
            if AddLocalColumn:
                ColumnValues.update(cache.LocalColumn)
            self.parent.AddRecord(self.parentPrefix + TableName, ColumnValues)
        else:
            table = self.GetTable(TableName)
            if table is None:
                #raise Exception(TableName)
                table = self.CreateTable(TableName, [*ColumnValues.keys(), *param.LocalColumnNames], 
                    SavePath=DLUtils.GetMainSaveDir() + "data/" + "%s.pdl"%TableName)
            if AddLocalColumn:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            else:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            table.commit()

def LogList2EpochsFloat(Log, **kw):
    # for _Log in Log:
    #     EpochIndices.append(_Log[0])
    #     BatchIndices.append(_Log[1])
    if "EpochFloat" in Log and len(Log["EpochFloat"])==len(Log["Epoch"]):
        pass
    else:
        Log["EpochFloat"] = DLUtils.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)
    return Log["EpochFloat"]

def LogDict2EpochsFloat(Log, **kw):
    return DLUtils.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)

def PlotLogList(Name, Log, SaveDir=None, **kw):
    EpochsFloat = LogList2EpochsFloat(Log, BatchNum=kw["BatchNum"])
    Ys = Log["Value"]
    fig, ax = plt.subplots()
    DLUtils.plot.PlotLineChart(ax, EpochsFloat, Ys, Title="%s-Epoch"%Name, XLabel="Epoch", YLabel=Name)
    DLUtils.plot.SaveFigForPlt(SavePath=SaveDir + "%s-Epoch.png"%Name)
    DLUtils.file.Table2TextFile(
        {
            "Epoch": EpochsFloat,
            Name: Ys,
        },
        SavePath=SaveDir + "%s-Epoch.txt"%Name
    )

class LogAlongEpochBatchTrain(AbstractLogAlongEpochBatchTrain):
    # def __init__(self, param=None, **kw):
    #     super().__init__(**kw)
    #     DLUtils.transform.InitForNonModel(self, param, ClassPath="DLUtils.train.LogAlongEpochBatchTrain", **kw)
    #     self.Build(IsLoad=False)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        #param = self.param
        data = self.data
        cache = self.cache
        data.log = defaultdict(lambda:[])
        #self.IsPlotable = defaultdict(lambda:True)
        data.logType = defaultdict(lambda:"Unknown")
        self.GetLog = self.GetLogByName
        self.AddLog = self.AddLogList
        self.Get = self.GetLog
        return self
    # def UpdateEpoch(self, EpochIndex):
    #     cache = self.cache
    #     cache.EpochIndex = EpochIndex
    # def UpdateBatch(self, BatchIndex):
    #     cache = self.cache
    #     cache.BatchIndex = BatchIndex
    def AddLogList(self, Name, Value, Type=None):
        data = self.data
        cache =self.cache
        if not Name in data.log:
            data.log[Name] = {
                "Epoch":[],
                "Batch":[],
                "Value":[]
            }
            if Type is not None:
                data.logType[Name] = Type
        log = data.log[Name]
        log["Epoch"].append(self.GetEpochIndex())
        log["Batch"].append(self.GetBatchIndex()),
        log["Value"].append(Value)
    def AddLogsOfType(self, Logs, Type):
        cache = self.cache
        data = self.data
        for Name, Log in Logs.items():
            if "Value" in Log:
                self.AddLogList(
                    Name, Log["Value"], Type
                )
            else:
                _Log = DLUtils.DeleteKeysIfExist(dict(Log), ["Epoch", "Batch"])
                self.AddLogDict(
                    Name, _Log, Type
                )
    def AddLogDict(self, Name, Dict, Type=None):
        data = self.data
        cache = self.cache
        if not Name in data.log:
            data.log[Name] = defaultdict(lambda:[])
            if Type is not None:
                data.logType[Name] = Type
        Log = data.log[Name]
        for key, value in Dict.items():
            # if key in ["Epoch", "Batch"]:
            #     continue
            Log[key].append(value)
        Log["Epoch"].append(self.GetEpochIndex())
        Log["Batch"].append(self.GetBatchIndex())
    def LogCache(self, Name, Data, Type="Cache"):
        cache = self.cache
        data = self.data
        data.logType[Name] = Type
        data.log[Name] = {
            "Epoch": self.GetEpochIndex(),
            "Batch": self.GetBatchIndex(),
            "Value": Data
        }
    def RegisterLog(self, Name, Type="List"):
        data = self.data
        if Type in ["List"]:
            data.log[Name] = []
        elif Type in ["Dict"]:
            data.log[Name] = {}
        else:
            raise Exception(Type)
    def SetPlotType(self, Name, Type):
        self.PlotType[Name] = Type

    def SetLocal(self, Name, Value):
        setattr(self, Name, Value)
    def SetLogType(self, Name, Value):
        data = self.data
        if not Name in data.log:
            raise Exception()
        data.logType[Name] = Value
    def PlotLogOfGivenType(self, Type, PlotType="LineChart", SaveDir=None):
        DLUtils.EnsureDir(SaveDir)
        data = self.data
        for Name, Log in data.log.items():
            if not data.logType[Name] in [Type]:
                continue
            if PlotType in ["LineChart"]:
                self.PlotLogList(Name, Log, SaveDir)
            elif PlotType in ["Statistics"]:
                self.PlotLogDictStatistics(Name, Log, SaveDir)
            else:
                raise Exception(PlotType)
    def Log2EpochsFloat(self, Log, **kw):
        kw["BatchNum"] = self.BatchNum
        return DLUtils.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)
    def PlotLogDict(self, Name, Log, SaveDir=None):
        DLUtils.EnsureDir(SaveDir)
        LogNum = len(Log.keys()[0])
        PlotNum = len(Log.keys() - 2) # Exclude Epoch, Batch
        fig, axes = DLUtils.plot.CreateFigurePlt(PlotNum)
        Xs = self.GetEpochsFloatFromLogDict(Log)
        for index, Key in enumerate(Log.keys()):
            Ys = Log[Key]
            ax = DLUtils.plot.GetAx(axes, Index=index)
            DLUtils.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Key, XLabel="Epoch", YLabel=Key)
        plt.tight_layout()
        DLUtils.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png"%Name)
        DLUtils.file.Table2TextFileDict(Log, SavePath=SaveDir + "%s-Epoch"%Name)
    def ListLog(self):
        return list(self.data.log.keys())
    def ReportLog(self):
        Report = []
        log = self.data.log
        for Name, Log in self.data.log.items():
            if isinstance(Log, list):
                Report.append(
                    "%s: listlog length=%d type=%s"%(Name, len(Log), type(Log[0]["Value"]))
                )
            elif isinstance(Log, dict):
                ExampleKey = Log.keys()[0]
                Report.append(
                    "%s: dictlog length=%d example=%s"%(Name, len(Log[ExampleKey]), type(Log[ExampleKey]["Value"]))
                )
            else:
                ExampleKey = Log.keys()[0]
                Report.append(
                    "%s: cachelog length=%d type=%s"%(Name, type(Log["Value"]))
                )
    def GetCache(self, Name):
        return self.data.log[Name]["Value"]
    def GetLogValueByName(self, Name):
        Log = self.GetLogByName(Name)
        if isinstance(Log, dict) and "Value" in Log:
            return Log["Value"]
        else:
            return Log
    def GetLogByName(self, Name):
        data = self.data
        if not Name in data.log:
            #raise Exception(Name)
            DLUtils.AddWarning("No such log: %s"%Name)
            return None
        return data.log[Name]
    def GetCacheByName(self, Name):
        data = self.data
        if not Name in data.log:
            #raise Exception(Name)
            DLUtils.AddWarning("No such log: %s"%Name)
            return None
        return data.log[Name]["Value"]
    def GetLogValueOfType(self, Type):
        data = self.data
        Logs = {}
        for Name, Log in data.log.items():
            if data.logType[Name] == Type:
                Logs[Name] = Log["Value"]
        return Logs
    def GetLogOfType(self, Type):
        data = self.data
        Logs = {}
        for Name, Log in data.log.items():
            if data.logType[Name] == Type:
                Logs[Name] = Log
        return Logs
    def PlotAllLogs(self, SaveDir=None):
        DLUtils.EnsureDir(SaveDir)
        data = self.data
        for Name, Log in data.log.items():
            if isinstance(Log, dict):
                self.PlotLogDict(self, Name, Log, SaveDir)
            elif isinstance(Log, list):
                self.PlotLogList(self, Name, Log, SaveDir)
            else:
                continue
#DLUtils.transform.SetEpochBatchMethodForModule(LogForEpochBatchTrain)
LogAlongEpochBatchTrain.AddLogCache = LogAlongEpochBatchTrain.LogCache

class Log:
    def __init__(self, Name, **kw):
        self.log = _CreateLog(Name, **kw)
    def AddLog(self, log, TimeStamp=True, FilePath=True, RelativeFilePath=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s]%s"%(DLUtils.system.GetTime(), log)
        if FilePath:
            if RelativeFilePath:
                log = "%s File \"%s\""%(log, DLUtils.file.GetRelativePath(Caller.filename, "."))
            else:
                log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.log.debug(log)

    def AddWarning(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s][WARNING]%s"%(DLUtils.system.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.log.debug(log)

    def AddError(self, log, TimeStamp=True, **kw):
        if TimeStamp:
            self.log.error("[%s][ERROR]%s"%(DLUtils.system.GetTime(), log))
        else:
            self.log.error("%s"%log)
    def Save(self):
        return

def ParseLog(log, **kw):
    if log is None:
        log = GetLogGlobal()
    elif isinstance(log, str):
        log = GetLog(log, **kw)
    else:
        return log
    return log

def AddLog(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddLog(Str, *args, StackIndex=2, **kw)

def AddWarning(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddWarning(Str, *args, StackIndex=2, **kw)

def AddError(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddError(Str, *args, StackIndex=2, **kw)

def AddLog2GlobalParam(Name, **kw):
    import DLUtils
    setattr(DLUtils.GlobalParam.log, Name, CreateLog(Name, **kw))

def CreateLog(Name, **kw):
    return Log(Name, **kw)

def _CreateLog(Name, SaveDir=None, **kw):
    if SaveDir is None:
        SaveDir = DLUtils.GetMainSaveDir()
    DLUtils.EnsureDir(SaveDir)
    HandlerList = ["File", "Console"]
    if kw.get("FileOnly"):
        HandlerList = ["File"]

    # 输出到file
    log = logging.Logger(Name)
    log.setLevel(logging.DEBUG)
    log.HandlerList = HandlerList

    for HandlerType in HandlerList:
        if HandlerType in ["Console"]:
            # 输出到console
            ConsoleHandler = logging.StreamHandler()
            ConsoleHandler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
            log.addHandler(ConsoleHandler)
        elif HandlerType in ["File"]:
            FileHandler = logging.FileHandler(SaveDir + "%s.txt"%(Name), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
            FileHandler.setLevel(logging.DEBUG)     
            log.addHandler(FileHandler)
        else:
            raise Exception(HandlerType)

    HandlerNum = len(HandlerList)
    if len(HandlerList)==0:
        raise Exception(HandlerNum)

    return log

def SetLogGlobal(GlobalParam):
    GlobalParam.log.Global = CreateLog('Global')

def SetLog(Name, log):
    setattr(DLUtils.GlobalParam.log, Name, log)

def GetLogGlobal():
    return DLUtils.GlobalParam.log.Global

def SetGlobalParam(GlobalParam):
    DLUtils.GlobalParam = GlobalParam

def GetGlobalParam():
    return DLUtils.GlobalParam

def GetDatasetDir(Type):
    GlobalParam = DLUtils.GetGlobalParam()
    Attrs = "config.Dataset.%s.Dir"%Type
    if not HasAttrs(GlobalParam, Attrs):
        raise Exception()
    else:
        return GetAttrs(GlobalParam, Attrs)
    
def CreateMainSaveDir(SaveDir=None, Name=None, GlobalParam=None, Method="FromIndex"):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    if SaveDir is None:
        if Method in ["FromTime", "FromTimeStamp"]:
            SaveDir = "./log/%s-%s/"%(Name, DLUtils.system.GetTime("%Y-%m-%d-%H:%M:%S"))
        elif Method in ["FromIndex"]:
            SaveDir = DLUtils.RenameDirIfExists("./log/%s/"%Name)
        else:
            raise Exception(Method)
    DLUtils.EnsureDir(SaveDir)
    #print("[%s]Using Main Save Dir: %s"%(DLUtils.system.GetTime(),SaveDir))
    #DLUtils,AddLog("[%s]Using Main Save Dir: %s"%(DLUtils.system.GetTime(),SaveDir))
    SetAttrs(DLUtils.GetGlobalParam(), "SaveDir.Main", value=SaveDir)
    return SaveDir

def GetMainSaveDir(GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    return GlobalParam.SaveDir.Main

def SetMainSaveDir(SaveDir, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    if not SaveDir.endswith("/"):
        SaveDir += "/"
    assert not DLUtils.file.ExistsFile(SaveDir.rstrip("/"))
    SetAttrs(GlobalParam, "SaveDir.Main", value=SaveDir)
    return
    
def ChangeMainSaveDir(SaveDir, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    assert HasAttrs(GlobalParam, "SaveDir.Main")
    SaveDirOld = GlobalParam.SaveDir.Main
    DLUtils.file.RenameDir(SaveDirOld, SaveDir)
    SetMainSaveDir(SaveDir, GlobalParam)
    DLUtils.AddLog("DLUtils.GlobalParam.SaveDir.Main %s -> %s"%(SaveDirOld, SaveDir))

def SetSubSaveDir(SaveDir=None, Name="Experiment", GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    SetAttrs(GlobalParam, "SaveDir" + "." + Name, value=SaveDir)
    return SaveDir

def GetSubSaveDir(Type):
    if not hasattr(DLUtils.GetGlobalParam().SaveDir, Type):
        setattr(DLUtils.GetGlobalParam().SaveDir, Type, DLUtils.GetMainSaveDir() + Type + "/")
    return getattr(DLUtils.GetGlobalParam().SaveDir, Type)        


def SetSubSaveDirEpochBatch(Name, EpochIndex, BatchIndex, BatchInternalIndex=None, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    if BatchInternalIndex is None:
        DirName = "Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
    else:
        DirName = "Epoch%d-Batch%d-No%d"%(EpochIndex, BatchIndex, BatchInternalIndex)
    SaveDir = DLUtils.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"
    SetSubSaveDir(Name, SaveDir)
    SetAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName, SaveDir)
    return SaveDir
def GetSubSaveDirEpochBatch(Name, EpochIndex, BatchIndex, BatchInternalIndex=None, GlobalParam=None):
    if BatchInternalIndex is None:
        DirName = "Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
    else:
        DirName = "Epoch%d-Batch%d-No%d"%(EpochIndex, BatchIndex, BatchInternalIndex)
    # if HasAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName):
    #     return GetAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName)
    # else: # As a guess
    #     DLUtils.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"
    return DLUtils.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"

def GetAllSubSaveDirsEpochBatch(Name, SaveDir=None, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = DLUtils.GetGlobalParam()
    if SaveDir is None:
        SaveDir = DLUtils.GetMainSaveDir(GlobalParam)
    SaveDirs = DLUtils.file.ListAllDirs(SaveDir + Name + "/")
    SaveDirNum = len(SaveDirs)
    if SaveDirNum == 0:
        raise Exception(SaveDirNum)
    for Index, SaveDir in enumerate(SaveDirs):
        SaveDirs[Index] = DLUtils.GetMainSaveDir(GlobalParam) + Name + "/" + SaveDir # SaveDir already ends with "/"
    return SaveDirs



def GetDataLog():
    return DLUtils.GlobalParam.log.Data

def GetLog(Name, CreateIfNone=True, **kw):
    if not hasattr(DLUtils.GlobalParam.log, Name):
        if CreateIfNone:
            DLUtils.AddLog(Name)
        else:
            raise Exception()
    return getattr(DLUtils.GlobalParam.log, Name)

