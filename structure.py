import re
import DLUtils
from DLUtils.attr import *
import numpy as np

class FixedSizeQueuePassiveOut(): # first in, first out. out is passive. queue is fixed size.
    def __init__(self, Size, DataType=None):
        self.Size = Size
        self.Index = 0
        self.SizeNow = 0
        self.IsFull = False
        assert Size > 0
    def put(self, In):
        self.SizeNow += 1
    def _Sum(self):
        return self.Data.sum()

class FixedSizeQueuePassiveOutInt32(FixedSizeQueuePassiveOut):
    def __init__(self, Size, KeepSum=True):
        self.Data = np.zeros((Size), np.int32)
        if KeepSum:
            self.Sum = self._SumCached
        else:
            self.Sum = self._Sum
        self.SumCache = 0
        super().__init__(Size, DataType=np.int32)
        self.append = self.put
    def _SumCached(self):
        return self.SumCache
    def put(self, In):
        Index = self.Index
        Data = self.Data
        Out = self.Data[Index]
        Data[Index] = In
        
        self.Index += 1
        if self.Index == self.Size:
            self.Index = 0
        self.SumCache += In

        if self.IsFull:
            self.SumCache -= Out
            return Out
        else:
            self.SizeNow += 1
            if self.SizeNow == self.Size:
                self.IsFull = True
            return None

class IntRange(DLUtils.module.AbstractModule):
    # represesnt a single int range.
    # to be implemented. represents multiple int ranges.
    # to be implemented. automatically detect overlap and converge ranges.
    def __init__(self, Logger=None):
        super().__init__(Logger)
        Param = self.Param
        Param._CLASS = "DLUtils.structure.IntRange"
        self.Start = None
        self.Next = None
        self.append = self._appendFirst
    def _appendFirst(self, Num):
        self.Start = Num
        self.Next = Num + 1
        self.append = self._appendNext
        return self
    def _appendNext(self, Num):
        if Num == self.Next:
            self.Next += 1
        return self
    def ExtractParam(self, *List, RetainSelf=True, **Dict):
        Param = self.Param
        Start = self.Start
        if self.Next is None:
            End = None
        else:
            End = self.Next - 1
        Param.Range.Start = Start
        Param.Range.End = End
        Param.Range.IncludeRight = True
        return super().ExtractParam(RetainSelf=RetainSelf)
    def LoadParam(self, Param):
        return super().LoadParam(Param)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Start = Param.Range.Start
        self.Next = Param.Range.End
        self.append = self._appendNext
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def Extract(self):
        if self.Start is None:
            return None
        else:
            return [self.Start, self.End]

def CheckRoutingsInputOutNum(Router):
    for Index, Routing in enumerate(Router.Routings):
        if isinstance(Routing.Module, DLUtils.PyObj):
            RoutingInNum = len(Routing.In)
            ModuleInNum = len(Routing.Module.In)
            if RoutingInNum != ModuleInNum:
                raise Exception("Routing inputs %d param to its Module, which accepts %d param."%(RoutingInNum, ModuleInNum))
            RoutingOutNum = len(Routing.Out)
            ModuleOutNum = len(Routing.Module.Out)
            if RoutingOutNum != ModuleOutNum:
                raise Exception("Routing.Module ouputs %d param, whilc Routing accepts %d param."^(ModuleOutNum, RoutingOutNum))

def ParseRouterDynamic(Router, ObjRefList=[], InPlace=False, **kw):
    assert isinstance(Router, DLUtils.PyObj), "Object %s is not a Router."%Router
    RouterParsed = DLUtils.parse.ParsePyObjDynamic(
        Router, ObjRefList=ObjRefList, InPlace=InPlace, RaiseFailedParse=True, **kw
    )
    for routing in RouterParsed.Routings:
        #if hasattr(routing, "OnlineParseAttrs"):
        if "RepeatTime" not in routing.OnlineParseAttrs:
            routing.cache.RepeatTime = routing.RepeatTime
        if "Condition" not in routing.OnlineParseAttrs:
            routing.cache.Condition = routing.Condition
        if "InheritStates" not in routing.OnlineParseAttrs:
            routing.cache.InheritStates = routing.InheritStates
        # else:
        #     routing.cache.RepeatTime = routing.RepeatTime
        #     routing.cache.Condition = routing.Condition
        #     routing.cache.InheritStates = routing.InheritStates
        routing.cache.InDict = DLUtils.ToDict(routing.InDict)
    return RouterParsed

def SetOnlineParseAttrsForRouter(routing):
    routing.OnlineParseAttrs = {}
    for Attr, Value in ListAttrsAndValues(routing):
        #if isinstance(Value, str) and "%" in Value: # Dynamic Parse
        if isinstance(Value, str) and Value.startswith("%"):
                routing.OnlineParseAttrs[Attr] = Value

    routing.InDictOnlineParseAttrs = {}
    for Key, Value in routing.InDict.items():
        #if isinstance(Value, str) and '%' in Value:
        if isinstance(Value, str) and Value.startswith("%"):
            #routing.InDictOnlineParseAttrs[Key] = Value[1:]
            routing.InDictOnlineParseAttrs[Key] = Value

    if len(routing.OnlineParseAttrs)>0 or len(routing.InDictOnlineParseAttrs)>0:
        routing.HasOnlineParseAttrs = True
    else:
        routing.HasOnlineParseAttrs = False
        #delattr(routing, "OnlineParseAttrs")
        delattr(routing, "InDictOnlineParseAttrs")

def ParseRoutingAttrsOnline(routing, States):
    #DLUtils.parse.RedirectPyObj(Routing, States)
    for attr, value in routing.OnlineParseAttrs.items():
        value = GetAttrs(routing, attr)    
        #value = re.sub("(%\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        value = eval(value.replace("%", "States."))
        setattr(routing.cache, attr, value)
    for attr, value in routing.InDictOnlineParseAttrs.items():
        routing.cache.InDict[attr] = eval(value.replace("%", "States."))
    return routing

class RouterStatic(DLUtils.PyObj):
    def FromPyObj(self, Obj):
        self.FromPyObj(DLUtils.router.ParseRouterStatic(Obj, InPlace=False))
    def ToRouterDynamic(self, **kw):
        return DLUtils.router.ParseRouterDynamic(self, **kw)

