import DLUtils
import numpy as np
from ..module import AbstractNetwork

class ModuleSeries(DLUtils.module.AbstractModuleGroup):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("OutName", "OutputName"): "Out.Name"
    })
    def LoadParam(self, Param):
        super().LoadParam(Param)
        Param = self.Param
        self.ModuleList = []
        for Name, SubModuleParam in Param.SubModules.items():
            self.ModuleList.append(self.SubModules[Name])
        Param.Module.Num = self.ModuleNum = len(self.ModuleList)
        return self
    def AddSubModule(self, Name=None, SubModule=None, **Dict):
        if Name is not None:
            assert len(Dict) == 0
            assert SubModule is not None
            return self.AppendSubModule(Name=Name, SubModule=SubModule)
        else:
            for _Name, _SubModule in Dict.items():
                self.AppendSubModule(Name=_Name, SubModule=_SubModule)
        return self
    def AppendSubModule(self, Name=None, SubModule=None):
        Param = self.Param
        ModuleList = self.ModuleList
        Index = len(ModuleList)
        self.ModuleList.append(SubModule)
        if Name is None:
            Name = f"L{Index}"
        super().AddSubModule(Name, SubModule)
        self.ModuleNum = Param.Module.Num = len(ModuleList)
        return self
    def _forward_out(self, In):
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            In = Out
        return Out
    def _forward_all_tuple(self, In):
        OutList = []
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            OutList.append(Out)
            In = Out
        return tuple(OutList)
    def _forward_all_dict(self, In):
        OutDict = {"In": In}
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            OutDict[self.OutNameList[LayerIndex]] = Out
            In = Out
        return OutDict

    def Init(self, IsSuper=False, IsRoot=True):
        if self.IsLoad():
            self.ModuleList = list(self.SubModules.values())
        
        Param = self.Param
        OutType = Param.Out.setdefault("Type", "Out")
        import functools

        ReceiveMethodMap = DLUtils.IterableKeyToElement({
            ("Out", "OutOnly"): self._forward_out,
            ("All", "AllInTuple"): self._forward_all_tuple,
            ("AllInDict"): self._forward_all_dict
        })

        #self.Receive = functools.partial(self.ReceiveMethodMap[OutType], self=self)
        self.Receive = ReceiveMethodMap[OutType]
        self.ModuleNum = Param.Module.Num = len(self.ModuleList)
        assert self.ModuleNum > 0
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.LayerNum = self.ModuleNum
        if OutType in ["AllInDict"]:
            if Param.Out.hasattr("Name"):
                self.OutNameList = Param.Out.Name
            else:
                self.OutNameList = ["L%d"%LayerIndex for LayerIndex in range(self.LayerNum)]

        return self

class ModuleGraph(DLUtils.module.AbstractModuleGroup):
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
        Param = self.Param
        Param.RouteList = DLUtils.Param([])
    def AddRoute(self, ModuleName=None, InList=None, OutName=None, **Dict):
        if ModuleName is not None:
            self._AddRoute(ModuleName, InList, OutName)
        for ModuleName, InOut in Dict.items():
            if isinstance(InOut, tuple) or isinstance(InOut, list):
                In = InOut[0]
                Out = InOut[1]
            elif isinstance(InOut, str):
                In = InOut
                Out = None
            else:
                raise Exception()
            self._AddRoute(ModuleName, In, Out)
        return self
    def _AddRoute(self, ModuleName, In=None, Out=None):
        Param = self.Param
        if ModuleName in ["AddDictItem"]:
            assert isinstance(In, str)
            Param.RouteList.append(
                DLUtils.Param({
                    "Type": "AddDictItem",
                    "Module": ModuleName,
                    "In": In
                })
            )
        else:
            RouteParam = DLUtils.Param({
                "In": In,
                "Module": ModuleName,
                "Out": Out
            })
            if isinstance(In, str):
                if isinstance(Out, str):
                    RouteParam.Type = "StrInStrOut"
                elif isinstance(Out, tuple):
                    RouteParam.Type = "StrInTupleOut"
                elif isinstance(Out, set):
                    RouteParam.Type = "StrInDictOutSameName"
                else:
                    raise Exception()
            else:
                if isinstance(In, list) or isinstance(In, tuple):
                    RouteParam.Type = "ListInStrOut"
                else:
                    raise Exception()
            Param.RouteList.append(RouteParam)
    def SetOut(self, *List, **Dict):
        Param = self.Param
        Param.Out.List = List
        Param.Out.Dict = Dict
        return self
    def SetOutDict(self, *List, **Dict):
        return self.SetDictOut(*List, **Dict)
    def SetDictOut(self, *List, **Dict):
        Param = self.Param
        for OutName in List:
            Dict[OutName] = OutName
        Param.Out.Dict = Dict
        Param.Out.List = []
        return self
    def SetIn(self, *List, **Dict):
        Param = self.Param
        Param.In.List = List
        Param.In.Dict = Dict
        return self
    def SetDictIn(self, *List, **Dict):
        Param = self.Param
        Param.In.Type = "DictIn"
        return self
    def Receive(self, *List, **Dict):
        Nodes = self.In(*List, **Dict)
        for RouteFunc, Route in self.RouteList:
            RouteFunc(Nodes, Route, Dict)
        return self.Out(Nodes)
    def _ReceiveDict(self, _Dict={}, **Dict):
        Nodes = _Dict
        Nodes.update(Dict)
        for RouteFunc, Route in self.RouteList:
            RouteFunc(Nodes, Route, Dict)
        return self.Out(Nodes)
    def _StrInStrOut(self, Nodes, Route, Dict):
        In = Nodes[Route.InName]
        Out = Route.Module(In)
        Nodes[Route.OutName] = Out
    def _StrInTupleOut(self, Nodes, Route, Dict):
        In = Nodes[Route.InName]
        OutList = Route.Module(In)
        for Index, OutName in enumerate(Route.OutNameList):
            Nodes[OutName] = OutList[Index]
    def _ListInStrOut(self, Nodes, Route, Dict):
        InList = [Nodes[InName] for InName in Route.InNameList]
        Out = Route.Module(*InList)
        Nodes[Route.OutName] = Out
    def _StrInDictOutSameName(self, Nodes, Route, Dict):
        In = Nodes[Route.InName]
        OutDict = Route.Module(In)
        for Index, OutName in enumerate(Route.OutDict):
            Nodes[OutName] = OutDict[OutName]
    def _StrInDictOut(self, Nodes, Route, Dict):
        In = Nodes[Route.InName]
        OutDict = Route.Module(In)
        for MapName, OutName in Route.OutDict.items():
            Nodes[MapName] = OutDict[OutName]
    def _StrInDictOut(self, Nodes, Route, Dict):
        In = Nodes[Route.InName]
        OutList = Route.Module(In)
        for Index, OutName in enumerate(Route.OutNameList):
            Nodes[OutName] = OutList[Index]
    def _AddDictItem(self, Nodes, Route, Dict):
        for Key, Value in Nodes[Route.InName].items():
            Nodes[Key] = Value
    def _InStr(self, *List, **Dict):
        assert len(List) == 1
        Nodes = {self.InName: List[0]}
        return Nodes
    def _InDefault(self, *List, **Dict):
        assert len(List) == 1
        Nodes = {"In": List[0]}
        return Nodes
    def _InList(self, *List, **Dict):
        Nodes = {}
        for Index, Name in enumerate(self.InNameList):
            Nodes[Name] = List[Index]
        return Nodes
    def _InDict(self, *List, **Dict):
        Nodes = dict(Dict)
        return Nodes
    def _InDictMap(self, *List, **Dict):
        Nodes = {}
        for MapName, InName in self.InNameDict.items():
            Nodes[MapName] = Dict[InName]
        return Nodes
    def _NoOut(self, Nodes):
        return
    def _SingleOut(self, Nodes):
        return Nodes[self.OutName]
    def _MultiOutTuple(self, Nodes):
        #return [Nodes[OutName] for OutName in self.OutList]
        OutDict = {}
        for Name in self.OutList:
            OutDict[Name] = Nodes[Name]
        return OutDict
    def _MultiOutDict(self, Nodes):
        #return [Nodes[OutName] for OutName in self.OutList]
        OutDict = {}
        for StatusName, OutName in self.OutNameMap.items():
            OutDict[StatusName] = Nodes[OutName]
        return OutDict
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.RouteList = []
        RouteFuncMap = DLUtils.IterableKeyToElement({
            ("AddDictItem"): self._AddDictItem,
            ("StrInStrOut"): self._StrInStrOut,
            ("StrInTupleOut"): self._StrInTupleOut,
            ("ListInStrOut"): self._ListInStrOut,
            ("StrInDictOutSameName"): self._StrInDictOutSameName,
            ("StrInDictOut"): self._StrInDictOut
        })
        for RouteParam in Param.RouteList:
            Route = DLUtils.param()
            Type = RouteParam.Type
            RouteFunc = RouteFuncMap[Type]
            if Type in ["AddDictItem"]:
                Route.InName = str(RouteParam.In)
                self.RouteList.append((RouteFunc, Route))
                continue
            Route.Module = self.GetSubModule(RouteParam.Module)
            if Type in ["ListInStrOut"]:
                Route.InNameList = list(RouteParam.In)
                Route.OutName = str(RouteParam.Out)
            elif Type in ["StrInTupleOut"]:
                Route.InName = str(RouteParam.In)
                Route.OutNameList = tuple(RouteParam.Out)
            elif Type in ["StrInStrOut"]:
                Route.InName = str(RouteParam.In)
                Route.OutName = str(RouteParam.Out)
            elif Type in ["StrInDictOutSameName"]:
                Route.InName = str(RouteParam.In)
                OutDict = {}
                for Name in RouteParam.Out:
                    OutDict[Name] = Name
                Route.OutDict = OutDict
            elif Type in ["StrInDictOut"]:
                Route.InName = str(RouteParam.In)
                Route.OutDict = dict(RouteParam.Out)

            else:
                raise Exception()
            self.RouteList.append((RouteFunc, Route))
        # set input method
        if not Param.hasattr("In"):
            InList = Param.In.List = []
            InDict = Param.In.Dict = {}
        else:
            InList = Param.In.List
            InDict = Param.In.Dict
        if Param.In.hasattr("Type"):
            if Param.In.Type in ["DictIn"]:
                self.In = self._InDict
            else:
                raise Exception()
        else:
            if len(InDict) > 0:
                self.In = self._InDictMap
                assert len(InList) == 0
                return
            else:
                if len(InList) > 1:
                    self.In = self._InList
                    self.InNameList = Param.In.List
                elif len(InList) == 1:
                    self.In = self._InStr
                    self.InName = InList[0]
                    assert len(InList) == 1
                else: # default
                    self.In = self._InDefault
        if len(InList) == 0:
            Param.In.delattr("List")
        if len(InDict) == 0:
            Param.In.delattr("Dict")
        self.InitSetOut()
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def InitSetOut(self):
        Param = self.Param
        # set output method
        OutList = Param.Out.List
        OutDict = Param.Out.Dict
        if len(OutDict) == 0:
            if len(OutList) == 0:
                self.Out = self._NoOut
            elif len(OutList) == 1:
                self.Out = self._SingleOut
                self.OutName = Param.Out.List[0]
            else:
                self.Out = self._MultiOutTuple
                self.OutList = list(Param.Out.List)
        else:
            self.OutNameMap = dict(Param.Out.Dict)
            assert len(OutList) == 0
            self.Out = self._MultiOutDict
        if len(OutList) == 0:
            Param.Out.delattr("List")
        if len(OutDict) == 0:
            Param.Out.delattr("Dict")

ModuleList = ModuleSeries