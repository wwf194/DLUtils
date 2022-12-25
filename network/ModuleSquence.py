import DLUtils
import numpy as np
from ..module.AbstractModule import AbstractNetwork

class ModuleSequence(AbstractNetwork):
    def __init__(self, ModuleList=None, Log=None):
        super().__init__(Log)
        if ModuleList is None:
            ModuleList = []
        assert isinstance(ModuleList, list)
        if ModuleList is not None:
            self.SetModuleList(ModuleList)
        Param = self.Param
        Param._CLASS = "DLUtils.network.ModuleSequence"
    def LoadParam(self, Param):
        super().LoadParam(Param)
        Param = self.Param
        if Param.hasattr("Module.Num"):
            self.ModuleNum = Param.Module.Num
        self.ModuleList = []
        for Name, SubModuleParam in Param.SubModules.items():
            self.ModuleList.append(self.SubModules[Name])
        self.ModuleNum = len(self.ModuleList)
        return self
    def SetModuleList(self, ModuleList):
        Param = self.Param
        if isinstance(ModuleList, list):
            for Index, SubModule in enumerate(ModuleList):
                self.AddSubModule(
                    f"L{Index}", SubModule
                )
                # Key = str(Index)
                # self.SubModules[Key] = SubModule
                # setattr(Param.SubModules, Key, SubModule.Param)
                # SubModule.Param._PATH = Param._PATH + "." + Key
        if isinstance(ModuleList, dict):
            for Name, SubModule in ModuleList.items():
                self.AddSubModule(
                    Name, SubModule
                )
        self.ModuleNum = len(ModuleList)
        self.ModuleList = ModuleList
        Param.Module.Num = len(ModuleList)
        return self
    def Receive(self, Input):
        for ModuleIndex in range(self.ModuleNum):
            Output = self.ModuleList[ModuleIndex](Input)
            Input = Output
        return Output

class ModuleGraph(AbstractNetwork):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.module.ModuleGraph"
        Param.RouteList = DLUtils.Param([])
    def AddRoute(self, SubModule, InputList, OutputName):
        Param = self.Param
        Param.RouteList.append(
            DLUtils.Param({
                "InputList": InputList,
                "SubModule": SubModule,
                "Output": OutputName
            })
        )
        return self
    def SetOutput(self, Output):
        Param = self.Param
        Param.Output = Output
        return self
    def Receive(self, **Dict):
        Nodes = Dict
        for Route in self.RouteList:
            InputList = [Nodes[InputName] for InputName in Route.InputList]
            Output = Route.SubModule(*InputList)
            Nodes[Route.OutputName] = Output
        return self.Output(Nodes)
    def _NoOutput(self, Nodes):
        return
    def _SingleOutput(self, Nodes):
        return Nodes[self.OutputName]
    def _MultiOutput(self, Nodes):
        return [Nodes[OutputName] for OutputName in self.OutputList]
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.RouteList = []
        for RouteParam in Param.RouteList:
            Route = DLUtils.param()
            Route.SubModule = self.GetSubModule(RouteParam.SubModule)
            Route.InputList = list(RouteParam.InputList)
            Route.OutputName = RouteParam.Output
            self.RouteList.append(Route)
        Output = Param.Output
        if isinstance(Output, str):
            self.Output = self._SingleOutput
            self.OutputName = Param.Output
        elif isinstance(Output, list):
            if len(Output) == 0:
                self.Output = self._NoOutput
            elif len(Output) == 1:
                self.Output = self._SingleOutput
                self.OutputName = Param.Output
            else:
                self.Output = self._MultiOutput
                self.Output = list(Param.Output)
        else:
            raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)