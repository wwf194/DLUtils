import DLUtils
import functools
import warnings

def BuildModule(param, **Dict):
    if hasattr(param, "ClassPath"):
        try:
            Class = DLUtils.parse.ParseClass(param.ClassPath)
            return Class(**Dict)
        except Exception:
            DLUtils.AddWarning("Cannot parse ClassPath: %s"%param.ClassPath)

    module = BuildExternalModule(param, **Dict)
    if module is not None:
        return module
    raise Exception()

ExternalModules = {}

def RegisterExternalModule(Type, Class):
    ExternalModules[Type] = Class

def BuildExternalModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    if Type in ExternalModules:
        return ExternalModules[Type](**kw)
    else:
        return None

class LogComponent:
    # method for class
    def SetLog(self, Log, SetForSubModules=True):
        self._Log = Log
        if hasattr(self, "LogCache"):
            for Log in self.LogCache:
                self._Log.Add(Log)
        if SetForSubModules:
            self.SetLogRecur()
        return self
    def SetLogRecur(self, Log=None):
        if Log is None:
            _Log = self._Log
        else:
            _Log = Log
        for Name, SubModule in self.SubModules.items():
            SubModule.SetLog(_Log)
        return self
    def Log(self, Content, Type="Unknown"):
        Param = self.Param
        log = DLUtils.param({
                "Subject": Param._PATH,
                "Type": Type,
                "Content": Content
            })
        if not hasattr(self, "_Log"):
            if not hasattr(self, "LogCache"):
                self.LogCache = []
            self.LogCache.append(log)
        else:
            self._Log.Add(log)
        return self

from .abstract_module import AbstractModule, EmptyModule

class AbstractOperator(AbstractModule):
    # operation module without tensor or trainable parameter
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
    def AddSubModule(self, Name, SubModule):
        raise Exception("AbstractOperator module.")
try:
    from .abstract_network import AbstractNetwork
    class AbstractModuleGroup(AbstractNetwork):
        def __init__(self, *List, **Dict):
            super().__init__(**Dict)
            if len(List) == 0:
                ModuleList = Dict.get("ModuleList")
            else:
<<<<<<< HEAD
                ModuleList = List
        self.ModuleList = []
        if ModuleList is not None:
            if isinstance(ModuleList, tuple):
                ModuleList = list(ModuleList)
            assert isinstance(ModuleList, list) or isinstance(ModuleList, dict)
            self.SetModuleList(ModuleList)
    def SetModuleList(self, ModuleList):
        Param = self.Param
        if isinstance(ModuleList, list):
            for Index, SubModule in enumerate(ModuleList):
                self.AddSubModule(f"L{Index}", SubModule)
            self.ModuleList = ModuleList
        if isinstance(ModuleList, dict):
            for Name, SubModule in ModuleList.items():
                self.AddSubModule(
                    Name, SubModule
                )
            self.ModuleList = list(ModuleList.values())
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if self.IsLoad():
            self.ModuleList = list(self.SubModules.values())
        self.ModuleNum = Param.Module.Num = len(self.ModuleList)
        return super().Init(IsSuper=True, IsRoot=IsRoot)

from .module_graph import ModuleGraph
from .module_series import ModuleList, ModuleSeries, _ModuleList, _ModuleSeries
=======
                assert Dict.get("ModuleList") is None
                if len(List) == 1 and DLUtils.IsIterable(List[0]):
                    if isinstance(List[0], dict):
                        ModuleList = List[0]
                    else:
                        ModuleList = List[0]
                else:
                    ModuleList = List
            self.ModuleList = []
            if ModuleList is not None:
                if isinstance(ModuleList, tuple):
                    ModuleList = list(ModuleList)
                assert isinstance(ModuleList, list) or isinstance(ModuleList, dict)
                self.SetModuleList(ModuleList)
        def SetModuleList(self, ModuleList):
            Param = self.Param
            if isinstance(ModuleList, list):
                for Index, SubModule in enumerate(ModuleList):
                    self.AddSubModule(f"L{Index}", SubModule)
                self.ModuleList = ModuleList
            if isinstance(ModuleList, dict):
                for Name, SubModule in ModuleList.items():
                    self.AddSubModule(
                        Name, SubModule
                    )
                self.ModuleList = list(ModuleList.values())
            return self
        def Init(self, IsSuper=False, IsRoot=True):
            Param = self.Param
            if self.IsLoad():
                self.ModuleList = list(self.SubModules.values())
            self.ModuleNum = Param.Module.Num = len(self.ModuleList)
            return super().Init(IsSuper=True, IsRoot=IsRoot)
except Exception:
    pass
try:
    from .module_graph import ModuleGraph
except Exception:
    pass
try:
    from .module_series import ModuleList, ModuleSeries, _ModuleList, _ModuleSeries
except Exception:
    pass
>>>>>>> 312cd1e34230841141c04fa6d32e6782cd09db27
try:
    from ..backend.torch.module import TorchModuleWrapper, TorchModelWrapper, TorchModule, TorchModuleParallel
except Exception:
    pass