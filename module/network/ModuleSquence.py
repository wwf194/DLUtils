import DLUtils

from ..AbstractModule import AbstractNetwork
class ModuleSequence(AbstractNetwork):
    def __init__(self, ModuleList=None, Log=None):
        super().__init__(Log)
        if ModuleList is None:
            ModuleList = []
        assert isinstance(ModuleList, list)
        self.SetModuleList(ModuleList)
        self.Param.absorb_dict({
            "_CLASS": "DLUtils.NN.ModuleSequence",
            "Module.Num": len(self.ModuleList)
        })
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
        for Index, SubModule in enumerate(ModuleList):
            Key = str(Index)
            self.SubModules[Key] = SubModule
            setattr(Param.SubModules, Key, SubModule.Param)
            SubModule.Param._PATH = Param._PATH + "." + Key
        self.ModuleNum = len(ModuleList)
        self.ModuleList = ModuleList
        return self
    def Receive(self, Input):
        for ModuleIndex in range(self.ModuleNum):
            Output = self.ModuleList[ModuleIndex](Input)
            Input = Output
        return Output