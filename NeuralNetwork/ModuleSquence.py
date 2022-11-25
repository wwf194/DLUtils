import DLUtils

from .AbstractModule import AbstractModule
class ModuleSequence(AbstractModule):
    def __init__(self, ModuleList=None):
        super().__init__()
        if ModuleList is None:
            ModuleList = []
        assert isinstance(ModuleList, list)
        self.SetModuleList(ModuleList)
        self.Param.AbsorbDict({
            "_CLASS": "DLUtils.NN.ModuleSequence",
            "Module.Num": len(self.ModuleList)
        })
    def SetModuleList(self, ModuleList):
        Param = self.Param
        for Index, SubModule in enumerate(ModuleList):
            Key = str(Index)
            setattr(Param.SubModules, Key, SubModule.Param)
            SubModule.Param._PATH = Param._PATH + "." + Key
        self.ModuleNum = len(ModuleList)
        self.ModuleList = ModuleList
        return self