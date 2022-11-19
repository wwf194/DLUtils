import DLUtils
class AbstractModule:
    def __init__(self):
        self.Name = "NullName"
        self.SubModules = {}
    def ToDict(self):
        SubModuleDict = {}
        for Name, Module in self.SubModules:
            SubModuleDict[Name] = Module.ToDict()
        Dict = {
            "Class": "DLUtils.NeuralNetwork.AbstractNetwork",
        }
        Dict = self.ToDictRecur(Dict)
        return Dict
    def ToDictRecur(self, Dict):
        SubModuleDict = {}
        for Name, Module in self.SubModules:
            SubModuleDict[Name] = Module.ToDict()
        Dict["SubModules"] = SubModuleDict
        return Dict
    def FromDict(self, Dict):
        self.FromDictRecur(Dict)
        self.Dict = Dict
    def FromDictRecur(self, Dict):
        for Name, ModuleDict in Dict["SubModules"].items():
            ModuleClass = DLUtils.SearchClass(ModuleDict["Class"])()
            self.SubModules[Name] = ModuleClass().FromDict(ModuleDict)
        Dict.pop("SubModules")
    def ToFile(self, FilePath):
        DLUtils.file.Obj2File(
            self.ToDict(),
            FilePath
        )
    def FromFile(self, FilePath):
        self.SubModules = {}
        Dict = DLUtils.file.File2Obj(FilePath)
        self.FromDict(Dict)