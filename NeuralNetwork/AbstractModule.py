import DLUtils
class AbstractModule:
    def __init__(self):
        self.Name = "NullName"
        self.SubModules = {}
        Param = self.Param = DLUtils.Param()
        Param.Tensors = []
        Param.Class = None
    def ToDict(self):
        self.UpdateDictFromTensor()
        Dict = self.Param.ToDict()
        Dict["Class"] = "DLUtils.NeuralNetwork.AbstractModule"
        Dict = self.ToDictRecur(Dict)
        return Dict
    def ToDictRecur(self, Dict):
        SubModuleDict = {}
        for Name, Module in self.SubModules:
            SubModuleDict[Name] = Module.ToDict()
        Dict["SubModules"] = SubModuleDict
        return Dict
    def FromDict(self, Dict):
        self.Param = DLUtils.Param().FromDict(Dict)
        self.UpdateTensorFromDict()
        self.FromDictRecur(Dict)
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
    def SetAsRoot(self):
        self.Param.IsRoot = True
    def UpdateTensorFromDict(self):
        Param = self.Param
        for Name in Param.Tensors:
            setattr(self, Name, DLUtils.ToTorchTensor(getattr(Param.Data, Name)))
    def UpdateDictFromTensor(self):
        Param = self.Param
        for Name in Param.Tensors:
            setattr(Param.Data, Name, DLUtils.ToNpArrayOrNum(getattr(self, Name)))