import DLUtils
class AbstractModule:
    def __init__(self):
        self.Name = "NullName"
        self.SubModules = {}
        Param = self.Param = DLUtils.Param()
        Param.Tensors = []
        Param._CLASS = "DLUtils.NN.AbstractModule"
        Param._PATH = "Root"
    def ExtractParam(self, RetainSelf=True):
        Param = self.Param
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if len(Param.Tensors) == 0:
                Param.delattr("Tensors")
        self.UpdateDictFromTensor()
        return Param
    def ExtractParamRecur(self, Param):
        for Name, SubModule in self.SubModules:
            setattr(Param, Name, SubModule.ExtractParam())
        return self.Param
    def LoadParam(self, Param):
        self.Param = Param
        if Param.hasattr("Tensors"):
            self.UpdateTensorFromDict()
        else:
            Param.Tensors = []
        self.LoadParamRecur(Param)
    def LoadParamRecur(self, Param):
        for Name, SubModule in Param.SubModules.items():
            ModuleClass = DLUtils.python.ParseClass(SubModule._CLASS)
            SubModule._PATH = Param._PATH + "." + Name
            self.SubModules[Name] = ModuleClass().LoadParam(SubModule)
    def ToFile(self, FilePath):
        Param = self.ExtractParam(RetainSelf=False)
        DLUtils.file.Obj2File(
            Param,
            FilePath
        )
        return self
    def FromFile(self, FilePath):
        self.SubModules = {}
        Param = DLUtils.file.File2Obj(FilePath)
        self.LoadParam(Param)
        return self
    def SetAsRoot(self):
        self.Param._IS_ROOT = True
        return self
    def UpdateTensorFromDict(self):
        Param = self.Param
        for Name in Param.Tensors:
            setattr(self, Name, DLUtils.ToTorchTensorOrNum(getattr(Param.Data, Name)))
    def UpdateDictFromTensor(self):
        Param = self.Param
        for Name in Param.Tensors:
            setattr(Param.Data, Name, DLUtils.ToNpArrayOrNum(getattr(self, Name)))
    def ToJsonFile(self, FilePath):
        self.ExtractParam(RetainSelf=True).ToJsonFile(FilePath)
        return self