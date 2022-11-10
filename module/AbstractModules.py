import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
from DLUtils.attr import *

class AbstractModule:
    def __init__(self, **kw):
        return
    def RegisterExternalMethod(self, Name, Method):
        if not callable(Method):
            Method = DLUtils.parse.ResolveStr(Method)
        setattr(self, Name, Method)
    def LoadParam(self, param=None):
        param = DLUtils.ToPyObj(param)
        self.param = param
        return self
    def LoadData(self, data):
        self.data = data
        return self
    def LoadDataFromFile(self, FileDir):
        FilePath = FileDir + self.param.FullName + ".data"
        if DLUtils.ExistsFile(FilePath):
            self.LoadData(
                DLUtils.file.DataFile2PyObj(FilePath)
            )
        else:
            if hasattr(self.__class__, "DataIsNotEmpty") and self.__class__.DataIsNotEmpty is True:
                raise Exception()
            self.LoadData(DLUtils.EmptyPyObj())
        return self
    def LoadDataFromDir(self, LoadDir):
        FilePath = LoadDir = self.param.FullName + ".data"
        self.LoadDataFromFile(self, FilePath)
        return self
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        if hasattr(cache, "Tensors"):
            for ParamIndex in cache.Tensors:
                setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location).detach().requires_grad_(True))
        if hasattr(cache, "Modules"):
            for name, module in ListAttrsAndValues(cache.Modules):
                if hasattr(module, "SetTensorLocation"):
                    module.SetTensorLocation(Location)
                # else:
                #     if isinstance(module, nn.Module):
                #         DLUtils.AddWarning("%s is an instance of nn.Module, but has not implemented SetTensorLocation method."%name)
    def AddSubModule(self, Name, Module, Type=None):
        param = self.param
        cache = self.cache
        if not hasattr(param, "Modules"):
            param.Modules = DLUtils.EmptyPyObj()
        if hasattr(Module, "param"):
            setattr(param.Modules, Name, Module.param)
        else:
            assert Type is not None and isinstance(Type, str)
            setattr(param.Modules, Name, DLUtils.PyObj({"Type": Type}))
        setattr(cache.Modules, Name, Module)
    def HasSubModule(self, Name):
        return hasattr(self.param.Modules, Name)

class AbstractModuleWithParam(AbstractModule):
    def __init__(self, **kw):
        return
    def BeforeBuild(self, IsLoad, ParseParam=True):
        if hasattr(self, "HasBeforeBuild") and self.HasBeforeBuild is True:
            return self

        if not hasattr(self, "cache"):
            self.cache = DLUtils.EmptyPyObj()
        
        cache = self.cache
        IsInit = not IsLoad
        cache.IsLoad = IsLoad
        cache.IsInit = IsInit
        cache.__object__ = self

        if not hasattr(self, "data"):
            if IsLoad:
                if hasattr(self.__class__, "HasData") and self.__class__.HasData is True:
                    raise Exception()
                #DLUtils.AddWarning("Instance of class %s has not loaded data."%self.__class__)
            self.data = DLUtils.EmptyPyObj()

        #assert hasattr(self, "param")
        if hasattr(self, "param"):
            param = self.param
        else:
            #DLUtils.AddWarning("Instance of class %s has not loaded param."%self.__class__)
            param = self.param = DLUtils.PyObj()
            
        if not hasattr(param, "FullName"):
            if hasattr(self.__class__, "FullNameDefault"):
                param.FullName = self.__class__.FullNameDefault
            else:
                param.FullName = "DefaultFullName"

        if not hasattr(param, "Modules"):
            param.Modules = DLUtils.EmptyPyObj()
        #param.Modules.SetResolveBase()
        self.Modules = cache.Modules = DLUtils.EmptyPyObj()
        
        if not hasattr(param, "Dynamics"):
            param.Dynamics = DLUtils.EmptyPyObj()
        param.Dynamics.SetResolveBase()
        self.Dynamics = cache.Dynamics = cache.Dynamics = DLUtils.EmptyPyObj()

        param.SetResolveBaseRecur()
        if ParseParam:
            DLUtils.parse.ParsePyObjStatic(param, ObjCurrent=param)

        self.HasBeforeBuild = True
        
        return self
    def ParseRouters(self, **kw):
        GlobalParam = DLUtils.GetGlobalParam()
        param = self.param
        cache = self.cache
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__IsResolveBase__", "__Entry__"]):
            if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
                setattr(cache.Dynamics, Name, getattr(self, Name))
                continue
            if cache.IsInit:
                DLUtils.router.ParseRouterStatic(RouterParam)
                setattr(RouterParam, "Name", param.FullName + "." + Name) # For Debug
            setattr(cache.Dynamics, Name, DLUtils.EmptyPyObj())
        
        ObjRefList = [
            cache.Modules, cache.Dynamics, cache,
            param, self, DLUtils.transform.operator,
        ]
        if hasattr(GlobalParam.cache, "AdditionalObjRefListForParseRouters"):
            ObjRefList += GlobalParam.cache.AdditionalObjRefListForParseRouters
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__IsResolveBase__", "__Entry__"]):
            if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
                continue
            getattr(cache.Dynamics, Name).FromPyObj(
                DLUtils.router.ParseRouterDynamic(
                    RouterParam, 
                    ObjRefList = ObjRefList, ObjRoot = DLUtils.GetGlobalParam(),
                    InPlace=False
                )
            )
        return
    def InitModule(self, param=None, data=None, ClassPath=None, **kw):
        LoadDir = kw.get("LoadDir")
        FullName = kw.setdefault("FullName", "Unnamed")
        HasTensor = kw.setdefault("HasTensor", True)

        if param is None:
            param = DLUtils.EmptyPyObj()
        
        EnsureAttrs(param, "FullName", default=FullName)

        param.cache.__object__ = self
        if hasattr(param, "Modules"):
            #param.Modules.SetResolveBase()
            pass
        if hasattr(param, "Dynamics"):
            pass
            #param.Dynamics.SetResolveBase()

        if data is None:
            if LoadDir is not None:
                DataPath = LoadDir + param.FullName + ".data"
                if DLUtils.FileExists(DataPath):
                    data = DLUtils.json.DataFile2PyObj(DataPath)
                else:
                    data = DLUtils.EmptyPyObj()
            else:
                data = DLUtils.EmptyPyObj()

        cache = DLUtils.EmptyPyObj()
        if LoadDir is not None:
            cache.LoadDir = LoadDir
        else:
            cache.LoadDir = None
        if ClassPath is not None:
            param.ClassPath = ClassPath
        
        if not hasattr(cache, "Modules"):
            self.Modules = cache.Modules = DLUtils.EmptyPyObj()
        if not hasattr(cache, "Modules"):
            self.Dynamics = cache.Dynamics = DLUtils.EmptyPyObj()

        if HasTensor:
            cache.Tensors = []

        self.param = param
        self.data = data
        self.cache = cache
    def SetFullName(self, FullName):
        cache = self.cache
        param = self.param
        if FullName not in [""]:
            param.FullName = FullName
        if hasattr(cache, "Modules"):   
            for Name, Module in ListAttrsAndValues(cache.Modules):
                if hasattr(Module, "SetFullName"):
                    if FullName in [""]:
                        Module.SetFullName(Name)
                    else:
                        Module.SetFullName(FullName + "." + Name)

    def GetFullName(self):
        if hasattr(self, "param"):
            if hasattr(self.param, "FullName"):
                return self.param.FullName
        return None
    def DoInitTasks(self):
        param = self.param
        EnsureAttrs(param, "InitTasks", default=[])
        for Task in self.param.InitTasks:
            DLUtils.DoTask(Task, ObjCurrent=self.cache, ObjRoot=DLUtils.GetGlobalParam())
    def LoadFromParam(self):
        self.Build(IsLoad=True)
    def ParseParam(self, **kw):
        GlobalParam = DLUtils.GetGlobalParam()
        DLUtils.parse.ParsePyObjStatic(self.param, ObjCurrent=self.param, ObjRoot=GlobalParam)
    def ToFile(self, SaveDir=None, SaveName=None):
        param = self.param
        data = self.data
        cache = self.cache
        if SaveName is None:
            SaveName = self.param.FullName
        if not data.IsEmpty():
            DLUtils.file.PyObj2DataFile(data,  SaveDir + SaveName + ".data")
        DLUtils.file.PyObj2JsonFile(param, SaveDir + SaveName + ".jsonc")
        
        if hasattr(cache, "Modules"):
            for Name, Module in cache.Modules.Items():
                if hasattr(Module, "ToFile"):
                    Module.ToFile(SaveDir)        
        return self
    def FromFile(self, SaveDir, SaveName, LoadParam=True, IsRoot=None):
        if IsRoot is False:
            LoadParam = True
        elif IsRoot is True:
            LoadParam = False
        if DLUtils.file.ExistsFile(SaveDir + SaveName + ".data"):
            self.data  = DLUtils.file.DataFile2PyObj(SaveDir + SaveName + ".data")
        else:
            if hasattr(self.__class__, "HasData") and self.__class__.HasData is True:
                raise Exception()
        if LoadParam:
            self.param = DLUtils.file.JsonFile2PyObj(SaveDir + SaveName + ".jsonc")
        # param = self.param
        # cache = self.cache = DLUtils.EmptyPyObj()

        # This shoud be done in Build.
        # if hasattr(param, "Modules"):
        #     for ModuleName, ModuleParam in param.Modules.ListAttrsAndValues():
        #         if isinstance(ModuleParam, str) and ModuleParam in ["ClassMethod", "Internal", "External"]:
        #             continue
        #         if hasattr(ModuleParam, "Type") and ModuleParam.Type in ["Internal", "External"]: 
        #             continue
        #         module = DLUtils.module.BuildModule(ModuleParam).LoadParam(ModuleParam)
        #         if hasattr(module, "FromFile"):
        #             module.FromFile(
        #                 SaveDir, SaveName + "." + ModuleName, LoadParam=False
        #             ).Build(IsLoad=True)
        #         setattr(cache.Modules, "ModuleName", module)
        return self
    def OverwriteParam(self, ParamPath, Value):
        SetAttrs(self.param, ParamPath, value=Value)
    def InitForNonModel(self, param=None, data=None, ClassPath=None, **kw):
        self.InitForModule(self, param, data, ClassPath, HasTensor=False, **kw)
        return
    def BuildModules(self, IsLoad=False, LoadDir=None):
        # initialize modules
        # for module in ListAttrs(param.modules):
        IsInit = not IsLoad
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__IsResolveBase__"]):
            if isinstance(ModuleParam, str):
                assert ModuleParam in ["Internal", "External"]
                continue
            ModuleParam.Name = Name
            ModuleParam.FullName = param.FullName + "." + Name

            if not HasAttrs(ModuleParam, "Type"):
                if HasAttrs(ModuleParam, "Name"):
                    SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
                else:
                    raise Exception()

            if ModuleParam.Type in ["Internal", "External"]:
                continue
            module = DLUtils.module.BuildModule(ModuleParam)
            if IsInit:
                if hasattr(module, "LoadParam"):
                    module.LoadParam(ModuleParam)
            else:
                if hasattr(module, "LoadParam"):
                    module.LoadParam(ModuleParam)
                if hasattr(module, "LoadDataFromFile"):
                    module.LoadDataFromFile(LoadDir)
            # if isinstance(module, nn.Module) and isinstance(self, nn.Module):
            #     self.add_module(Name, Module)
            setattr(cache.Modules, Name, module)
    # def BuildModules(self):                                                                                
    #     # initialize modules                                                                         
    #     # for module in ListAttrs(param.modules):                                                        
    #     param = self.param
    #     cache = self.cache
    #     for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__IsResolveBase__"]):
    #         ModuleParam.Name = Name
    #         ModuleParam.FullName = param.FullName + "." + Name

    #         if not HasAttrs(ModuleParam, "Type"):
    #             if HasAttrs(ModuleParam, "Name"):
    #                 SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
    #             else:
    #                 raise Exception()
    #         if ModuleParam.Type in ["Internal", "External"]:
    #             continue
    #         if cache.IsInit:
    #             module = DLUtils.module.BuildModule(ModuleParam)
    #             if hasattr(module, "LoadParam"):
    #                 module.LoadParam(ModuleParam)
    #         else:
    #             module = DLUtils.module.BuildModule(ModuleParam).LoadDataFromDir(cache.LoadDir)
    #         # if isinstance(module, nn.Module) and isinstance(self, nn.Module):
    #         #     self.add_module(Name, module)
    #         setattr(cache.Modules, Name, module)

class AbstractModuleWithoutParam(AbstractModule):    
    def __init__(self, **kw):
        return
    def ToFile(self, FilePath):
        if not FilePath.endswith(".data"):
            FilePath += ".data"
        DLUtils.file.PyObj2DataFile(self.data, FilePath)
        return self
    def FromFile(self, SaveDir, SaveName):
        SavePath = SaveDir + SaveName + ".data"
        self.data = DLUtils.file.DataFile2PyObj(SavePath)
        return self
    def BeforeBuild(self, IsLoad=False):
        if hasattr(self, "HasBeforeBuild") and self.HasBeforeBuild is True:
            return

        self.cache = DLUtils.EmptyPyObj()
        if not (hasattr(self.__class__, "HasData") and self.__class__.HasData is False) and not hasattr(self, "data"):
            self.data = DLUtils.EmptyPyObj()
            self.HasData = True
        else:
            self.HasData = False
        
        self.HasBeforeBuild = True
        return self