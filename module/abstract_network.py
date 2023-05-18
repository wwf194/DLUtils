import DLUtils
import torch
from .abstract_module import AbstractModule
class AbstractNetwork(AbstractModule):
    # network with trainable weights.
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        Param.setemptyattr("TrainParam")
        Param.setemptyattr("Tensor")
    def SetTensor(self, Name=None, Path=None, Data=None, Trainable=False, **Dict):
        if Trainable:
            self.SetTrainParam(Name=Name, Path=Path, Data=Data, **Dict)
            return self
        if Name is not None:
            assert Data is not None
            self._SetTensor(Name, Path, Data)
        for _Name, _Data in Dict.items():
            self._SetTensor(_Name, None, _Data)
        return self
    SetUntrainableParam = SetUnTrainableParam = SetUnTrainableParam = SetTensor
    def _SetTensor(self, Name, Path=None, Data=None):
        if Path is None:
            Path = Name
        assert Data is not None
        Param = self.Param
        Param.Tensor.setattr(Name, Path)
        Param.setattr(Path, Data)
        return self
    def RegisterTensor(self, Name=None, Path=None, **Dict):
        if Name is not None:
            assert Path is not None
            self._RegisterTensor(Name, Path)
        for _Name, _Path in Dict.items():
            self._RegisterTensor(_Name, _Path)
        return self
    def _RegisterTensor(self, Name, Path=None):
        if Path is None:
            Path = Name
        Param = self.Param
        Param.Tensor.setattr(Name, Path)
        return self
    RegisterTensorName = RegisterTensor
    def SetTensorAsTrainParam(self, Name):
        Param = self.Param
        assert Param.Tensor.hasattr(Name)
        Path = Param.Tensor.getattr(Name)
        Param.TrainParam.setattr(Name, Path)
        return self
    SetTrainable = SetTensorAsTrainParam
    def SetTrainParam(self, Name=None, Path=None, Data=None, **Dict):
        if Name is not None:
            assert Data is not None
            self._SetTrainParam(Name, Path, Data)
        for _Name, _Data in Dict.items():
            self._SetTrainParam(_Name, None, _Data)
        return self
    SetTrainableParam = SetTrainParam
    def _SetTrainParam(self, Name, Path=None, Data=None):
        Param = self.Param
        if Path is None:
            Path = "Data." + Name
        assert Data is not None
        if hasattr(Data, "requires_grad"):
            Data.requires_grad = True
        Param.TrainParam.setattr(Name, Path)
        Param.setattr(Path, Data)
        self._RegisterTensor(Name, Path)
        setattr(self, Name, Data)
        return self
    def RegisterTrainParam(self, Name, Path=None):
        Param = self.Param
        if Path is None:
            if self.Tensor.hasattr(Name):
                Path = self.Tensor.getattr(Name)
                Param.TrainParam.setattr(Name, Path)
        else:
            Param.TrainParam.setattr(Name, Path)
            if self.Tensor.hasattr(Name):
                assert Path == self.Tensor.getattr(Name) 
            else: # train param is also tensor
                self.RegisterTensor(Name, Path)
        return self
    RegisterTrainParamName = RegisterTrainParam
    def GetTensor(self, Name):
        assert not self.HandleTensorBySelf()
        Param = self.Param
        # return Param.Data.getattr(Name)
        assert Param.Tensor.hasattr(Name)
        if hasattr(self, Name):
            return getattr(self, Name)
        else:
            Path = Param.Tensor.getattr(Name)
            return Param.getattr(Path)
    def UpdateTensorFromDict(self, Recur=False):
        Param = self.Param        
        if self.HandleTensorBySelf():
            pass
        else:
            if Param.hasattr("Tensor"):
                for Name, Path in Param.Tensor.items():
                    assert Param.hasattr(Path)
                    TensorData = Param.getattr(Path)
                    Tensor = DLUtils.ToTorchTensorOrNum(TensorData)
                    if hasattr(self, "Device"):
                        TensorDevice = Tensor.to(self.Device).detach()
                        TensorDevice.requires_grad = Tensor.requires_grad
                    else:
                        TensorDevice = Tensor
                    setattr(self, Name, TensorDevice)
            if Param.hasattr("TrainParam"):
                for Name, Path in Param.TrainParam.items():
                    assert Param.Tensor.hasattr(Name)
                    Tensor = getattr(self, Name)
                    Tensor.requires_grad = True
                    Tensor = torch.nn.Parameter(Tensor, requires_grad=True)
                    setattr(self, Name, Tensor)
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateTensorFromDict"):
                    SubModule.UpdateTensorFromDict(Recur=True)
        self.OnTensorMovement()
        return self
    def UpdateDictFromTensor(self, Recur=False):
        Param = self.Param
        if self.HandleTensorBySelf():
            pass
        else:
            if Param.hasattr("TrainParam"):
                for Name, Path in Param.TrainParam.items():
                    if hasattr(self, Name):
                        TrainParamData = getattr(self, Name)
                        Param.setattr(Path, DLUtils.ToNpArray(TrainParamData))
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateDictFromTensor"):
                    SubModule.UpdateDictFromTensor(Recur=True)
        return self
    def ExtractTrainParam(self, TrainParamDict={}, PathStrPrefix=True, Recur=True):
        self.UpdateDictFromTensor(Recur=False)
        # self.UpdateTensorFromDict()
        Param = self.Param
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        if Param.get("TrainParam") is not None:
            for Name, Path in Param.TrainParam.items():
                TrainParam = getattr(self, Name)
                assert isinstance(TrainParam, torch.nn.Parameter)
                TrainParamDict[Prefix + Name] = TrainParam
        if Recur:
            self.ExtractTrainParamRecur(TrainParamDict=TrainParamDict, PathStrPrefix=PathStrPrefix)
        return TrainParamDict
    def ExtractTrainParamRecur(self, TrainParamDict={}, PathStrPrefix=True):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "ExtractTrainParam"):
                SubModule.ExtractTrainParam(
                    TrainParamDict=TrainParamDict,
                    PathStrPrefix=PathStrPrefix,
                    Recur=True
                )
        return TrainParamDict
    def PlotWeight(self, SaveDir=None, SaveName=None):
        Param = self.Param
        Param = self.ExtractParam()
        SavePath = DLUtils.ParseSavePath(SaveDir, SaveName, SaveNameDefault=Param._PATH)
        if Param.hasattr("TrainParam"):
            for WeightName in Param.TrainParam:
                # Data = Param.Data.getattr(WeightName)
                Data = Param.getattr(WeightName)
                if hasattr(Data, "shape"):
                    DimNum = len(Data.shape)
                else:
                    DimNum = 0
                if DimNum == 1 or DimNum == 0:
                    # DLUtils.plot.PlotData1D(
                    #     Name=WeightName,
                    #     Data=Data,
                    #     SavePath=SavePath + "." + WeightName + ".svg",
                    #     #XLabel="Dimension 0", YLabel="Dimension 0"
                    # )
                    DLUtils.plot.PlotDataAndDistribution1D(
                        Name=WeightName,
                        Data=Data,
                        SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                        XLabel="Dimension 0", YLabel="Dimension 0",
                        Title=f"Shape {Data.shape[0]}"
                    )
                elif DimNum == 2:
                    # DLUtils.plot.PlotData2D(
                    #     Name=WeightName,
                    #     Data=Data,
                    #     SavePath=SavePath + "." + WeightName + ".svg",
                    #     XLabel="Output Dimension", YLabel="Input Dimension"
                    # )
                    DLUtils.plot.PlotDataAndDistribution2D(
                        Name=WeightName,
                        Data=Data,
                        SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                        XLabel="Output Dimension", YLabel="Input Dimension",
                        TitlePlot=f"Shape {Data.shape[0], Data.shape[1]}"
                    )
            self.PlotWeightRecur(SaveDir, SaveName)
        return self
    def ExtractParam(self, RetainSelf=True):
        Param = DLUtils.Param(self.Param)
        self.UpdateDictFromTensor(Recur=False)
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if Param.hasattr("TrainParam") and len(Param.TrainParam) == 0:
                Param.delattr("TrainParam")
        Attr = Param.delattrifexists("BindModules")
        self.ExtractParamRecur(Param, RetainSelf)
        return Param
    def LoadParam(self, Param):
        super().LoadParam(Param)
        # if Param.hasattr("TrainParam"):
        #     self.UpdateTensorFromDict()
        self.LoadParamRecur(Param)
        return self
    def PlotWeightRecur(self, SaveDir, SaveName):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "PlotWeight"):
                SubModule.PlotWeight(SaveDir, SaveName)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.UpdateTensorFromDict()
        if IsRoot:
            self.Log(f"{Param._CLASS}: initialization finished.", Type="initialization")
        return self
    def OnTensorMovement(self):
        for Event in self.EventDict.TensorMovement:
            Event(Model=self, Param=self.ExtractTrainParam())
    def parameters(self):
        return
    def ReportModelSizeInFile(self, SavePath):
        Dict = self.ReportModelSize(IsRoot=True)
        DLUtils.file.JsonDict2JsonFile(Dict, SavePath)
        return self
    def ReportModelSize(self, IsRoot=True):
        Param = self.Param
        TensorDict = {}
        SubModules = {}
        Dict = {
            "Tensors": TensorDict,
            "SubModules": SubModules,
            "ByteNum": 0,
            "ParamNum": 0
        }
        for Name, TensorPath in Param.Tensor.items():
            Tensor = getattr(self, Name)
            ParamNum = DLUtils.GetTensorElementNum(Tensor)
            
            ByteNum = DLUtils.GetTensorByteNum(Tensor)
            Shape = list(Tensor.size())
            TensorDict[Name] = {
                "ByteNum": ByteNum,
                "ParamNum": ParamNum,
                "Shape": Shape
            }
            Dict["ByteNum"] += ByteNum
            Dict["ParamNum"] += ParamNum


        self.ReportModelSizeRecur(Dict)
        ByteNumStr = DLUtils.ByteNum2Str(Dict["ByteNum"])
        ParamNumStr = DLUtils.Num2Str(Dict["ParamNum"])
        Dict["ByteNumStr"] = ByteNumStr
        Dict["ParamNumStr"] = ParamNumStr
        if IsRoot:
            print("Model Size: %s. ParamNum: %s"%(ByteNumStr, ParamNumStr))
        return Dict
    def ReportModelSizeRecur(self, Dict):
        SubModules = Dict["SubModules"]
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "ReportModelSize"):
                SubModuleDict = SubModule.ReportModelSize(IsRoot=False)
                SubModules[Name] = SubModuleDict
                Dict["ByteNum"] += SubModuleDict["ByteNum"]
                Dict["ParamNum"] += SubModuleDict["ParamNum"]
        return self