import DLUtils
import torch

def GradientDescendOptimizer(Type, **Dict):
    assert Type in ["GradientDescend"]
    SubType = Dict.get("SubType")
    return GradientDescend(SubType=SubType)

class GradientDescend(DLUtils.module.AbstractModule):
    ParamMap = DLUtils.IterableKeyToElement({
        ("LearningRate", "LR", "lr"): "LearningRate",
        ("Nesterov"): "Nesterov.Enable",
        ("Alpha"): "Momentum.Value",
        ("Beta"): ""
        # ("Momentum"): "Momentum.Enable"
    })
    def __init__(self, SubType=None):
        super().__init__()
        if SubType is not None:
            self.SetSubType(SubType)
    def SGD(self, *List, **Dict):
        self.__class__ = SGD
        assert isinstance(self, SGD)
        self._INIT(*List, **Dict)
        return self
    def Adam(self, *List, **Dict):
        self.__class__ = Adam
        assert isinstance(self, Adam)
        self.SetParam(*List, **Dict)
        return self
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in Dict.items():
            if Key in ["Momentum"]:
                if isinstance(Value, float):
                    assert Value > 0.0
                    if Value == 0.0:
                        Param.Momentum.Enable = False
                    else:
                        Param.Momentum.Enable = True
                        Param.Momentum.Value = Value
                elif isinstance(Value, bool):
                    Param.Momentum.Enable = Value
                else:
                    super().SetParam()
                Dict.pop(Key)
        super().SetParam(**Dict)
        return super().SetParam(**Dict)
    def SetSubType(self, SubType, **Dict):
        if SubType in ["SGD", "sgd"]:
            return self.SGD()
        elif SubType in ["Adam", "adam"]:
            return self.Adam()
        else:
            raise Exception()
    SubType = SetSubType
    # def SetTrainParam(self, TrainParam):
    #     self.TrainParam = TrainParam
    #     return self
    # def BindEvaluator(self, Evaluator):
    #     self.AddSubModule("Evaluator", Evaluator)
    #     return self
    def BindModel(self, Model):
        self.Model = Model
        # Model.On("TensorMovement", self.ResetOptimizer)
        return self
    def BeforeTrain(self, Dict):
        # self.SetTrainParam(
        #     self.Model.ExtractTrainParam()
        # )
        self.ResetOptimizer()
        return self
    def SetDevice(self, Device, IsRoot=False):
        self.Device = Device
        # self.ResetOptimizer()
        super().SetDevice(Device, IsRoot=IsRoot)
        return self
    def ResetOptimizer(self, *List, **Dict):
        raise Exception()

from .sgd import SGD
from .adam import Adam