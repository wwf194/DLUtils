import DLUtils
import torch

def GradientDescendOptimizer(Type, **Dict):
    assert Type in ["GradientDescend"]
    SubType = Dict.get("SubType")
    return GradientDescend(SubType=SubType)

class GradientDescend(DLUtils.module.AbstractModule):
    def __init__(self, SubType=None):
        super().__init__()
        if SubType is not None:
            self.SetSubType(SubType)
    def SetSubType(self, SubType, **Dict):
        if SubType in ["SGD", "sgd"]:
            self.__class__ = SGD #
            assert isinstance(self, SGD)
            return self
        elif SubType in ["Adam", "adam"]:
            self.__class__ = Adam
            assert isinstance(self, Adam)
            return self
        else:
            raise Exception()
    def SetTrainParam(self, TrainParam):
        self.TrainParam = TrainParam
        return self
    def BindEvaluator(self, Evaluator):
        self.AddSubModule("Evaluator", Evaluator)
        return self
    def BindModel(self, Model):
        self.Model = Model
        return self
    def BeforeTrain(self, Device):
        self.SetTrainParam(
            self.Model.ExtractTrainParam()
        )
        self.ResetOptimizer()
        return self

from .SGD import SGD
from .Adam import Adam

