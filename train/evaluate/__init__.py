import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()

from ..EpochBatchTrain.Component import EvaluatorPredAndTargetSelect1FromN, EvaluatorPredAndTarget
def Evaluator(Type):
    if Type in EvaluatorMap:
        return EvaluatorMap[Type]()
    else:
        raise Exception()

EvaluatorMap = DLUtils.IterableKeyToElement({
    ("ImageClassification"): EvaluatorPredAndTargetSelect1FromN,
    ("PredAndTarget"): EvaluatorPredAndTarget
})

def EvaluationLog(Type):
    if Type in EvaluationLogMap:
        return EvaluationLogMap[Type]()
    else:
        raise Exception()

from ..Select1FromN import EvaluationLogSelect1FromN, EvaluationLogLoss
EvaluationLogMap = DLUtils.IterableKeyToElement({
    ("ImageClassification"): EvaluationLogSelect1FromN,
    ("EvaluationLogLoss", "PredAndTarget"): EvaluationLogLoss
})

class _Evaluator:
    def __init__(self):
        self.Param = DLUtils.Param({})
        Param = self.Param
        Param.Log = DLUtils.Param([])

def InitAccuracy():
    Accuracy = DLUtils.EmptyPyObj()
    Accuracy.NumTotal = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def ResetAccuracy(Accuracy):
    Accuracy.NumTotal = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def CalculateAccuracy(Accuracy):
    Accuracy.RateCorrect = 1.0 * Accuracy.NumCorrect / Accuracy.Num
    return Accuracy

