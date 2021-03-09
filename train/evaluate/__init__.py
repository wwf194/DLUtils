import DLUtils
import torch
import numpy as np

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

