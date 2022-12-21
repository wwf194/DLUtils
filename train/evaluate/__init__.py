import DLUtils
import torch
import numpy as np


from .EpochBatchTrainEvaluation import XFixedSizeYFixedSizeProb
def Evaluator(Type):
    if Type in ["ImageClassification"]:
        return XFixedSizeYFixedSizeProb()
    else:
        raise Exception()

class _Evaluator:
    def __init__(self):
        self.Param = DLUtils.Param({})
        Param = self.Param
        Param.Log = DLUtils.Param([])

