import numpy as np

import DLUtils



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
    Accuracy.RatioCorrect = 1.0 * Accuracy.NumCorrect / Accuracy.Num
    return Accuracy

