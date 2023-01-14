import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

import DLUtils
#from DLUtils.attr import *
import DLUtils.train.algorithm as algorithm
import DLUtils.train.evaluate as evaluate
import DLUtils.train.loss as loss

import DLUtils.train.EpochBatchTrain as EpochBatchTrain
from .EpochBatchTrain import EpochBatchTrainSession

import DLUtils.train.SingleClassification as SingleClassification

def TrainSession(Type="EpochBatchTrain"):
    if Type in ["EpochBatchTrain", "Epoch-Batch"]:
        return EpochBatchTrainSession()
    else:
        raise Exception()

def NotifyEpochIndex(ObjList, EpochIndex):
    for Obj in ObjList:
        Obj.NotifyEpochIndex(EpochIndex)

def NotifyBatchIndex(ObjList, BatchIndex):
    for Obj in ObjList:
        Obj.NotifyBatchIndex(BatchIndex)

def NotifyEpochNum(ObjList, EpochNum):
    for Obj in ObjList:
        Obj.NotifyEpochNum(EpochNum)

def NotifyBatchNum(ObjList, BatchNum):
    for Obj in ObjList:
        Obj.NotifyBatchNum(BatchNum)

def ParseRoutersFromOptimizeParam(param, **kw):
    Routers = DLUtils.PyObj()
    for Name, RouterParam in ListAttrsAndValues(param.Batch.Routers):
        Router = DLUtils.router.ParseRouterStaticAndDynamic(RouterParam, ObjRefList=[RouterParam, param], **kw)
        setattr(Routers, Name, Router)
    return Routers

def SetSaveDirForSavedModel(EpochIndex, BatchIndex):
    SaveDirForSavedModel = DLUtils.GetMainSaveDir() + "SavedModel/" + "Epoch%d-Batch%d/"%(EpochIndex, BatchIndex)
    DLUtils.SetSubSaveDir(SaveDirForSavedModel, Type="Obj")

def ParseEpochBatchFromStr(Str):
    MatchResult = re.match(r"^.*Epoch(-?\d*)-Batch(\d*).*$", Str)
    if MatchResult is None:
        raise Exception(Str)
    EpochIndex = int(MatchResult.group(1))
    BatchIndex = int(MatchResult.group(2))
    return EpochIndex, BatchIndex

def ClearGrad(TensorDict):
    #for Name, Tensor in TensorDict.items():
    for Tensor in TensorDict.values():
        if Tensor.grad is not None:
                Tensor.grad.detach_()
                Tensor.grad.zero_()

def EpochBatchInFloat(EpochIndex, BatchIndex, BatchNum):
    return EpochIndex + BatchIndex / BatchNum * 1.0


def Labels2OneHotVectors(Labels, VectorSize=None):
    # Labels: [SampleNum]
    SampleNum = Labels.shape[0]
    Labels = DLUtils.ToNpArray(Labels, dtype=np.int32)
    if VectorSize is None:
        LabelMin, LabelMax = np.min(Labels), np.max(Labels)
        VectorSize = LabelMax
    OneHotVectors = np.zeros((SampleNum, VectorSize), dtype=np.float32)
    OneHotVectors[range(SampleNum), Labels] = 1
    return OneHotVectors

# def Probability2MaxIndex(Probability):
#     # Probability: [BatchSize, ClassNum]
#     return torch.argmax(Probability, axis=1)

def CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2):
    if EpochIndex1 < EpochIndex2:
        return -1
    elif EpochIndex1 > EpochIndex2:
        return 1
    else:
        if BatchIndex1 < BatchIndex2:
            return -1
        elif BatchIndex1 > BatchIndex2:
            return 1
        else:
            return 0   

def CmpEpochBatchData(data1, data2):
    EpochIndex1 = data1.EpochIndex
    BatchIndex1 = data1.BatchIndex
    EpochIndex2 = data2.EpochIndex
    BatchIndex2 = data2.BatchIndex
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def CmpEpochBatchDict(Dict1, Dict2):
    EpochIndex1 = Dict1["Epoch"]
    BatchIndex1 = Dict1["Batch"]
    EpochIndex2 = Dict2["Epoch"]
    BatchIndex2 = Dict2["Batch"]
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def CmpEpochBatchObj(Obj1, Obj2):
    EpochIndex1 = Obj1.GetEpochIndex()
    BatchIndex1 = Obj1.GetBatchIndex()
    EpochIndex2 = Obj2.GetEpochIndex()
    BatchIndex2 = Obj2.GetBatchIndex()
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def GetEpochBatchIndexFromPyObj(Obj):
    if hasattr(Obj, "Epoch"):
        EpochIndex = Obj.Epoch
    elif hasattr(Obj, "EpochIndex"):
        EpochIndex = Obj.EpochIndex
    else:
        raise Exception()
    
    if hasattr(Obj, "Batch"):
        BatchIndex = Obj.Batch
    elif hasattr(Obj, "BatchIndex"):
        BatchIndex = Obj.BatchIndex
    else:
        raise Exception()
    return EpochIndex, BatchIndex
