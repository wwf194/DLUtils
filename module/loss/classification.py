
import torch
import DLUtils
def Probability2MostProbableIndex(Probability):
    # Probability: [BatchSize, ClassNum]
    #Max, MaxIndices = torch.max(Probability, dim=1)
    #return DLUtils.TorchTensor2NpArray(MaxIndices) # [BatchSize]
    return torch.argmax(Probability, axis=1)

def LogAccuracyForSingleClassPrediction(ClassIndexPredicted, ClassIndexTruth, log):
    #log = DLUtils.ParseLog(log)
    ClassIndexPredicted = DLUtils.ToNpArray(ClassIndexPredicted)
    ClassIndexTruth = DLUtils.ToNpArray(ClassIndexTruth)
    NumCorrect, NumTotal = DLUtils.evaluate.CalculateAccuracyForSingelClassPrediction(ClassIndexPredicted, ClassIndexTruth)
    log.AddLogDict(
        "Accuracy",
        {
            "SampleNumTotal": NumTotal,
            "SampleNumCorrect": NumCorrect,
        }
    )

class SoftMax(DLUtils.module.AbstractModule):
    def __init__(self):
        Param = self.Param
        Param._CLASS = "DLUtils.module.Loss.SoftMax"
    def Receive(self, Input):
        # Input: [BatchNum, FeatureNum]
        return torch.softmax(Input, dim=1)

SoftMax1D = SoftMax

class SoftMaxAndCrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.module.Loss.SoftMaxAndCrossEntropy"
    
    def Init(self, **Dict):
        self.AddSubModule(
            "CrossEntropy", CrossEntropy()
        )
        self.AddSubModule(
            "SoftMax", SoftMax()
        )
        super().__init__()
        return self

class CrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        return
    def Receive(self, Input, Output):
        # Input: [BatchSize, OutuptNum, Probability]
        # Output: [BatchSize, OutputNum, Probability]
        return Output * torch.log(Input)