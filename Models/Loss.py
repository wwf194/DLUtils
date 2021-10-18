import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

def GetLossMethod(param, **kw):
    if param.Type in ["MeanSquareError", "MSE"]:
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.mse_loss
        else:
            return lambda x, xTarget: Coefficient * F.mse_loss(x, xTarget)
    elif param.Type in ["CrossEntropyLossForSingleClassPrediction", "CrossEntropyLossForLabels"]:
        # By convention, softmax is included in the loss function.
        # Hard labels. Input: [SampleNum, ClassNum]. Target: Labels in shape of [SampleNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.cross_entropy
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * F.cross_entropy(Output, ProbabilitiesTarget)
    elif param.Type in ["CrossEntropyLoss", "CEL"]:
        # By convention, softmax is included in the loss function.
        # Soft labels. Input: [SampleNum, ClassNum]. Target: Probabilities in shape of [SampleNum, ClassNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return CrossEntropyLossForTargetProbabilities
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * CrossEntropyLossForTargetProbabilities(Output, ProbabilitiesTarget)
    else:
        raise Exception(param.Type)

def CrossEntropyLossForTargetProbabilities(Output, ProbabilitiesTarget, Method='Average'):
    # Output: [SampleNum, OutputNum]
    # ProbabilitiesTarget: [SampleNum, OutputNum], must be positive and sum to 1 on axis 1.
    LogProbabilities = -F.log_softmax(Output, dim=1) # [SampleNum, OutputNum]
    BatchSize = Output.shape[0]
    CrossEntropy = torch.sum(LogProbabilities * ProbabilitiesTarget, axis=1) # [SampleNum]
    if Method == 'Average':
        CrossEntropy = torch.mean(CrossEntropy)
    elif Method in ["Sum"]:
        CrossEntropy = torch.sum(CrossEntropy)
    else:
        raise Exception(Method)
    return CrossEntropy