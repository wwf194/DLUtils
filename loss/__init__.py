import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")

from .entropy_loss import \
    SoftMax, \
    SoftMaxAndCrossEntropy, \
    CrossEntropy2Class, CrossEntropyNClass, CrossEntropy

from .mse_loss import MSELoss

from .entropy_loss import KLNormAndNorm01, KLNormMuSigmaAndNorm01

def Loss(Type):
    if Type in ["CrossEntropy"]:
        return CrossEntropy()
    elif Type in ["SoftMaxAndCrossEntropy"]:
        return SoftMaxAndCrossEntropy()
    else:
        raise Exception(Type)

def CrossEntropyLossForTargetProbability(Output, ProbabilityTarget, Method='Average'):
    # Output: [SampleNum, OutNum]
    # ProbabilitiesTarget: [SampleNum, OutNum], must be positive, and sum to 1 on axis 1.
    LogProbabilities = -F.log_softmax(Output, dim=1) # [SampleNum, OutNum]
    BatchSize = Out.shape[0]
    CrossEntropy = torch.sum(LogProbabilities * ProbabilityTarget, axis=1) # [SampleNum]
    if Method == 'Average':
        CrossEntropy = torch.mean(CrossEntropy)
    elif Method in ["Sum"]:
        CrossEntropy = torch.sum(CrossEntropy)
    else:
        raise Exception(Method)
    return CrossEntropy

from .PixelClassification import JaccardLoss