import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()

# def Probability2MostProbableIndex(Probability):
#     # Probability: [BatchSize, ClassNum]
#     #Max, MaxIndices = torch.max(Probability, dim=1)
#     #return DLUtils.TorchTensor2NpArray(MaxIndices) # [BatchSize]
#     return torch.argmax(Probability, axis=1)

# def LogAccuracyForSingleClassPrediction(ClassIndexPredicted, ClassIndexTruth, log):
#     #log = DLUtils.ParseLog(log)
#     ClassIndexPredicted = DLUtils.ToNpArray(ClassIndexPredicted)
#     ClassIndexTruth = DLUtils.ToNpArray(ClassIndexTruth)
#     NumCorrect, NumTotal = DLUtils.evaluate.CalculateAccuracyForSingelClassPrediction(ClassIndexPredicted, ClassIndexTruth)
#     log.LogDict(
#         "Accuracy",
#         {
#             "SampleNumTotal": NumTotal,
#             "SampleNumCorrect": NumCorrect,
#         }
#     )
