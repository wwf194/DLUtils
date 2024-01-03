
import torch
import torch.nn as nn
import DLUtils

import functools
# https://github.com/bao18/open_earth_map.git
class PixelClassificationLoss(DLUtils.AbstractModule):
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
    def _Receive(self, Predict, Target):
        """
        In: [BatchSize(ImageNum), ClassNum, ImageHeight, ImageWeight].
        Probability a pixel of image i at height j and width k belongs Predidct[i, c, j, k]
        Target: [BatchSize(ImageNum), ClassNum, ImageHeight, ImageWeight]
        """
        # In = torch.softmax(In, dim=1)
        ClassNum = Predict.Shape[1]
        Loss = 0.0
        for i in range(self.ClassIndexStart, ClassNum):
            ClassProbPredict = Predict[:, i, :, :]
            ClassProbTruth = Target[:, i, :, :]
            Loss += 1 - self.IoU(ClassProbPredict, ClassProbTruth)
        return Loss
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.ExcludeBackgroundClass.setdefault("Enable", True)
        if Param.ExcludeBackgroundClass.Enable:
            Param.ExcludeBackgroundClass.setdefault("Index", 0)
            if Param.ExcludeBackgroundClass.Index == 0:
                self.ClassIndexStart = 1
                self.Receive = self._Receive
            else:
                raise Exception()
        else:
            self.ClassIndexStart = 0
            self.Receive = self._Receive
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class JaccardLoss(PixelClassificationLoss):
    def Build(self, IsSuper=False, IsRoot=True):
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.EvaluateFunction = IoU
        return self

class DiceLoss(PixelClassificationLoss):
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        super().Init(IsSuper=True, IsRoot=IsRoot)
        Param.FScore.setdefault("Beta", 1.0)
        self.EvaluateFunction = functools.partial(
            FScore, Beta=Param.FScore.Beta)
        return self

def FScore(ProbPredict, ProbTruth, Beta=1.0, Eps=1.0e-7, ProbPredictMin=None):
    """Calculate F-score between ground truth and prediction
    Args:
        ProbPredict: [BatchSize, ImageHeight, ImageWeight].
            Predicted probability of pixels belonging to a class. 
        ProbTarget:  [BatchSize, ImageHeight, ImageWeight].
            Truth probability of pixels belonging to a class. 
        Beta (float): Importance of recall over precision.
            Larger Beta makes reduction of FN more important than FP.
        Eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    If ProbTruth is hard(either 0.0 or 1.0), for every single pixel:
        If ProbTruth == 0.0, mearning current pixel does not belong to current class:
            TP == 0.0, FN == 0.0. FP == ProbPredict.
        If ProbTruth == 1.0,  mearning current pixel does belongs to current class:
            TP == ProbPredict, FN == 1.0 - ProbPredict. FP = 0.0.
        Both conditions satisfies:
            TP == ProbTruth * ProbPredict,
            FN == ProbTruth - ProbPredict,
            FP == ProbPredict - TP.
    ? If ProbTruth is soft(between 0.0 and 1.0) ?
    """
    ProbPredict = _threshold(ProbPredict, threshold=ProbPredictMin)
    TP = torch.sum(ProbTruth * ProbPredict)
    FP = torch.sum(ProbPredict) - TP
    FN = torch.sum(ProbTruth) - TP
    _FScore = ((1 + Beta**2) * TP + Eps) / ((1 + Beta**2) * TP + Beta**2 * FN + FP + Eps)
    return _FScore

def IoU(ProbPredict, ProbTruth, Eps=1e-7, ProbPredictMin=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        ProbPredict: [BatchSize, ImageHeight, ImageWeight].
            Predicted probability of pixels belonging to current class. 
        ProbTarget:  [BatchSize, ImageHeight, ImageWeight].
            Truth probability of pixels belonging to current class. 
        Beta (float): Importance of recall over precision.
            Larger Beta makes reduction of FN more important than FP.
        Eps (float): epsilon to avoid zero division
    Returns:
        float: IoU(Jaccard) score
    """
    ProbPredict = _threshold(ProbPredict, threshold=ProbPredictMin)
    I = torch.sum(ProbTruth * ProbPredict) # Intersection
    U = torch.sum(ProbTruth) + torch.sum(ProbPredict) - I + Eps # Union
    return (I + Eps) / U

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

# -----------------
# --- FocalLoss ---
# -----------------
class FocalLoss(DLUtils.AbstractModule):
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", None]
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.name = "Focal"
    def forward(self, Predict, Truth):
        """
        Predict: [BatchSize, ClassNum, ImageHeight, ImageWidth]
        Truth: [BatchSize, ClassNum, ImageHeight, ImageWidth]
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            Predict, Truth.float(), reduction="none"
        ) # [BatchSize, ClassNum, ImageHeight, ImageWidth]
        # binary_loss = 
        #   - truth * log(sigmoid(predict))
        #   - (1.0 - truth) * log((1 - sigmoid(predict)))
        pt = torch.exp(-bce_loss) # [BatchSize, ClassNum, ImageHeight, ImageWidth]
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        focal_loss = self.LossReduction(focal_loss)
        return focal_loss # single num
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Loss.setdefault("ReductionMethod", "Mean")
        if Param.Loss.ReductionMethod in ["Mean"]:
            self.LossReduction = torch.mean
        elif Param.Loss.ReductionMethod in ["Sum"]:
            self.LossReduction = torch.sum
        else:
            raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)