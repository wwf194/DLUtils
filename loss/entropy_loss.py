import torch
import torch.nn.functional as F
import DLUtils

class CrossEntropyNClass(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.train.loss.CrossEntropy"
        return
    def Receive(self, Pred, Target):
        # Input:  [BatchSize, OutNum, Probability]
        # Output: [BatchSize, OutNum, Probability]
        Loss = - Target * torch.log(Pred) # [BatchSize, OutNum]
        Loss = torch.sum(Loss, dim=1) # [BatchSize]
        return self.AfterOperation(Loss)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Operation.setdefault("After", "Mean")
        if Param.Operation.After in ["Mean"]:
            self.AfterOperation = lambda x:torch.mean(x)
        else:
            self.AfterOperation = lambda x:x
        return super().Init(IsSuper, IsRoot)
CrossEntropy = CrossEntropyNClass

class SoftMax1D(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.module.loss.SoftMax"
    def Receive(self, Logits):
        # Logits: [BatchSize, FeatureNum]
        return torch.softmax(Logits, dim=1)
SoftMax = SoftMax1D

class SoftMaxAndCrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.loss.SoftMaxAndCrossEntropy"
    def Receive(self, Out=None, OutTarget=None):
        # Out: logits.
        OutProb = self.SoftMax(Out)
        Loss = self.CrossEntropy(OutProb, OutTarget)
        return Loss
    def Init(self, IsSuper=False, IsRoot=True):
        self.AddSubModule(
            "CrossEntropy", CrossEntropy()
        )
        self.AddSubModule(
            "SoftMax", SoftMax()
        )
        super().Init(IsSuper=IsSuper, IsRoot=IsRoot)
        return self

class CrossEntropy2Class(DLUtils.module.AbstractNetwork):
    SetParamMap = {
        ("AfterOperation"): "Operation.After"
    }
    def __init__(self, *List, **Dict):
        super().__init__( *List, **Dict)
    def Receive(self, Out, OutTarget):
        """
        calculate 2-class cross entropy at all positions in Pred and Target.
        Pred: value at each point represents predicted probability of being class 0 here. range: (0.0, 1.0]
        Target: value at each point represents truth probability of being class 0 here. range: [0.0, 1.0]
        """
        Loss = F.binary_cross_entropy(Out, OutTarget, reduction="none")
        Loss = torch.sum(Loss, dim=(1, 2))
        Loss = self.AfterOperation(Loss)
        return Loss
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # pytorch default: mean 
        AfterOperation = Param.Operation.setdefault("After", "Mean")
        if AfterOperation in ["Mean"]:
            self.reduce = "mean"
            self.AfterOperation = lambda x: torch.mean(x)
        elif AfterOperation in ["Sum"]:
            self.reduce = "sum"
            self.AfterOperation = lambda x: torch.sum(x)
        elif AfterOperation in ["None"]:
            self.reduce = "none"
            self.AfterOperation = lambda x: x
        else:
            raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)
CrossEntropy2Class3D = CrossEntropy2Class

class KLNormMuSigmaAndNorm01(DLUtils.module.AbstractNetwork):
    """
    KLDivergence between multivariate Gaussian: N(mu; I * sigma ** 2) and N(0; I)
    Elemements/Dimensions in both Gaussian distribution are independent.
    """
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
    # def _ReceiveMean(self, Mu, LogOfVar):
    #     # Mu: [BatchSize, FeatureNum]
    #     # Sigma2: [BatchSize, FeatureNum]. log of square of sigma.
    #     # 0.5 * (1.0 + log(sigma ** 2) - mu ** 2 - sigma ** 2)
    #     Var = torch.exp(LogOfVar) # Std ** 2. in gaussian, std often written as sigma.
    #     KLDivergence = - 0.5 * torch.mean(1.0 + LogOfVar - Mu ** 2 - Var)
    #     return KLDivergence
    def Receive(self, Mu, LogOfVar):
        Var = torch.exp(LogOfVar) # Std ** 2. in gaussian, std often written as sigma.
        KLDivergence = - 0.5 * torch.sum(1.0 + LogOfVar - Mu ** 2 - Var, dim=1)
        KLDivergence = self.AfterOperation(KLDivergence)
        return KLDivergence
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        AfterOperation = Param.Operation.setdefault("After", "Mean")
        if AfterOperation in ["Mean"]:
            #self.Receive = self._ReceiveMean
            self.AfterOperation = lambda x: torch.mean(x)
        elif AfterOperation in ["Sum", "sum"]:
            #self.Receive = self._ReceiveSum
            self.AfterOperation = lambda x: torch.sum(x)
        else:
            raise Exception()
        return super().Init(IsSuper, IsRoot)

KLNormAndNorm01 = KLNormMuSigmaAndNorm01