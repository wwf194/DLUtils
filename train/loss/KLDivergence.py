import torch
import numpy as np
import DLUtils
class KLNormAndNorm01(DLUtils.module.AbstractNetwork):
    """
    KLDivergence between multivariate Gaussian: N(mu; I * sigma ** 2) and N(0; I)
    Elemements/Dimensions in both Gaussian distribution are independent.
    """
    def __init__(self):
        super().__init__()
    def Receive(self, Mu, Sigma2):
        # Mu: [BatchSize, FeatureNum]
        # Sigma2: [BatchSize, FeatureNum]. Square of sigma.
        # 0.5 * (1.0 + log(sigma ** 2) - mu ** 2 - sigma ** 2)
        KLDivergence = 0.5 * torch.sum(1.0 + torch.log_(Sigma2) - Mu ** 2 - Sigma2)
        return KLDivergence
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("SubModule")
        return super().Init(IsSuper, IsRoot)