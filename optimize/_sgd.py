
from .gradient_descend import GradientDescend
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
class _SGD(GradientDescend):
    def _Updatemodule(self):
        # for p in group['params']:
        #     if p.grad is None:
        #         continue
        #     d_p = p.grad
        #     # p = p - d_p * lr
        #     p.add_(d_p, alpha=-group['lr'])
        for Param in self.ParamList:
            if Param.Grad is None:
                continue
            Grad = Param.Grad
            # Param -= self.LR * Param._grad
            Param.add_(Grad, alpha=-self.LR)
    def Enable(self, Type):
        Param = self.Param
        if Type in ["Momentum"]:
            Param.Momentum.Enable = True
            Param.Momentum.Value = 0.0
        else:
            raise Exception()
        return self
    def SetParam(self, **Dict):
        Param = self.Param
        Momentum = Dict.setdefault("Momentum", False)
        if isinstance(Momentum, float):
            Param.Momentum.Enable = True
            Param.Momentum.Value  = Momentum
        elif Momentum is True:
            Param.Momentum.Enable = True
            Param.Momentum.Value  = 0.1
        else:
            Param.Momentum.Enable = False
            Param.Momentum.delattr("Value")

        Nesterov = Dict.setdefault("Nesterov", False)
        Param.Nesterov.Enable = Nesterov

        EnableMomentum = Param.Momentum.Enable
        if EnableMomentum:
            if Nesterov:
                self.Optimize = self._UpdateParamMomentumNesterov
        else:
            if Nesterov:
                #self.Optimize = self._UpdateParamNesterov
                raise Exception("Nesterov can only be enabled when momentum is on.")
            else:
                self.Optimize = self._UpdateParam
        self.Nesterov = Nesterov
        return self
    # def ResetOptimizer(self, *List, **Dict):
    #     self.Optimizer = torch.optim.sgd(
    #         #self.TrainParam.values(),
    #         Dict["Param"],
    #         lr=self.LearningRate,
    #         nesterov=self.Nesterov,
    #     )
    def _UpdateParamMomentumNesterov(self, *List, **Dict):
        for Param in self.TrainParam:
            alpha = self.alpha # momentum coefficient
            # applying nesterov usually requires damp = 0.0
            LearningRate = self.LearningRate
            if Param.grad is None:
                continue
            Grad = Param.grad
            H = self.MomentumDict[Param]
            H.mul_(alpha).add_(Grad, alpha=1.0)
            # Nesterov. GradWithMomentum = Grad + alpha * H
            GradWithMomentum = Grad.add(H, alpha=alpha)
            Param.add_(GradWithMomentum, alpha=-LearningRate)
        return self
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.LearningRate = Param.LearningRate
        return super().Init(IsSuper=True, IsRoot=IsRoot)