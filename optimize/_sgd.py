
from .GradientDescend import GradientDescend
import torch
# class Adam(GradientDescend):
#     def SetParam(self, **Dict):
#         Param = self.Param
#         LearningRate = Dict.setdefault("LearningRate", None)
#         Param.LearningRate = LearningRate

#         Alpha = Dict.get("Alpha") # momentum
#             # exponential average on historical step vector
#         Beta = Dict.get("Beta") # gradient element-wise nomalization
#             # exponential average on historical gradient absolute value
#         if Beta is not None:
#             assert Beta >= 0.0
#             if Beta > 0.0:
#                 Param.GradientNorm.Enable = True
#                 Param.GradientNorm.Value = Beta
#                 self.Beta = Beta
#             else:
#                 Param.GradientNorm = False
#                 Param.GradientNorm.delattrifexists("Value")
#         if Alpha is not None:
#             assert Alpha >= 0.0
#             if Alpha > 0.0:
#                 Param.Momentum.Enable = True
#                 Param.Momentum.Value = Alpha
#                 self.Alpha = Alpha
#             else:
#                 Param.Momentum.Enable = False
#                 Param.Momentum.delattrifexists("Value")
#         return self
#     def Enable(self, Type):
#         Param = self.Param
#         if Type in ["Momentum"]:
#             self.EnableMomentum()
#         else:
#             raise Exception()
#         return self
#     def EnableMomentum(self):
#         Param = self.Param
#         Param.Momentum.Enable = True
#         Param.Momentum.setdefault("Value", 0.0)
#         return self
#     def _UpdateParam(self, Dict):
#         self.Optimizer.zero_grad()
#         Dict.Evaluation["Loss"].backward()
#         Cache0 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
#         #Cache0 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
#         self.Optimizer.step()
#         Cache1 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
#         #Cache1 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
#         Diff = Cache1 - Cache0
#         #TrainParam = Dict.Model.ExtractTrainParam()
#         return self
#     def Init(self, IsSuper=False, IsRoot=True):
#         Param = self.Param
#         super().Init(IsSuper=True, IsRoot=IsRoot)
#         if Param.Momentum.Enable:
#             self.Alpha = Param.Momentum.Value
#         else:
#             self.Alpha = 0.0
#         if Param.GradientNorm.Enable:
#             self.Beta = Param.GradientNorm.Value
#         else:
#             self.Beta = 0.0
#         self.LearningRate = Param.LearningRate
#         self.ResetOptimizer()
#         self.Optimize = self._UpdateParam
#         return self
#     def ResetOptimizer(self, *List, **Dict):
#         """
#         !important
#         after binded model is saved and reloaded, optimizer should be reset with 
#         newly-loaded model trainable params.
#         """

#         Param = Dict.get("Param")
#         if Param is None:
#             #Param = list(self.TrainParam.values())
#             Param = list(self.Model.ExtractTrainParam().values())
#         else:
#             Param = list(Param.values())
#         self.Optimizer = torch.optim.Adam(
#             Param,
#             lr=self.LearningRate,
#             betas=[
#                 self.Alpha, # Momentum
#                 self.Beta   # GradientNorm
#             ]
#         )


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
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.LearningRate = Param.LearningRate
        return super().Init(IsSuper=True, IsRoot=IsRoot)