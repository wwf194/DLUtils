
from .GradientDescend import GradientDescend
import torch
class Adam(GradientDescend):
    def SetParam(self, **Dict):
        Param = self.Param
        LearningRate = Dict.setdefault("LearningRate", None)
        Param.LearningRate = LearningRate

        Alpha = Dict.get("Alpha") # momentum
            # exponential average on historical step vector
        Beta = Dict.get("Beta") # gradient element-wise nomalization
            # exponential average on historical gradient absolute value
        if Beta is not None:
            assert Beta >= 0.0
            if Beta > 0.0:
                Param.GradientNorm.Enable = True
                Param.GradientNorm.Value = Beta
                self.Beta = Beta
            else:
                Param.GradientNorm = False
                Param.GradientNorm.delattrifexists("Value")
        if Alpha is not None:
            assert Alpha >= 0.0
            if Alpha > 0.0:
                Param.Momentum.Enable = True
                Param.Momentum.Value = Alpha
                self.Alpha = Alpha
            else:
                Param.Momentum.Enable = False
                Param.Momentum.delattrifexists("Value")
        return self
    def Enable(self, Type):
        Param = self.Param
        if Type in ["Momentum"]:
            self.EnableMomentum()
        else:
            raise Exception()
        return self
    def EnableMomentum(self):
        Param = self.Param
        Param.Momentum.Enable = True
        Param.Momentum.setdefault("Value", 0.0)
        return self
    def _UpdateParam(self, Dict):
        self.Optimizer.zero_grad()
        Dict.Evaluator.Loss.backward()
        # print("%.3e"%(self.Model.SubModules["2"].Weight.grad[0][0].item()))
        self.Optimizer.step()
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        super().Init(IsSuper=True, IsRoot=IsRoot)
        if Param.Momentum.Enable:
            self.Alpha = Param.Momentum.Value
        else:
            self.Alpha = 0.0
        if Param.GradientNorm.Enable:
            self.Beta = Param.GradientNorm.Value
        else:
            self.Beta = 0.0
        self.LearningRate = Param.LearningRate
        self.ResetOptimizer()
        self.Optimize = self._UpdateParam
        return self
    def ResetOptimizer(self):
        self.Optimizer = torch.optim.Adam(
            self.TrainParam.values(),
            lr=self.LearningRate,
            betas=[
                self.Alpha, # Momentum
                self.Beta   # GradientNorm
            ]
        )