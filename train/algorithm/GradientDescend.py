import DLUtils
import torch
def Evaluator(Type):
    if Type in ["ImageClassification"]:
        return XFixedSizeYFixedSizeProb()
    else:
        raise Exception()


def GradientDescendOptimizer(Type, **Dict):
    assert Type in ["GradientDescend"]
    SubType = Dict.get("SubType")
    return GradientDescend(SubType=SubType)

class GradientDescend():
    def __init__(self, SubType=None):
        self.Param = DLUtils.Param()
        if SubType is not None:
            self.SetSubType(SubType)
    def SetSubType(self, SubType, **Dict):
        if SubType in ["SGD", "sgd"]:
            self.__class__ = SGD #
        elif SubType in ["Adam", "adam"]:
            self.__class__ = Adam
        else:
            raise Exception()
        return self
    
class SGD(GradientDescend):
    def _UpdateParam(self):
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
                self.UpdateParam = self._UpdateParamMomentumNesterov
        else:
            if Nesterov:
                self.UpdateParam = self._UpdateParamNesterov
            else:
                self.UpdateParam = self._UpdateParam
        self.Nesterov = Nesterov
        return self
    def BeforeBatchNesterov(self):
        
        return
    def _UpdateParamMomentum(self):
        # for p in group['params']:
        #     if p.grad is None:
        #         continue
        #     d_p = p.grad

        #     if momentum != 0:
        #         param_state = self.state[p]
        #             # 如果没有momentum_buffer，就初始化一个
        #             if 'momentum_buffer' not in param_state:
        #                 # buf = d_p
        #                 buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
        #             else:
        #                 # buf = buf * momentum + d_p
        #                 buf = param_state['momentum_buffer']
        #                 buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
        #             if nesterov:
        #                 # d_p = d_p + buf * momentum
        #                 d_p = d_p.add(buf, alpha=momentum)
        #             else:
        #                 d_p = buf
                    
        #     # p = p - d_p * lr = p - (d_p + (buf * momentum + d_p) * momentum) * lr
        #     p.add_(d_p, alpha=-group['lr'])
        for Param in self.TrainParam:
            alpha = self.alpha # momentum coefficient
            damp = self.damp
            LearningRate = self.LearningRate
            Nesterov = self.Nesterov
            if Param.grad is None:
                continue
            Grad = Param.grad
            H = self.MomentumDict[Param]
            # if H is None:
            #     # H = alpha * H + Grad
            #     H.mul_(alpha).add_(Grad, alpha=1.0 - damp)
            # else:
            #     # H = Grad
            #     H = Grad.clone().detach()

            # to avoid branch
            H.mul_(alpha).add_(Grad, alpha=1.0 - damp)

            if Nesterov:
                # GradWithMomentum = Grad + alpha * H
                GradWithMomentum = Grad.add(H, alpha=alpha)
            else:
                GradWithMomentum = H
            Param.add_(GradWithMomentum, alpha=-LearningRate)
        return self
    def _UpdateParamMomentumNesterov(self):
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

class Adam(GradientDescend):
    def SetParam(self, **Dict):
        Param = self.Param
        Alpha = Dict.setdefault("Alpha", 0.0) # momentum
            # exponential average on historical step vector
        Beta = Dict.setdefault("Beta", 0.0) # gradient element-wise nomalization
            # exponential average on historical gradient absolute value
        assert Alpha >= 0.0
        assert Beta >= 0.0
        if Beta > 0.0:
            Param.GradientNorm.Enable = True
            Param.GradientNorm.Value = Beta
            self.Beta = Beta
        else:
            Param.GradientNorm = False
            Param.GradientNorm.delattrifexists("Value")
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
            Param.Momentum.Enable = True
            Param.Momentum.Value = 0.0
        else:
            raise Exception()
        return self
    def _UpdateParam():
        return self

class XFixedSizeYFixedSizeProb:
    def BeforeTrain(self):
        Model = self.Model
        TrainableParam = Model.ExtractTrainableParam()
    def Optimize(self, Input=None, OutputTarget=None, Model=None, Evaluation=None):
        Model.ClearGrad()
        Evaluation.Loss.backward()
        self.optimizer.UpdateParam()
        Model