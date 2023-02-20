
from .GradientDescend import GradientDescend
import torch
import DLUtils
class Adam(GradientDescend):
    def __init__(self):
        super().__init__()
        SetParamMap = super().SetParamMap.update(DLUtils.IterableKeyToElement({
            # ("Beta"): "GradientNorm.Enable"
        }))
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in dict(Dict).items():
            if Key in ["Alpha", "Momentum"]: # momentum
                # exponential average on historical step vector
                continue # will be handled in super().SetParam
            elif Key in ["Beta", "GradientNorm"]:
                # gradient element-wise nomalization
                # exponential average on historical gradient absolute value
                assert Value >= 0.0
                if isinstance(Value, float):
                    assert Value >= 0.0
                    if Value > 0.0:
                        Param.GradientNorm.Enable = True
                        Param.GradientNorm.Value = Value
                        self.Beta = Value
                    else:
                        Param.GradientNorm.Enable = False
                        Param.GradientNorm.delattrifexists("Value")
                elif isinstance(Value, bool):
                    Param.GradientNorm.Enable = Value
                else:
                    raise Exception()
                Dict.pop(Key)
        return super().SetParam(**Dict)
    # def Enable(self, Type):
    #     Param = self.Param
    #     if Type in ["Momentum"]:
    #         self.EnableMomentum()
    #     else:
    #         raise Exception()
    #     return self
    # def EnableMomentum(self):
    #     Param = self.Param
    #     Param.Momentum.Enable = True
    #     Param.Momentum.setdefault("Value", 0.0)
    #     return self
    def _UpdateParam(self, Dict):
        self.optimizer.zero_grad()
        Dict.Evaluation["Loss"].backward()
        # Cache0 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
        # Cache0 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
        self.optimizer.step()
        # Cache1 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
        # Cache1 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
        # Diff = Cache1 - Cache0
        #TrainParam = Dict.Model.ExtractTrainParam()
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
    def ResetOptimizer(self, *List, **Dict):
        """
        !important
        after binded model is saved and reloaded, optimizer should be reset with 
        newly-loaded model trainable params.
        """
        Param = Dict.get("Param")
        if Param is None:
            #Param = list(self.TrainParam.values())
            TrainParam = list(self.Model.ExtractTrainParam().values())
        else:
            TrainParam = list(Param.values())
        


        if hasattr(self, "optimizer"):
            StateDict = self.optimizer.state_dict()
            self.optimizer = torch.optim.Adam(TrainParam)
            self.optimizer.load_state_dict(StateDict)
        else:
            self.optimizer = torch.optim.Adam(
                TrainParam,
                lr=self.LearningRate,
                betas=[
                    self.Alpha, # Momentum
                    self.Beta   # GradientNorm
                ]
            )
        return self