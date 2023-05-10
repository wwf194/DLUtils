import DLUtils
from .gradient_descend import GradientDescend
import torch

class SGD(GradientDescend):
    def _INIT(self, *List, **Dict):
        self.SetParam(*List, **Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in dict(Dict).items():
            if Key in ["Momentum"]:
                if isinstance(Value, float):
                    Param.Momentum.Value = Value
                elif isinstance(Value, bool):
                    Param.Momentum.Enable = Value
                else:
                    super().SetParam()
                Dict.pop("Momentum")
        super().SetParam(**Dict)
        return self
    def _UpdateParam(self, Dict):
        self.Optimizer.zero_grad()
        Dict.Evaluation["Loss"].backward()
        Cache0 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
        # Cache0 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
        self.Optimizer.step()
        Cache1 = Dict.Model.Encoder.MLP.L0.Weight.clone().detach().cpu()
        # Cache1 = Dict.Model.Decoder.L0.L0.Weight.clone().detach().cpu()
        Diff = Cache1 - Cache0
        #TrainParam = Dict.Model.ExtractTrainParam()
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # momentum setting
        Param.Momentum.setdefault("Enable", True)
        if not Param.Momentum.hasattr("Value"):
            Param.Momentum.Value = 0.9
        self.LearningRate = Param.LearningRate
        self.Nesterov = Param.Nesterov.setdefault("Enable", True)

        if Param.Momentum.Enable:
            self.Alpha = Param.Momentum.Value
        else:
            self.Alpha = 0.0

        if self.IsInit():
            Param.setdefault("WeightDecay", 0.0)

        self.WeightDecay = Param.WeightDecay

        self.Optimize = self._UpdateParam
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.ResetOptimizer()
        return self
    def ResetOptimizer(self, *List, **Dict):
        """
        !important
        after binded model is saved and reloaded, optimizer should be reset with 
        newly-loaded model trainable params.
        """
        TrainParam = Dict.get("TrainParam")
        if TrainParam is None:
            #Param = list(self.TrainParam.values())
            TrainParam = list(self.Model.ExtractTrainParam().values())
        else:
            TrainParam = list(TrainParam.values())

        if hasattr(self, "optimizer"):
            StateDict = self.optimizer.state_dict()
            self.optimizer = torch.optim.SGD(TrainParam)
            self.optimizer.load_state_dict(StateDict)
        else:
            self.Optimizer = torch.optim.SGD(
                TrainParam,
                lr=self.LearningRate,
                dampening=0.0,
                momentum=self.Alpha,
                weight_decay=self.WeightDecay,
                nesterov=self.Nesterov
            )
        return self