import torch
import DLUtils

class BatchNorm2D(DLUtils.module.TorchModuleWrapper):
    def __init__(self, *List, **Dict):
        self.ModuleParamMap = {
            "Affine.Weight": "weight",
            "Affine.Bias": "bias",
            "History.Mean": "running_mean",
            "History.Var": "running_var",
            "History.NumBatchesTracked": "num_batches_tracked"
        }
        super().__init__(*List, **Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in Dict.items():
            if Key in ["FeatureNum"]:
                Param.Feature.Num = Value
            else:
                raise Exception()
        return self
    def SetTrain(self, Recur=True):
        self.CallMethod = self.Receive = self.ReceiveTrain
        return super().SetTrain(Recur=Recur)
    def SetTest(self, Recur=True):
        self.CallMethod = self.Receive = self.ReceiveTest
        return super().SetTest(Recur=Recur)
    def ReceiveTrain(self, In):
        return self.module(In)
    def ReceiveTest(self, In):
        return self.module(In)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param

        if self.IsInit():
            Param.Affine.setdefault("Enable", True)
            Param.Momentum.setdefault("Enable", True)
            if Param.Momentum.Enable:
                Momentum = Param.Momentum.setdefault("Value", 0.1)
            else:
                Momentum = 0.0

        self.module = torch.nn.BatchNorm2d(
            num_features=Param.Feature.Num,
            affine=Param.Affine.Enable,
            track_running_stats=Param.Momentum.Enable,
            momentum=Momentum
            # var_apply = (1 - momentum) var_history + varmomentum * var_current
        )
        if self.IsInit():
            self.UpdateParamFromModule()
        else:
            self.UpdateModuleFromParam()
            
            if Param.Affine.Enable:
                StateDict = self.module.state_dict()
                self.SetTrainParam(Weight=DLUtils.ToNpArray(StateDict['weight']))
                self.SetTrainParam(Bias=DLUtils.ToNpArray(StateDict['bias']))
        return super().Init(IsSuper=True, IsRoot=IsRoot)