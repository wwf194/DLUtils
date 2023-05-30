import torch
import DLUtils

class BatchNorm2D(DLUtils.module.TorchModuleWrapper):
    ModuleParamMap = DLUtils.ExpandIterableKey({
        ("InNum", "FeatureNum"): "In.Num",
        "Affine.Weight": "weight",
        "Affine.Bias": "bias",
        "History.Mean": "running_mean",
        "History.Var": "running_var",
        "History.NumBatchesTracked": "num_batches_tracked"
    })
    ParamMap = DLUtils.ExpandIterableKey({
        ("FeatureNum"): "Feature.Num"
    })
    def Receive(self, In):
        """
        In: (BatchSize, ChannelNum, Height, Width)
        Out: (BatchSize, Channelnum, Height, Width)
        for each channel, there is an alpha and a beta.
            for every x in this channel, x = (x - mean(x)) / sqrt(var(x) + epsilon) * alpha + beta
                mean(x) is mean of all x in same channel, with different index on batch, height, and width dimension.
        alpha: (FeatureNum)
        beta: (FeatureNum)
            alpha and beta can be set to trainable
        """
        return self.module(In)
    # def SetTrain(self, Recur=True):
    #     self.CallMethod = self.Receive = self.ReceiveTrain
    #     return super().SetTrain(Recur=Recur)
    # def SetTest(self, Recur=True):
    #     self.CallMethod = self.Receive = self.ReceiveTest
    #     return super().SetTest(Recur=Recur)
    # def ReceiveTrain(self, In):
    #     return self.module(In)
    # def ReceiveTest(self, In):
    #     return self.module(In)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.In.hasattr("Num")
        if self.IsInit():
            Param.Affine.setdefault("Enable", True)
            Param.Momentum.setdefault("Enable", True)
            if Param.Momentum.Enable:
                Momentum = Param.Momentum.setdefault("Value", 0.1)
            else:
                Momentum = 0.0

            self.module = torch.nn.BatchNorm2d(
                num_features=Param.In.Num,
                affine=Param.Affine.Enable,
                track_running_stats=Param.Momentum.Enable,
                momentum=Momentum
                # var_apply = (1 - momentum) var_history + varmomentum * var_current
            )
            # self.UpdateParamFromModule()
        else:
            # self.UpdateModuleFromParam()
            self.module = torch.nn.BatchNorm2d(
                num_features=Param.In.Num
            )
            self.module.load_state_dict(Param.Module.Data)
            
        # if Param.Affine.Enable:
        #     StateDict = self.module.state_dict()
        #     self.SetTrainParam(Weight=DLUtils.ToNpArray(StateDict['weight']))
        #     self.SetTrainParam(Bias=DLUtils.ToNpArray(StateDict['bias']))
        return super().Init(IsSuper=True, IsRoot=IsRoot)