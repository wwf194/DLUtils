import torch
import numpy as np

import DLUtils
class SampleFromNormalDistribution(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        "ReceiveFormat": "Receive.Format",
        ("InType", "InputType"): "In.Type",
    })
    # def __init__(self, *List, **Dict):
    #     super().__init__(*List, **Dict)
    def _ReceiveSplitMeanStd(self, Mean, Std):
        # X: [BatchSize, FeatureNum]
        # Std = LogVar.mul(0.5).exp_()
        """
        reparameterization trick
        In some cases, Log of Variance is provided.
        """
        return self.Sample(Mean, Std)
    def _ReceiveCatMeanStdDynamic(self, MeanStd):
        UnitNum = MeanStd.size(1) // 2
        Mean = MeanStd[:, :UnitNum]
        Std = MeanStd[:, UnitNum:]
        return self.Sample(Mean, Std)
    def _ReceiveCatMeanStd(self, MeanStd):
        Mean = MeanStd[:, :self.InNum]
        Std = MeanStd[:, self.InNum:]
        return self.Sample(Mean, Std)
    def Sample(self, Mean, Std):
        Std = self.GetStd(Std)
        Epsilon = torch.FloatTensor(Std.size()).normal_()
        Epsilon = Epsilon.to(self.Device)
        # return Epsilon.mul(Std).add_(Mean).detach()
        return Epsilon * Std + Mean
    def _GetStdFromLogOfVar(self, LogOfVar):
        # Std = LogOfVar.mul(0.5).exp_()
        # Std = LogOfVar.detach().mul(0.5).exp_()
        # LogOfVar = Log ^ (Std ^ 2) = 2.0 * Log (Std)
        # Std = exp ^ (LogOfVar * 0.5)
        # _LogOfVar = LogOfVar.detach() # this will block backward gradient.s
        _LogOfVar = LogOfVar
        _LogOfStd = _LogOfVar * 0.5
        _Std = torch.exp(_LogOfStd)
        return _Std
    def _GetStd(self, Std):
        return Std
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Receive.setdefault("Format", "SplitMeanStd")
        Param.In.setdefault("Type", "LogOfVar")

        if Param.In.Type in ["LogOfVariance", "LogVar", "LogOfVar"]:
            Param.In.Type = "LogOfVar"
            self.GetStd = self._GetStdFromLogOfVar
        elif Param.In.Type in ["Std"]:
            self.GetStd = self._GetStd
        else:
            raise Exception()

        if Param.Receive.Format in ["SplitMeanStd"]:
            self.Receive = self._ReceiveSplitMeanStd
        elif Param.Receive.Format in ["CatMeanStd"]:
            if Param.In.hasattr("Num"):
                self.Receive = self._ReceiveCatMeanStd
                assert isinstance(Param.In.Num, int)
                self.InNum = Param.In.Num
            else:
                self.Receive = self._ReceiveCatMeanStdDynamic
        else:
            raise Exception()
        
        return super().Init(IsSuper, IsRoot)
    