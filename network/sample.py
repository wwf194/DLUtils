import torch
import numpy as np

import DLUtils
class SampleFromNormalDistribution(DLUtils.module.AbstractNetwork):
    SetParamMap = DLUtils.IterableKeyToElement({
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
        Std = self.ProcessStd(Std)
        Epsilon = torch.FloatTensor(Std.size()).normal_()
        Epsilon = Epsilon.to(self.Device)
        return Epsilon.mul(Std).add_(Mean)
    def _ProcessLogOfVariance(self, LogOfVariance):
        Std = LogOfVariance.mul(0.5).exp_()
        return Std
    def _ProcessStd(self, Std):
        return Std
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Receive.setdefault("Format", "CatMeanStd")
        Param.In.setdefault("Type", "LogOfVariance")
        if Param.In.Type in ["LogOfVariance", "LogVar"]:
            self.ProcessStd = self._ProcessLogOfVariance
        elif Param.In.Type in ["Std"]:
            self.ProcessStd = self._ProcessStd
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
    