import DLUtils
import torch.nn.functional as F
class MSELoss(DLUtils.module.AbstractNetwork):
    def __init__(self, *List, **Dict):
        super().__init__( *List, **Dict)
    def Receive(self, Out, OutTarget):
        Loss = F.mse_loss(Out, OutTarget, reduction=self.reduce)
        return Loss
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # pytorch default: mean 
        AfterOperation = Param.Operation.setdefault("After", "Mean")
        if AfterOperation in ["Mean"]:
            self.reduce = "mean"
        elif AfterOperation in ["Sum"]:
            self.reduce = "sum"
        elif AfterOperation in ["None"]:
            self.reduce = "none"
        else:
            raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)