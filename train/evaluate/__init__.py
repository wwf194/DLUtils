import DLUtils

def Evaluator(Type):
    if Type in ["ImageClassification"]:
        return XFixedSizeYFixedSizeProb()
    else:
        raise Exception()

class _Evaluator:
    def __init__(self):
        self.Param = DLUtils.Param()

class XFixedSizeYFixedSizeProb:
    def BeforeTrain(self):
        Model = self.Model
        self.TrainableParam = Model.ExtractTrainableParam()
    def Optimize(self, Input=None, OutputTarget=None, Model=None, Evaluation=None):
        Model.ClearGrad()
        Evaluation.Loss.backward()
        Output = Model(Input)
        Loss = self.LossModule(Output, OutputTarget)
        self.optimizer.UpdateParam()
        return self
    def SetLoss(self, LossModule, *List, **Dict):
        if isinstance(LossModule, str):
            LossModule = DLUtils.Loss(LossModule, *List, **Dict)
        self.LossModule = LossModule
        return self