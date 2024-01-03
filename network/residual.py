import DLUtils
class ResidualLayer(DLUtils.module.AbstractNetwork):
    def __init__(self, SubModule=None):
        super().__init__()
        if SubModule is not None:
            self.AddSubModule("SubModule", SubModule)
    def Receive(self, X):
        # X: [BatchSize, FeatureNum]
        Y = self.SubModule(X)
        Output = X + Y
        return Output
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        return super().Init(IsSuper=True, IsRoot=IsRoot)