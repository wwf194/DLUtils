import DLUtils
class GlobalOptimizer(DLUtils.module.AbstractModule):
    def AddModule():
        return
    def Optimize(self):
        for Param in self.TrainParams:
            Grad = ExtractGradient(Param)
            Param -= self.LearningRate * Grad
        return
    
