
from .GradientDescend import GradientDescendOptimizer, GradientDescend
from .adam import Adam
from .sgd import SGD
import DLUtils

OptimizeAlgorithmMap = DLUtils.IterableKeyToElement({
    ("GradientDescend"): GradientDescend,
    ("SGD"): SGD,
    ("Adam", "adam"): Adam
})

class Optimizer:
    def __init__(self, Type=None):
        self.Type = Type
        if Type is not None:
            return OptimizeAlgorithmMap[Type]
    def GradientDescend(self, Type=None):


        return GradientDescend(Type)
# def Optimizer(Type):
#     if Type in ["GradientDescend"]:
#         return GradientDescendOptimizer(Type)
#     else:
#         raise Exception()