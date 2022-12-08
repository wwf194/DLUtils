
from .GradientDescend import GradientDescendOptimizer

def Optimizer(Type):
    if Type in ["GradientDescend"]:
        return GradientDescendOptimizer(Type)
    else:
        raise Exception()