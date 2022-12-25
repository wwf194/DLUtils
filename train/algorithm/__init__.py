
from .GradientDescend import GradientDescendOptimizer
from .Adam import Adam
from .SGD import SGD

def Optimizer(Type):
    if Type in ["GradientDescend"]:
        return GradientDescendOptimizer(Type)
    else:
        raise Exception()