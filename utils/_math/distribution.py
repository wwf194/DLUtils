import DLUtils
import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
# else:
#     import numpy as np

def SampleFromKaimingUniform(Shape, NonLinear="ReLU", **Dict):
    """
    Y = f(WX). Keep variance of forward signal or backward gradient.
    """
    assert len(Shape) == 2
    InNum = Shape[0]
    OutNum = Shape[1]
    if NonLinear in ["ReLU", "relu"]:
        Priority = Dict.setdefault("Priority", "forward")
        if Priority in ["forward", "Forward"]:
            Max = math.sqrt(6.0 / InNum)
        elif Priority in ["backward", "Backward"]:
            Max = math.sqrt(6.0 / OutNum)
        elif Priority in ["all"," ForwardAndBackward"]:
            Average = 2.0 / (InNum + OutNum)
            Max = math.sqrt(6.0 / Average)
        else:
            raise Exception()
        Min = - Max
        return SampleFromUniformDistribution(Shape, Min, Max)
    elif NonLinear in ["None", "none"] or NonLinear is None:
        return SampleFromXaiverUniform(Shape, **Dict)
    else:
        raise Exception()

def SampleFromKaimingNormal(Shape, NonLinearFunction="ReLU", **Dict):
    # Y = f(WX). Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InNum = Shape[0]
    OutNum = Shape[1]
    if NonLinearFunction in ["ReLU", "relu"]:
        Priority = Dict.setdefault("Priority", "all")
        if Priority in ["forward", "Forward"]:
            Std = math.sqrt(2.0 / InNum)
        elif Priority in ["backward", "Backward"]:
            Std = math.sqrt(2.0 / OutNum)
        elif Priority in ["all","ForwardAndBackward"]:
            Average = 2.0 / (InNum + OutNum)
            Std = math.sqrt(2.0 / Average)
        else:
            raise Exception()
        return SampleFromUniformDistribution(Shape, 0.0, Std)
    else:
        raise Exception()

def SampleFromXaiverUniform(Shape, **Dict):
    # Y = WX. Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InNum = Shape[0]
    OutNum = Shape[1]
    Priority = Dict.setdefault("Priority", "all")
    if Priority in ["forward", "Forward"]:
        Max = math.sqrt(3.0 / InNum)
    elif Priority in ["backward", "Backward"]:
        Max = math.sqrt(3.0 / OutNum)
    elif Priority in ["all"," ForwardAndBackward"]:
        Average = (InNum + OutNum) / 2.0
        Max = math.sqrt(3.0 / Average)
    else:
        raise Exception()
    Min = - Max
    return SampleFromUniformDistribution(Shape, Min, Max)

def SampleFromXaiverNormal(Shape, NonLinearFunction, **Dict):
    # Y = WX. Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InNum = Shape[0]
    OutNum = Shape[1]
    Priority = Dict.get("Priority")
    if Priority in ["forward", "Forward"]:
        Std = math.sqrt(1.0 / InNum)
    elif Priority in ["backward", "Backward"]:
        Std = math.sqrt(1.0 / OutNum)
    elif Priority in ["all"," ForwardAndBackward"]:
        Average = (InNum + OutNum) / 2.0
        Std = math.sqrt(1.0 / Average)
    else:
        raise Exception()
    return SampleFromNormalDistribution(Shape, 0.0, Std)

def SampleFrom01NormalDistribution(Shape):
    return np.random.normal(size=Shape, loc=0.0, scale=1.0)

def SampleFromNormalDistribution(Shape, Mean=0.0, Std=1.0):
    return np.random.normal(size=Shape, loc=Mean, scale=Std)

SampleFromGaussianDistribution = SampleFromNormalDistribution

def SampleFromUniformDistribution(Shape, Min=0.0, Max=1.0):
    assert Min <= Max
    if Min == Max:
        return SampleFromConstantDistribution(Shape, Min)
    elif Min < Max:
        return np.random.uniform(low=Min, high=Max, size=Shape)
    else:
        raise Exception()

def SampleFromConstantDistribution(Shape, Value):
    return Value * np.ones(Shape)