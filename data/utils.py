import numpy as np

from enum import Enum
import math

def SampleFromKaimingUniform(Shape, NonLinearFunction="ReLU", **Dict):
    # Y = f(WX). Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    if NonLinearFunction in ["ReLU", "relu"]:
        Priority = Dict.setdefault("Priority", "forward")
        if Priority in ["forward", "Forward"]:
            Max = math.sqrt(6.0 / InputNum)
        elif Priority in ["backward", "Backward"]:
            Max = math.sqrt(6.0 / OutputNum)
        elif Priority in ["all"," ForwardAndBackward"]:
            Average = 2.0 / (InputNum + OutputNum)
            Max = math.sqrt(6.0 / Average)
        else:
            raise Exception()
        Min = - Max
        return SampleFromUniformDistribution(Shape, Min, Max)
    else:
        raise Exception()

def SampleFromKaimingNormal(Shape, NonLinearFunction="ReLU", **Dict):
    # Y = f(WX). Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    if NonLinearFunction in ["ReLU", "relu"]:
        Priority = Dict.setdefault("Priority", "all")
        if Priority in ["forward", "Forward"]:
            Std = math.sqrt(2.0 / InputNum)
        elif Priority in ["backward", "Backward"]:
            Std = math.sqrt(2.0 / OutputNum)
        elif Priority in ["all"," ForwardAndBackward"]:
            Average = 2.0 / (InputNum + OutputNum)
            Std = math.sqrt(2.0 / Average)
        else:
            raise Exception()
        return SampleFromUniformDistribution(Shape, 0.0, Std)
    else:
        raise Exception()

def SampleFromXaiverUniform(Shape, **Dict):
    # Y = WX. Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    Priority = Dict.setdefault("Priority", "all")
    if Priority in ["forward", "Forward"]:
        Max = math.sqrt(3.0 / InputNum)
    elif Priority in ["backward", "Backward"]:
        Max = math.sqrt(3.0 / OutputNum)
    elif Priority in ["all"," ForwardAndBackward"]:
        Average = (InputNum + OutputNum) / 2.0
        Max = math.sqrt(3.0 / Average)
    else:
        raise Exception()
    Min = - Max
    return SampleFromUniformDistribution(Shape, Min, Max)

def SampleFromXaiverNormal(Shape, NonLinearFunction, **Dict):
    # Y = WX. Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    Priority = Dict.get("Priority")
    if Priority in ["forward", "Forward"]:
        Std = math.sqrt(1.0 / InputNum)
    elif Priority in ["backward", "Backward"]:
        Std = math.sqrt(1.0 / OutputNum)
    elif Priority in ["all"," ForwardAndBackward"]:
        Average = (InputNum + OutputNum) / 2.0
        Std = math.sqrt(1.0 / Average)
    else:
        raise Exception()
    return SampleFromNormalDistribution(Shape, 0.0, Std)

def SampleFromNormalDistribution(Shape, Mean, Std):
    return np.random.normal(size=Shape, loc=Mean, scale=Std)

def SampleFromUniformDistribution(Shape, Min, Max):
    assert Min <= Max
    if Min == Max:
        return SampleFromConstantDistribution(Shape, Min)
    elif Min < Max:
        return np.random.uniform(low=Min, high=Max, size=Shape)
    else:
        raise Exception()

def SampleFromConstantDistribution(Shape, Value):
    return Value * np.ones(Shape)