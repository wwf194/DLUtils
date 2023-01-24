import numpy as np

from enum import Enum
import math

def Conv2DKernel(Shape, GroupNum=1, **Dict):
    """
    Shape: [InNum, OutNum, KernelHeight, KernelWidth]
    torch: w ~ U(-sqrt(k), sqrt(k))
        where k = GroupNum / (InNum * KernelWidth * KernelHeight)
    """
    Max = GroupNum / (Shape[0] * Shape[2] * Shape[3])
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    return SampleFromUniformDistribution((Shape[1], Shape[0] // GroupNum, Shape[2], Shape[3]), Min, Max)

def Conv2DBias(Shape, GroupNum=1, **Dict):
    """
    Shape: [InNum, OutNum, KernelHeight, KernelWidth]
    pytorch: w ~ U(-sqrt(k), sqrt(k)). k = GroupNum / (InNum * KernelWidth * KernelHeight)
    """
    Max = math.sqrt(GroupNum / (Shape[0] * Shape[2] * Shape[3]))
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    return SampleFromUniformDistribution((Shape[1], Shape[0] // GroupNum, Shape[2], Shape[3]), Min, Max)

def UpConv2DKernel(Shape, GroupNum=1, **Dict):
    """
    Shape: [InNum, OutNum, KernelHeight, KernelWidth]
        where k = GroupNum / (OutNum * KernelWidth * KernelHeight)
    pytorch: w ~ U(-sqrt(k), sqrt(k))
    """
    Max = math.sqrt(GroupNum / (Shape[1] * Shape[2] * Shape[3]))
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    # Weight: [InNum, OutNum // GroupNum, KernelHeight, KernelWidth]
    return SampleFromUniformDistribution((Shape[0], Shape[1] // GroupNum, Shape[2], Shape[3]), Min, Max)

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

def SampleFromNormalDistribution(Shape, Mean, Std):
    return np.random.normal(size=Shape, loc=Mean, scale=Std)

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