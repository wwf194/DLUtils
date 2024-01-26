import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
else:
    np = DLUtils.GetLazyNumpy()
from enum import Enum
import math

def ShapeWithSameValue(Shape, Value):
    return np.full(Shape, Value)

def DefaultNonLinearLayerWeight(Shape, NonLinear):
    return DLUtils.math.SampleFromKaimingUniform(Shape, NonLinear=NonLinear)

def DefaultNonLinearLayerBias(Shape):
    if isinstance(Shape, list) or isinstance(Shape, tuple):
        UnitNum = Shape[0]
    elif isinstance(Shape, int):
        UnitNum = Shape
    else:
        raise Exception()
        
    Min = - 1.0 / math.sqrt(UnitNum)
    Max = - Min
    return DLUtils.math.SampleFromUniformDistribution((UnitNum), Min, Max)

DefaultLinearLayerBias = DefaultNonLinearLayerBias

def DefaultLinearLayerWeight(Shape):
    return DLUtils.math.SampleFromKaimingUniform(Shape, NonLinear=None)

def DefaultVanillaRNNHiddenWeightTorch(Shape):
    """
    torch implementation
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    Shape: (HiddenNum, HiddenNum)
    w ~ U(-sqrt(k), sqrt(k))
        where k = 1 / (HiddenNum)
    """
    HiddenNum = Shape[0]
    assert Shape[0] == Shape[1]
    Max = 1.0 / math.sqrt(HiddenNum)
    Min = - Max
    return DLUtils.math.SampleFromUniformDistribution(Shape, Min, Max)

DefaultVanillaRNNHiddenWeight = DefaultVanillaRNNHiddenWeightTorch

def DefaultConv2DKernelTorch(Shape, GroupNum=1, **Dict):
    """
    torch implementation
    Shape: (InNum, OutNum, KernelHeight, KernelWidth)
    w ~ U(-sqrt(k), sqrt(k))
        where k = GroupNum / (InNum * KernelWidth * KernelHeight)
    """
    Max = GroupNum / (Shape[0] * Shape[2] * Shape[3])
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    # Kernel: (OutNum, InNum // GroupNum, Height, Width). torch.conv2d.
    return DLUtils.math.SampleFromUniformDistribution((Shape[1], Shape[0] // GroupNum, Shape[2], Shape[3]), Min, Max)
DefaultConv2DKernel = DefaultConv2DKernelTorch

def DefaultConv2DBias(Shape, GroupNum=1, **Dict):
    """
    Shape: [InNum, OutNum, KernelHeight, KernelWidth]
    pytorch: w ~ U(-sqrt(k), sqrt(k)). k = GroupNum / (InNum * KernelWidth * KernelHeight)
    """
    Max = math.sqrt(GroupNum / (Shape[0] * Shape[2] * Shape[3]))
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    return DLUtils.math.SampleFromUniformDistribution((Shape[1]), Min, Max)

def DefaultUpConv2DKernel(Shape, GroupNum=1, **Dict):
    """
    Shape: (InNum, OutNum, KernelHeight, KernelWidth)
        where k = GroupNum / (OutNum * KernelWidth * KernelHeight)
    pytorch: w ~ U(-sqrt(k), sqrt(k))
    """
    Max = math.sqrt(GroupNum / (Shape[1] * Shape[2] * Shape[3]))
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    # Weight: [InNum, OutNum // GroupNum, KernelHeight, KernelWidth]
    return DLUtils.math.SampleFromUniformDistribution((Shape[0], Shape[1] // GroupNum, Shape[2], Shape[3]), Min, Max)

def DefaultUpConv2DBias(Shape, GroupNum=1, **Dict):
    """
    Shape: [InNum, OutNum, KernelHeight, KernelWidth]
    pytorch: w ~ U(-sqrt(k), sqrt(k)). k = GroupNum / (OutNum * KernelWidth * KernelHeight)
    """
    Max = math.sqrt(GroupNum / (Shape[1] * Shape[2] * Shape[3]))
    Min = - Max
    assert Shape[0] % GroupNum == 0
    assert Shape[1] % GroupNum == 0
    return DLUtils.math.SampleFromUniformDistribution((Shape[1]), Min, Max)
