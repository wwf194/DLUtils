import DLUtils
import numpy as np
def NewNpArray(Param):
    if isinstance(Param, dict):
        Param = DLUtils.Param().FromDict(Param)
    
    Dim = Param.setdefault("Dim", 1)
    
    if Dim == 1:
        return NewNpArray1D(Param)
    elif Dim == 2:
        return NewNpArray2D(Param)
    else:
        raise Exception()
def NewNpArray1D(Param: DLUtils.Param):
    DataType = np.float32
    Brief = Param.Get("Brief")
    Size = Param.Size
    if Brief is not None:
        if Brief in ["Ones", "ones"]:
            np.ones((Size), dtype=DataType)
        elif Brief in ["Zeros", "zeros"]:
            np.zeros((Size), dtype=DataType)
        elif Brief in ["OneHot", "one_hot"]:
            HotIndex = Param.HotIndex
            np.one_
        else:
            raise Exception(Brief)
    else:

def NewNpArray2D(Param, DataType=torch.float32):
    DataType = np.float32
    if Init.Method in ["Kaiming", "KaimingUniform", "KaimingNormal"]:
        if Init.Method in ["KaimingNormal"]: # U ~ [-bound, bound], bound = sqrt(6/(1+a^2)*FanIn)
            SetAttrs(Init, "Distribution", value="Normal")
        elif Init.Method in ["KaimingUniform"]:
            SetAttrs(Init, "Distribution", value="Uniform")
        else:
            EnsureAttrs(Init, "Distribution", default="Uniform")
        EnsureAttrs(Init, "Mode", default="In")
        EnsureAttrs(Init, "Coefficient", default=1.0)
        if Init.Mode in ["BasedOnInputNum", "BasedOnInput", "In"]:
            if Init.Distribution in ["Uniform"]:
                Init.Range = [
                    - Init.Coefficient * (6.0 / param.Size[0]) ** 0.5,
                    Init.Coefficient * (6.0 / param.Size[0]) ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
            elif Init.Distribution in ["Uniform+"]:
                Init.Range = [
                    0.0,
                    2.0 * Init.Coefficient * 6.0 ** 0.5 / param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
            elif Init.Distribution in ["Normal"]:
                # std = sqrt(2 / (1 + a^2) * FanIn)
                Mean = 0.0
                Std = Init.Coefficient * (2.0 / param.Size[0]) ** 0.5
                weight = np.random.normal(Mean, Std, tuple(param.Size))
            else:
                # to be implemented
                raise Exception()
        else:
            raise Exception()
            # to be implemented
    elif Init.Method in ["xaiver", "glorot"]:
        Init.Method = "xaiver"
        raise Exception()
        # to be implemented
    else:
        raise Exception()
        # to be implemented
    return DLUtils.NpArray2Tensor(weight, DataType=DataType, RequiresGrad=True)

def SampleFromDistribution(Param):
    if isinstance(Param, dict):
        Param = DLUtils.Param().FromDict(Param)
    Param.Type = "SampleFromDistribution" # Param as a log of data creation.
    Distribution = Param.Distribution
    Shape = Param.Shape
    if Distribution in ["KaimingUniform"]:
        
        pass
    elif Distribution in ["XaiverUniform"]:
        pass
    elif Distribution in ["KaimingNormal"]:
        pass
    elif Distribution in ["XaiverNormal"]:
        pass
    else:
        raise Exception()
    return Param

def SingleValueNpArray(Param):
    Param = DLUtils.ToParam(Param)
    Shape = Param.Shape
    Value = Param.Value
    Array = Value * np.ones(tuple(Shape), dtype=np.float32)
    return Array
    #DLUtils.plot.PlotGaussianDensityCurve(axRight, weight) # takes too much time
    DLUtils.plot.PlotHistogram(
        ax2, weightForColorMap, Color="Black",
        XLabel="Connection Strength", YLabel="Ratio", Title="Distribution"
    )

    plt.suptitle("%s Shape:%s"%(Name, weight.shape))
    plt.tight_layout()
    if SavePath is None:
        SavePath = DLUtils.GetMainSaveDir + "weights/" + "%s.svg"%Name
    DLUtils.plot.SaveFigForPlt(SavePath=SavePath)
    return

def Weight2D(InputNum, OutputNum, Distribution, **Dict):
    return SampleFromDistribution(
        (InputNum, OutputNum), Distribution, **Dict
    )

from enum import Enum
import math

def SampleFromKaimingUniform(Shape, NonLinearFunction, **Dict):
    # Y = f(WX). Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    if NonLinearFunction in ["ReLU", "relu"]:
        Priority = Dict.get("Priority")
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

def SampleFromKaimingNormal(Shape, NonLinearFunction, **Dict):
    # Y = f(WX). Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    if NonLinearFunction in ["ReLU", "relu"]:
        Priority = Dict.get("Priority")
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

def SampleFromXaiverUniform(Shape, NonLinearFunction, **Dict):
    # Y = WX. Keep variance of forward signal or backward gradient.
    assert len(Shape) == 2
    InputNum = Shape[0]
    OutputNum = Shape[1]
    Priority = Dict.get("Priority")
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
    return np.random.normal(Shape, Mean, Std)

def SampleFromUniformDistribution(Shape, Min, Max):
    assert Min <= Max
    if Min == Max:
        return SampleFromConstantDistribution(Shape, Min)
    elif Min < Max:
        np.random.uniform(low=Min, high=Max, size=Shape)
    else:
        raise Exception()

def SampleFromConstantDistribution(Shape, Value):
    return Value * np.ones(Shape)