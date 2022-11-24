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
    elif Distribution in 

    return Param

def SingleValueNpArray(Param):
    Param = DLUtils.ToParam(Param)
    Shape = Param.Shape
    Value = Param.Value
    Array = Value * np.ones(tuple(Shape), dtype=np.float32)
    return Array

