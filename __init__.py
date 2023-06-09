
from .utils._dict import IterableKeyToElement, IterableKeyToKeys
import DLUtils.utils as utils
from .utils.json import \
    IsJsonObj, PyObj, EmptyPyObj, IsPyObj, IsDictLikePyObj, \
    IsListLikePyObj, CheckIsLegalPyName
import DLUtils.utils.json as json

#from .utils import _1DTo2D
import DLUtils.utils._param as param
from .utils._param import \
    Param, param, \
    new_param, NewParam, ToParam, Param2JsonFile, Param2JsonStr

from .utils import *
SystemType = GetSystemType()

import DLUtils.module as module
try:
    from .module import AbstractModule, AbstractNetwork, AbstractOperator
    from .module.abstract_module import GetParamMapDefault
    from .network.convolution import GetParamMapDefaultConv
except Exception:
    pass

import DLUtils.transform as transform
try:
    import DLUtils.transform.norm as norm
except Exception:
    pass
try:
    import DLUtils.log as log # log -> transform
except Exception:
    pass

import DLUtils.utils.attrs as attrs
import DLUtils.utils.parse as parse
import DLUtils.utils.file as file
import DLUtils.utils.func as function
import DLUtils.utils.math as math
import DLUtils.utils.__struct__ as struct
import DLUtils.utils.system as system
import DLUtils.utils.sql as sql
# import DLUtils.python as python
from .utils.__struct__ import FixedSizeQueuePassiveOutInt, FixedSizeQueuePassiveOutFloat
try:
    import DLUtils.optimize as optimize # module -> optimize
    from .optimize import Optimizer
    from .optimize import SGD, Adam
except Exception:
    pass
import DLUtils.evaluate as evaluate
import DLUtils.analysis as analysis
import DLUtils.transform as transform
try:
    import DLUtils.train as train # module -> train
    from .train.evaluate import Evaluator, EvaluationLog
    from .train import TrainSession, EpochBatchTrainSession
except Exception:
    pass
try:
    import DLUtils.network as network
except Exception:
    pass
try:
    import DLUtils.task as task
    from .task import Task, Dataset
except Exception:
    pass

# from .functions import *
# from .log import *

try:
    import DLUtils.plot as plot
except Exception:
    pass
try:
    import DLUtils.loss as loss
except Exception:
    pass
try:
    import DLUtils.data as data
    from DLUtils.data.generate import \
        SampleFromKaimingNormal, \
        SampleFromKaimingUniform, \
        SampleFromXaiverNormal, \
        SampleFromXaiverUniform, \
        SampleFromConstantDistribution, \
        SampleFromNormalDistribution, \
        SampleFrom01NormalDistribution, \
        SampleFromGaussianDistribution, \
        SampleFromUniformDistribution, \
        DefaultConv2DKernel, DefaultUpConv2DKernel, ShapeWithSameValue, \
        DefaultNonLinearLayerWeight, DefaultNonLinearLayerBias, DefaultLinearLayerWeight, \
        DefaultUpConv2DBias, DefaultConv2DBias, DefaultLinearLayerBias, \
        DefaultVanillaRNNHiddenWeight
except Exception:
    pass

try:
    import DLUtils.geometry2D as geometry2D
except Exception:
    pass

try:
    from DLUtils.log import \
        ResetGlobalLogIndex, \
        GlobalLogIndex
except Exception:
    pass
try:
    from DLUtils.plot import \
        NpArray2ImageFile, \
        Tensor2ImageFile
except Exception:
    pass
from DLUtils.utils.func import \
    EmptyFunction
from DLUtils.utils.file import ParseSavePath
import DLUtils.utils.system as system

try:
    import DLUtils.example as example
except Exception:
    pass

try:
    import DLUtils.backend as backend
    import DLUtils.backend.torch as torch
except Exception:
    pass



PackageFolderPath = DLUtils.file.FolderPathOfFile(__file__)

from DLUtils.utils._string import _print as print