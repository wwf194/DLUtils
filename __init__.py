
from .utils._dict import IterableKeyToElement, IterableKeyToKeys
import DLUtils.utils as utils
from .utils.json import \
    IsJsonObj, PyObj, EmptyPyObj, IsPyObj, IsDictLikePyObj, \
    IsListLikePyObj, CheckIsLegalPyName
import DLUtils.utils.json as json

#from .utils import _1DTo2D
from .utils._param import \
    Param, param, \
    new_param, NewParam, ToParam

from .utils import *
SystemType = GetSystemType()

import DLUtils.module as module
from DLUtils.module import AbstractModule, AbstractNetwork, AbstractOperator
import DLUtils.transform as transform
import DLUtils.transform.norm as norm

import DLUtils.log as log # log -> transform

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


import DLUtils.optimize as optimize # module -> optimize
import DLUtils.evaluate as evaluate
import DLUtils.analysis as analysis
import DLUtils.transform as transform
import DLUtils.train as train # module -> train
import DLUtils.network as network
# import DLUtils.network.nonlinear as NonLinear

import DLUtils.task as task
from .task import Task, Dataset

# from .functions import *
# from .log import *

from .optimize import Optimizer
from .optimize import SGD, Adam
from .train.evaluate import Evaluator, EvaluationLog
import DLUtils.plot as plot

import DLUtils.loss as loss

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

import DLUtils.geometry2D as geometry2D


from DLUtils.train import TrainSession, EpochBatchTrainSession

from DLUtils.log import \
    ResetGlobalLogIndex, \
    GlobalLogIndex

from DLUtils.utils.func import \
    EmptyFunction
from DLUtils.utils.file import ParseSavePath
import DLUtils.utils.system as system
from DLUtils.plot import \
    NpArray2ImageFile, \
    Tensor2ImageFile

import DLUtils.example as example

import DLUtils.backend as backend
import DLUtils.backend.torch as torch

PackageFolderPath = DLUtils.file.FolderPathOfFile(__file__)