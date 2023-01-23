# from DLUtils.PyObj import *
from DLUtils.system import GetSystemType
SystemType = GetSystemType()

from .utils.json import IsJsonObj, PyObj, EmptyPyObj, IsPyObj, IsDictLikePyObj, IsListLikePyObj, CheckIsLegalPyName
import DLUtils.utils as utils
import DLUtils.utils.json as json

#from .utils import _1DTo2D
from .utils._param import \
    Param, ToParam, param \

import DLUtils.attr as attrs
import DLUtils.parse as parse
import DLUtils.geometry2D as geometry2D
import DLUtils.file as file

from DLUtils.module import AbstractModule, AbstractNetwork, AbstractOperator
import DLUtils.module as module
import DLUtils.module.norm as norm
import DLUtils.module.transform as transform

import DLUtils.log as log # log -> transform

import DLUtils.structure as structure
import DLUtils.function as function
import DLUtils.utils.math as math

from .utils import *
import DLUtils.system as system

import DLUtils.optimize as optimize # module -> optimize
import DLUtils.evaluate as evaluate
import DLUtils.analysis as analysis
import DLUtils.transform as transform
import DLUtils.train as train # module -> train
import DLUtils.network as network
import DLUtils.network.NonLinear as NonLinear


import DLUtils.task as task
from .task import Task

# from .functions import *
# from .log import *

from .train.algorithm import Optimizer
from .train.evaluate import Evaluator, EvaluationLog
import DLUtils.plot as plot

import DLUtils.train.loss as loss
from .train.loss import Loss

from DLUtils.data.utils import \
    SampleFromKaimingNormal, \
    SampleFromKaimingUniform, \
    SampleFromXaiverNormal, \
    SampleFromXaiverUniform, \
    SampleFromConstantDistribution, \
    SampleFromNormalDistribution, \
    SampleFromUniformDistribution, \
    Conv2DKernel, UpConv2DKernel

from DLUtils.parse import ParseSavePath
from DLUtils.train import TrainSession
import DLUtils.train.loss as loss

from DLUtils.log import \
    ResetGlobalLogIndex, \
    GlobalLogIndex

from DLUtils.function import \
    EmptyFunction

from DLUtils.plot import \
    NpArray2ImageFile, \
    Tensor2ImageFile

import DLUtils.example as example

from .structure import FixedSizeQueuePassiveOutInt32 


PackageFolderPath = DLUtils.file.FolderPathOfFile(__file__)