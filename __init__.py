# from DLUtils.PyObj import *
from DLUtils.system import GetSystemType
SystemType = GetSystemType()

from .utils.json import IsJsonObj, PyObj, EmptyPyObj, IsPyObj, IsDictLikePyObj, IsListLikePyObj, CheckIsLegalPyName
import DLUtils.utils.json as json

from .utils import _1DTo2D
from .utils.utils import Param, ToParam, param
#import utils.utils as utils

import DLUtils.attr as attrs
import DLUtils.parse as parse
import DLUtils.geometry2D as geometry2D
import DLUtils.file as file
import DLUtils.plot as plot
import DLUtils.module as module
import DLUtils.log as log # log -> transform
import DLUtils.transform as transform
import DLUtils.train as train # module -> train
import DLUtils.router as router
import DLUtils.functions as functions
import DLUtils.utils.math as math

import DLUtils.optimize as optimize # module -> optimize
import DLUtils.dataset as dataset
import DLUtils.loss as loss
import DLUtils.evaluate as evaluate
import DLUtils.system as system
import DLUtils.analysis as analysis

import DLUtils.module.NeuralNetwork as NN

from .utils import *

from .functions import *
from .log import *

from DLUtils.NeuralNetwork.utils import \
    SampleFromKaimingNormal, \
    SampleFromKaimingUniform, \
    SampleFromXaiverNormal, \
    SampleFromXaiverUniform, \
    SampleFromConstantDistribution, \
    SampleFromNormalDistribution, \
    SampleFromUniformDistribution

from DLUtils.parse import ParseSavePath