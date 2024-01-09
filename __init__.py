Verbose = True

from .utils._dict import IterableKeyToElement, IterableKeyToKeys, ToDict
import DLUtils.utils._string as string
from .utils._string import PrintStrToLibOutPipe as print
from .utils._string import (
        ResetLibOutPipe,
        SetFileStrOut, CloseFileStrOut,
        Print2StdErr, PrintHeartBeatTo,
        PrintTo, PrintUTF8To, WriteTo, PrintStrTo, OutputTo,
        PrintTimeStrTo, PrintTimeStr2, PrintCurrentTimeTo, 
        PrintWithTimeStr, PrintTimeStr,
        PrintPIDTo,
        AddLibIndent, AddLibIndentLevel, IncreaseLibIndent, IncreaseLibIndentLevel,
        DecreaseLibIndent, DecreaseLibIndentLevel,
        SetLibIndentLevel,
        GetLibOutPipe, GetStdOut,
        RemoveHeadTailWhiteChars,
        ToHex, Address,
        GetLibOutPipeWriter, PrintStrTo, PrintTo
    )

import DLUtils.utils._json as json
from .utils._json import \
    IsJsonObj, PyObj, EmptyPyObj, IsPyObj, IsDictLikePyObj, \
    IsListLikePyObj, CheckIsLegalPyName
import DLUtils.utils as utils

import DLUtils.utils._param as param

from .utils._param import (
        Param, param,
        new_param, NewParam,
        ToParam,
        Param2JsonFile, Param2JsonStr
    )
# import DLUtils.utils._numpy as numpy
from .utils._numpy import (
        SetSeedForNumpy,
        NpArray2TextFile,
        NpArray2D2TextFile,
        FlattenNpArray,
        NpArray2List,
        EnsureFlatNp,
        EnsureFlat
    )

try:
    from .utils._numpy import NpArray2D2Str
except Exception:
    pass

from .utils.format import (
    Curve2TextFile
)

from DLUtils.utils import (
        GetSystemType,
        ExpandIterableKey,
        ToList,
        GetFirstValue,
        GetFirstNotNoneValue,
        Float2StrDisplay,
        ToLowerStr,
        EmptyListWithShape, GetListWithShape,
        LeafListToMean, LeafListToStd
    )
try:
    from .utils import ToNpArray
except Exception:
    pass
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

import DLUtils.utils.parse as parse
import DLUtils.utils.file as file
import DLUtils.utils.func as function
import DLUtils.utils._math as math


try:
    from DLUtils.utils._math import (
        SampleFromKaimingNormal,
        SampleFromKaimingUniform,
        SampleFromXaiverNormal,
        SampleFromXaiverUniform,
        SampleFromConstantDistribution,
        SampleFromNormalDistribution,
        SampleFrom01NormalDistribution,
        SampleFromGaussianDistribution,
        SampleFromUniformDistribution,
    )
except Exception:
    pass

try:
    import DLUtils.utils.plot as plot
except Exception:
    pass

import DLUtils.utils.__struct__ as struct
import DLUtils.utils.system as system

try:
    import DLUtils.utils.sql as sql
except Exception:
    pass
else:
    try:
        import DLUtils.utils.sql.sqlite as sqlite
    except Exception:
        pass
    try:
        import DLUtils.utils.sql.mysql as mysql
    except Exception:
        pass

import DLUtils.utils.python as python
from .utils.__struct__ import FixedSizeQueuePassiveOutInt, FixedSizeQueuePassiveOutFloat
try:
    import DLUtils.optimize as optimize # module -> optimize
    from .optimize import Optimizer
    from .optimize import SGD, Adam
except Exception:
    pass

import DLUtils.analysis as analysis
import DLUtils.transform as transform
try:
    import DLUtils.train as train # module -> train
    from .train.evaluate import Evaluator, EvaluationLog
    from .train import TrainSession, EpochBatchTrainSession
except Exception:
    pass

import DLUtils.network as network
try:
    import DLUtils.network as network
except Exception:
    print("Failed to import DLUtils.network")
try:
    import DLUtils.task as task
    from .task import Task, Dataset
except Exception:
    pass

try:
    import DLUtils.utils.image as image
except Exception:
    pass

# from .functions import *
# from .log import *

try:
    import DLUtils.loss as loss
except Exception:
    pass
try:
    import DLUtils.data as data
    from DLUtils.data.generate import (
        DefaultConv2DKernel, DefaultUpConv2DKernel, ShapeWithSameValue,
        DefaultNonLinearLayerWeight, DefaultNonLinearLayerBias, DefaultLinearLayerWeight,
        DefaultUpConv2DBias, DefaultConv2DBias, DefaultLinearLayerBias,
        DefaultVanillaRNNHiddenWeight
    )
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
    from DLUtils.utils.plot import \
        NpArray2ImageFile, \
        Tensor2ImageFile
except Exception:
    pass
from DLUtils.utils.func import (EmptyFunction)

from .utils.file import *
from .utils.file import ParseSavePath, ChangeFilePathSuffix, FolderPathOfFile

try:
    import DLUtils.example as example
except Exception:
    pass

try:
    import DLUtils.backend as backend
    import DLUtils.backend._torch as torch
except Exception:
    pass
else:
    from .backend._torch import (
        SetSeedForTorch,
        NpArray2Tensor,
        TorchTensor2NpArray, TorchTensorToNpArray
    )

try:
    import DLUtils.utils.parallel as parallel
    from .utils.parallel import RunProcessPool
except Exception:
    pass

PackageFolderPath = FolderPathOfFile(__file__)

try:
    import DLUtils.utils._time as time
except Exception:
    pass
else:
    from DLUtils.utils._time import CurrentTimeStr, TimeStr
from DLUtils.utils.system import (
        NewCmdArg, ParseCmdArg,
        ErrorStackStr, PrintErrorStackTo, PrintErrorStackWithInfoTo, PrintErrorStackWithInfo2,
        GetCurrentProcessID, CurrentPID, CurrentProcessID
    )
try:
    from DLUtils.backend._torch import NullParameter, ToTorchTensor, ToTorchTensorOrNum
except Exception:
    pass

import sys
def DoTask(Param):
    TaskName = GetFirstValue(Param, ["Task", "Name"])
    if TaskName in ["Move", "MoveFile"]:
        DirSourceList = Param["From"]
        if isinstance(DirSourceList, str):
            DirSourceList = [DirSourceList]
        for DirSource in DirSourceList:
            DirDest = Param["To"]
            Pattern = Param.get("WithPattern")
            if Pattern is not None:                        
                # for FileName in DLUtils.ListFileNameWithPattern(DirSource, Pattern):
                #     pass
                DLUtils.MoveFileWithFileNamePattern(
                    DirSource,
                    DirDest,
                    Pattern,
                    FileSizeMax="5.00GB",
                    MoveFileBeingUsed=False
                )
            else:
                assert len(Param) == 3
                DLUtils.file.MoveAllFile(DirSource, DirDest, MoveFileBeingUsed=False)
    elif TaskName in ["DeleteFile"]:
        Dir = Param["In"]
        Dir = DLUtils.StandardizeDirPath(Dir)
        DLUtils.DeleteFileWithFileNamePattern(
            Dir, Param["WithPattern"]
        )
    elif TaskName in ["ReleaseFile"]:
        FilePathSource = Param["From"]
        if isinstance(FilePathSource, str):
            FilePathSourceList = [FilePathSource]
        else:
            FilePathSourceList = list(FilePathSource)
        for Index, FilePathSource in enumerate(FilePathSourceList):
            FilePathSourceList[Index] = DLUtils.CheckFileExists(FilePathSource)
        DirPathDestList = Param["To"]
        if isinstance(DirPathDestList, str):
            DirPathDestList = [DirPathDestList]
        for DirPathDest in DirPathDestList:
            try:
                for FilePathSource in FilePathSourceList:
                    DirPathDest = DLUtils.EnsureDir(DirPathDest)
                    FilePathDest = DirPathDest + DLUtils.FileNameFromFilePath(FilePathSource)
                    print("Copying file: (%s)-->(%s)"%(FilePathSource, FilePathDest))
                    DLUtils.CopyFile(
                        FilePathSource, FilePathDest
                    )
            except Exception:
                print("Failed to release to folder: (%s)"%DirPathDest)
                print(traceback.format_exc())
    elif TaskName in ["MoveDirIntoDir"]:
        DLUtils.MoveDirIntoDir(
            DirSource=Param["From"],
            DirDest=Param["To"] 
        )
    else:
        raise Exception("Invalid task name: %s"%TaskName)

def RunAtBackground(
        MainFunc,
        IsDebug, __file__=None, StdOut=None, StdErr=None, *List, **Dict):
    if IsDebug:
        # exceptions will be directly thrown.
        # stdout / stderr will not be redirected to file.
        MainFunc(*List, **Dict)
    else:
        import sys
        if StdOut is None:
            f = open(
                DLUtils.FileNameFromPath(__file__, StripSuffix=True) + ".txt", "w"
            ) # stdout pipe
        else:
            f = StdOut

        if StdErr is None:
            e = open(
                DLUtils.FileNameFromPath(__file__, StripSuffix=True) + ".txt.err.txt", "w"
            ) # stderr pipe
        else:
            e = StdErr
        sys.stderr = e
        sys.stdout = f
        try:
            MainFunc(*List, **Dict)
        except Exception:
            DLUtils.PrintErrorStackWithInfoTo(e)
        else:
            DLUtils.PrintTo(f, "Finished. %s."%DLUtils.CurrentTimeStr())
        e.close()
        f.close()
    
def ExecuteFunction(Func, IsDebug, StdOut=None, StdErr=None, *List, **Dict):
    if IsDebug:
        # exceptions will be directly thrown.
        # stdout / stderr will not be redirected to file.
        return Func(*List, **Dict)
    else:
        if StdOut is None:
            pass
        else:
            _stdout = sys.stdout
            sys.stdout = StdOut
        try:
            return Func(*List, **Dict)
        except Exception:
            if StdErr is None:
                DLUtils.PrintErrorStackWithInfoTo(sys.stderr)
            else:
                DLUtils.PrintErrorStackWithInfoTo(StdErr)
        sys.stdout = _stdout
        return None

def SetSeedForRandom(Seed: int):
    import random
    random.seed(0)