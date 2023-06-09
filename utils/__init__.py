import os
import re
import functools
import threading
import time
import warnings
import random

from typing import Iterable, List
# from inspect import getframeinfo, stack
# from .attrs import *

from .file import *
import DLUtils.utils._string as string
from DLUtils.utils._string import *

import argparse
import traceback

from .system import GetSystemType, GetSysType
from .file import Str2File
from ._dict import *

def FromFile(FilePath):
    Param = DLUtils.File2Param(FilePath)
    assert hasattr(Param, "_CLASS")
    ModuleClass = DLUtils.python.ParseClass(Param._CLASS)
    Module = ModuleClass().LoadParam(Param)
    return Module

File2Module = FromFile

def File2Param(FilePath):
    return DLUtils.Param().FromFile(FilePath)

def EmptyObj():
    return types.SimpleNamespace()
GenerateEmptyObj = EmptyObj

# def JsonFile2ParamObj(FilePath):
#     JsonDict = JsonFile2JsonDict(FilePath)
#     Obj = utils.JsonStyleObj2Param(JsonDict)
#     DLUtils.JsonFile2Param()
#     return Obj

def Namespace2PyObj(Namespace):
    return DLUtils.json.JsonObj2PyObj(Namespace2Dict(Namespace))

def Namespace2Dict(Namespace):
    return vars(Namespace)

def Dict2Namespace(Dict):
    return argparse.Namespace(Dict)

def ParseTaskName(task):
    if task in ["CleanLog", "CleanLog", "cleanlog"]:
        task = "CleanLog"
    elif task in ["DoTasksFromFile"]:
        task = "DoTasksFromFile"
    elif task in ["CopyProject2DirAndRun", "CopyProject2FolderAndRun", "CPFR"]:
        task = "CopyProject2FolderAndRun"
    else:
        pass
    return task
#     SetAttrs(GlobalParam, "time.EndTime", value=DLUtils.system.GetTime())
#     DurationTime = DLUtils.system.GetTimeDifferenceFromStr(GlobalParam.time.StartTime, GlobalParam.time.EndTime)
#     SetAttrs(GlobalParam, "time.DurationTime", value=DurationTime)

def _StartEndTime2File():
    GlobalParam = DLUtils.GetGlobalParam()
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-0-Time-Start:%s"%GlobalParam.time.EndTime)
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-1-Time- End :%s"%GlobalParam.time.StartTime)
    DLUtils.file.EmptyFile(DLUtils.GetMainSaveDir() + "AAA-2-Time-Duration:%s"%GlobalParam.time.DurationTime)

def ParsedArgs2CmdArgs(ParsedArgs, Exceptions=[]):
    CmdArgsList = []
    for Name, Value in ListAttrsAndValues(ParsedArgs, Exceptions=Exceptions):
        CmdArgsList.append("--%s"%Name)
        CmdArgsList.append(Value)
    return CmdArgsList

def CopyProjectFolder2Dir(DestDir):
    EnsureDir(DestDir)
    DLUtils.file.CopyFolder2DestDir("./", DestDir)
    return

def CopyProjectFolderAndRunSameCommand(Dir):
    CopyProjectFolder2Dir(Dir)

def CleanLog():
    DLUtils.file.RemoveAllFilesAndDirsUnderDir("./log/")

def GetTensorLocation(Method="auto"):
    if Method in ["Auto", "auto"]:
        Location = DLUtils.GetGPUWithLargestUseableMemory()
    else:
        raise Exception()
    return Location

def LoadTaskFile(FilePath="./task.jsonc"):
    TaskObj = DLUtils.json.JsonFile2PyObj(FilePath)
    return TaskObj

def LoadJsonFile(Args):
    if isinstance(Args, DLUtils.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, dict):
        _LoadJsonFile(DLUtils.json.JsonObj2PyObj(Args))
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadJsonFile(Arg)
    elif isinstance(Args, DLUtils.PyObj):
        _LoadJsonFile(Args)
    else:
        raise Exception()

def _LoadJsonFile(Args, **kw):
    Obj = DLUtils.json.JsonFile2PyObj(Args.FilePath)
    MountObj(Args.MountPath, Obj, **kw)

def SaveObj(Args):
    Obj = DLUtils.parse.ResolveStr(Args.MountPath, ObjRoot=DLUtils.GetGlobalParam()),
    Obj.Save(SaveDir=Args.SaveDir)

def IsClassInstance(Obj):
    # It seems that in Python, all variables are instances of some class.
    return

import types
def IsFunction(Obj):
    return isinstance(Obj, types.FunctionType) \
        or isinstance(Obj, types.BuiltinFunctionType)

from collections.abc import Iterable   # import directly from collections for Python < 3.3
def IsIterable(Obj):
    if isinstance(Obj, Iterable):
        return True
    else:
        return False
def IsListLike(List):
    if isinstance(List, list) or isinstance(List, tuple):
        return True
    return False

def RemoveStartEndEmptySpaceChars(Str):
    Str = re.match(r"\s*([\S].*)", Str).group(1)
    Str = re.match(r"(.*[\S])\s*", Str).group(1)
    return Str

RemoveHeadTailWhiteChars = RemoveStartEndEmptySpaceChars

def RemoveWhiteChars(Str):
    Str = re.sub(r"\s+", "", Str)
    return Str

def TensorType(data):
    return data.dtype

def NpArrayType(data):
    if not isinstance(data, np.ndarray):
        return "Not an np.ndarray, but %s"%type(data)
    return data.dtype

def List2NpArray(data):
    return np.array(data)

def Dict2GivenType(Dict, Type):
    if Type in ["PyObj"]:
        return DLUtils.PyObj(Dict)
    elif Type in ["Dict"]:
        return Dict
    else:
        raise Exception(Type)

def ToSaveFormat(Data):
    if isinstance(Data, torch.Tensor):
        return ToNpArray(Data)
    else:
        return Data

def ToRunFormat(Data):
    if isinstance(Data, np.ndarray):
        return ToTorchTensor(Data)
    else:
        return Data

def ToNpArray(data, DataType=None):
    if DataType is None:
        DataType = np.float32
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data, dtype=DataType)
    elif isinstance(data, torch.Tensor):
        return Tensor2NpArray(data)
    elif isinstance(data, float):
        return np.asarray([data],dtype=DataType)
    else:
        raise Exception(type(data))

def ToNpArrayOrNum(data, DataType=None):
    if DataType is None:
        DataType = np.float32
    if isinstance(data, float):
        return data
    if isinstance(data, int):
        return data
    data = ToNpArray(data)
    if data.size == 0: # empty array
        return None
    elif data.size == 1: # single element array
        return data.reshape(1)[0]
    else:
        return data

def ToNpArrayIfIsTensor(data):
    if isinstance(data, torch.Tensor):
        return DLUtils.ToNpArray(data), False
    else:
        return data, True

def ToPyObj(Obj):
    if isinstance(Obj, DLUtils.json.PyObj):
        return Obj
    else:
        return DLUtils.PyObj(Obj)

def ToTrainableTorchTensor(data):
    if isinstance(data, np.ndarray):
        return NpArray2Tensor(data, RequiresGrad=True)
    elif isinstance(data, list):
        return NpArray2Tensor(List2NpArray(data), RequiresGrad=True)
    elif isinstance(data, torch.Tensor):
        data.requires_grad = True
        return data
    else:
        raise Exception(type(data))



def _1DTo2D(data):
    # turn 1-D data to 2-D data for visualization
    DimensionNum = DLUtils.GetDimensionNum(data)
    assert DimensionNum == 1, DimensionNum

    dataNum = data.shape[0]
    RowNum, ColNum = DLUtils.plot.ParseRowColNum(dataNum)
    mask = np.ones((RowNum, ColNum), dtype=np.bool8)

    maskNum = RowNum * ColNum - dataNum
    RowIndex, ColIndex = RowNum - 1, ColNum - 1 # Start from point at right bottom.
    
    for Index in range(maskNum):
        mask[RowIndex, ColIndex] = False
        ColIndex -= 1
    if maskNum > 0:
        dataEnd = np.zeros(maskNum,dtype=data.dtype)
        #dataEnd[:] = np.nan
        data = np.concatenate([data, dataEnd])
    data = data.reshape((RowNum, ColNum))
    return data, mask

def FlattenNpArray(data):
    return data.flatten()

def EnsureFlatNp(data):
    return data.flatten()

EnsureFlat = EnsureFlatNp


def NpArray2List(data):
    return data.tolist()

def ToStandardizeTorchDataType(DataType):
    if DataType in ["Float", "float"]:
        return torch.float32
    elif DataType in ["Double", "double"]:
        return torch.float64



def DeleteKeysIfExist(Dict, Keys):
    for Key in Keys:
        if Key in Dict:
            Dict.pop(Key)
    return Dict

try:
    NpDataTypeMap = IterableKeyToKeys({    
        ("np.float32", "Float32", "Float", "float"): np.float32,
        ("np.int8", "Int8", "int8"): np.int8
    })
except Exception:
    warnings.warn("lib numpy is not found")

def ParseDataTypeNp(DataType):    
    if isinstance(DataType, str):
        DataTypeParsed = NpDataTypeMap.get("DataType")
        if DataTypeParsed is not None:
            return DataTypeParsed
        else:
            try:
                DataTypeParsed = eval(DataType)
            except Exception:
                raise Exception()
            return DataTypeParsed
    else:
        return DataType

def ToGivenDataTypeNp(data, DataType):
    DataType = DLUtils.ParseDataTypeNp(DataType)
    return data.astype(DataType)

def TorchTensor2NpArray(data):
    data = data.detach().cpu().numpy()
    return data # data.grad will be lost.
Tensor2NpArray = TorchTensor2NpArray

def Tensor2Str(data):
    return NpArray2Str(Tensor2NpArray(data))

def Tensor2File(data, SavePath):
    EnsureFileDir(SavePath)
    np.savetxt(SavePath, DLUtils.Tensor2NpArray(data))

def Tensor2NumpyOrFloat(data):
    try:
        _data = data.item()
        return _data
    except Exception:
        pass
    data = data.detach().cpu().numpy()
    return data

def List2NpArray(data, Type=None):
    if Type is not None:
        return np.array(data, dtype=Type)
    else:
        return np.array(data)

def ToList(Obj):
    if isinstance(Obj, list):
        return Obj
    elif isinstance(Obj, np.ndarray):
        return Obj.tolist()
    elif isinstance(Obj, torch.Tensor):
        return NpArray2List(Tensor2NpArray(Obj))
    elif DLUtils.IsListLikePyObj(Obj):
        return Obj.ToList()
    elif isinstance(Obj, dict) or DLUtils.IsDictLikePyObj(Obj):
        raise Exception()
    else:
        return [Obj]

def ToDict(Obj):
    if isinstance(Obj, dict):
        return dict(Obj)
    elif isinstance(Obj, DLUtils.PyObj):
        return Obj.ToDict()
    else:
        raise Exception(type(Obj))

import functools
def SortListByCmpMethod(List, CmpMethod):
    # Python3 no longer supports list.sort(cmp=...)
    List.sort(key=functools.cmp_to_key(CmpMethod))

# def GetFunction(FunctionName, ObjRoot=None, ObjCurrent=None, **kw):
#     return eval(FunctionName.replace("&^", "ObjRoot.").replace("&", "ObjCurrent"))

def ContainAtLeastOne(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args] 
    for Item in Items:
        if Item in List:
            return True
    return False

def ContainAll(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args]   
    for Item in Items:
        if Item not in List:
            return False
    return True

# import timeout_decorator

def CallFunctionWithTimeLimit(TimeLimit, Function, *Args, **ArgsKw):
    # TimeLimit: in seconds.
    event = threading.Event()

    FunctionThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, Function, *Args), kwargs=ArgsKw)
    FunctionThread.setDaemon(True)
    FunctionThread.start()

    TimerThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, ReturnInGivenTime, TimeLimit))
    TimerThread.setDaemon(True)
    # So that this thread will be forced to terminate with the thread calling this function.
    # Which does not satisfy requirement. We need this thread to terminate when this function returns.
    TimerThread.start()
    event.wait()
    return 

def NotifyWhenFunctionReturn(event, Function, *Args, **ArgsKw):
    Function(*Args, **ArgsKw)
    event.set()

def ReturnInGivenTime(TimeLimit, Verbose=True):
    # TimeLimit: float or int. In Seconds.
    if Verbose:
        DLUtils.Log("Start counding down. TimeLimit=%d."%TimeLimit)
    time.sleep(TimeLimit)
    if Verbose:
        DLUtils.Log("TimeLimit reached. TimeLimit=%d."%TimeLimit)
    return

def GetGPUWithLargestUseableMemory(TimeLimit=10, Default='cuda:0'):
    # GPU = [Default]
    # CallFunctionWithTimeLimit(TimeLimit, __GetGPUWithLargestUseableMemory, GPU)
    # return GPU[0]
    return _GetGPUWithLargestUseableMemory()

def __GetGPUWithLargestUseableMemory(List):
    GPU= _GetGPUWithLargestUseableMemory()
    List[0] = GPU
    DLUtils.Log("Selected GPU: %s"%List[0])

def _GetGPUWithLargestUseableMemory(Verbose=True): # return torch.device with largest available gpu memory.
    try:
        import pynvml
        pynvml.nvmlInit()
        GPUNum = pynvml.nvmlDeviceGetCount()
        GPUUseableMemory = []
        for GPUIndex in range(GPUNum):
            Handle = pynvml.nvmlDeviceGetHandleByIndex(GPUIndex) # sometimes stuck here.
            MemoryInfo = pynvml.nvmlDeviceGetMemoryInfo(Handle)
            GPUUseableMemory.append(MemoryInfo.free)
        GPUUseableMemory = np.array(GPUUseableMemory, dtype=np.int64)
        GPUWithLargestUseableMemoryIndex = np.argmax(GPUUseableMemory)    
        if Verbose:
            DLUtils.Log("Useable GPU Num: %d"%GPUNum)
            report = "Useable GPU Memory: "
            for GPUIndex in range(GPUNum):
                report += "GPU%d: %.2fGB "%(GPUIndex, GPUUseableMemory[GPUIndex] * 1.0 / 1024 ** 3)
            DLUtils.Log(report)
        return 'cuda:%d'%(GPUWithLargestUseableMemoryIndex)
    except Exception:
        return "cuda:0"

def split_batch(data, batch_size): #data:(batch_size, image_size)
    sample_num = data.size(0)
    batch_sizes = [batch_size for _ in range(sample_num // batch_size)]
    if not sample_num % batch_size==0:
        batch_sizes.apend(sample_num % batch_size)
    return torch.split(data, section=batch_sizes, dim=0)

def cat_batch(dataloader): #data:(batch_num, batch_size, image_size)
    if not isinstance(dataloader, list):
        dataloader = list(dataloader)
    return torch.cat(dataloader, dim=0)

def import_file(file_from_sys_path):
    if not os.path.isfile(file_from_sys_path):
        raise Exception("%s is not a file."%file_from_sys_path)
    if file_from_sys_path.startswith("/"):
        raise Exception("import_file: file_from_sys_path must not be absolute path.")
    if file_from_sys_path.startswith("./"):
        module_path = file_from_sys_path.lstrip("./")
    module_path = module_path.replace("/", ".")
    return importlib.ImportModule(module_path)



def CopyDict(Dict):
    return dict(Dict)

def GetItemsfrom_dict(dict_, keys):
    items = []
    for name in keys:
        items.append(dict_[name])
    if len(items) == 1:
        return items[0]
    else:
        return tuple(items)   

def write_dict_info(dict_, save_path='./', save_name='dict info.txt'): # write readable dict info into file.
    values_remained = []
    with open(save_path+save_name, 'w') as f:
        for key in dict_.keys():
            value = dict_[value]
            if isinstance(value, _str) or isinstance(value, int):
                f.write('%s: %s'%(_str(key), _str(value)))
            else:
                values_remained.append([key, value])

def AtLeastOneKeyInDict(_Dict, *List, **Dict):
    Num = 0
    if len(KeyList) == 1 and DLUtils.IsListLike(KeyList[0]):
        KeyList = List[0]
    else:
        KeyList = List

    InDictKeyList = []
    for Key in KeyList:
        if Key in _Dict:
            Num += 1
            InDictKeyList.append(Key)
    if Num == 0:
        return None
    elif Num == 1:
        return _Dict[InDictKeyList[0]]
    else:
        raise Exception()


            

def update_key(dict_0, dict_1, prefix='', strip=False, strip_only=True, exempt=[]):
    if not strip:
        for key in dict_1.keys():
            dict_0[prefix + key]=dict_1[key]
    else:
        for key in dict_1.keys():
            trunc_key=trunc_prefix(key, prefix)
            if strip_only:
                if(trunc_key!=key or key in exempt):
                    dict_0[trunc_key]=dict_1[key]
            else:
                dict_0[trunc_key]=dict_1[key]

def set_instance_attr(self, dict_, keys=None, exception=[]):
    if keys is None: # set all keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    setattr(self, key, value)

set_instance_variable = set_instance_attr

def set_dict_variable(dict_1, dict_0, keys=None, exception=['self']): # dict_1: target. dict_0: source.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
        
def set_instance_variable_and_dict(self, dict_1, dict_0, keys=None, exception=['self']): # dict_0: source. dict_1: target dict. self: target class object.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
                    setattr(self, key, value)
                
def set_default_attr(Obj, Key, Value):
    if Obj.__dict__.get(Key) is None:
        setattr(Obj, Key, Value)
setdefault = set_default_attr

set_dict_and_instance_variable = set_class_variable_and_dict = set_instance_variable_and_dict



def GetLastestModel(model_prefix, base_dir='./', is_dir=True):
    # search for directory or file of most recently saved models(model with biggest epoch index)
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d*)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if(max_epoch is None):
                    max_epoch = epoch_num
                else:
                    if(max_epoch < epoch_num):
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + _str(max_epoch) + '/'
    else:
        return "error"

def standardize_suffix(suffix):
    pattern = re.compile(r'\.?(\w+)')
    result = pattern.match(suffix)
    if result is None:
        raise Exception('check_suffix: %s is illegal suffix.'%suffix)
    else:
        suffix = result.group(1)
    return suffix

def EnsureSuffix(name, suffix):
    if not suffix.startswith("."):
        suffix = "." + suffix
    if name.endswith(suffix):
        return suffix
    else:
        return name + suffix

def check_suffix(name, suffix=None, is_path=True):
    # check whether given file name has suffix. If true, check whether it's legal. If false, add given suffix to it.
    if suffix is not None:
        if isinstance(suffix, _str):
            suffix = standardize_suffix(suffix)
        elif isinstance(suffix, list):
            for i, suf_ in enumerate(suffix):
                suffix[i] = standardize_suffix(suf_)
            if len(suffix)==0:
                suffix = None
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(_str(suffix)))      

    pattern = re.compile(r'(.*)\.(\w+)')
    result = pattern.match(name)
    if result is not None: # match succeeded
        name = result.group(1)
        suf = result.group(2)
        if suffix is None:
            return name + '.' + suf
        elif isinstance(suffix, _str):
            if name==suffix:
                return name
            else:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix))
                return name + '.' + suffix
        elif isinstance(suffix, list):
            sig = False
            for suf_ in suffix:
                if suf==suf_:
                    sig = True
                    return name
            if not sig:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix[0]))
                return name + '.' + suffix[0]                
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(_str(suffix)))
    else: # fail to match
        if suffix is None:
            raise Exception('check_suffix: %s does not have suffix.'%name)
        else:
            if isinstance(suffix, _str):
                suf_ = suffix
            elif isinstance(suffix, _str):
                suf_ = suffix[0]
            else:
                raise Exception('check_suffix: invalid suffix: %s'%(_str(suffix)))
            warnings.warn('check_suffix: no suffix found in %s. adding suffix %s.'%(name, suffix))            
            return name + '.' + suf_

from ._string import HasSuffix, RemoveSuffix

try:
    from .math import RandomIntInRange, RandomSelect, RandomSelectFromList
except Exception:
    pass

from .format import NpArray2D2Str, NpArray2D2TextFile, NpArray2Str, NpArray2TextFile

def MultipleRandomIntInRange(Left, Right, Num, IncludeRight=False):
    if not IncludeRight:
        Right += 1
    return RandomSelect(range(Left, Right), Num)

def RandomOrder(List):
    if isinstance(List, range):
        List = list(List)
    random.shuffle(List) # InPlace operation
    return List
def GetLength(Obj):
    if DLUtils.IsIterable(Obj):
        return len(Obj)
    else:
        raise Exception()

import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

def CalculateGitProjectTotalLines(Verbose=False):
    import os
    GitCommand = 'git log  --pretty=tformat: --numstat | awk \'{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\\n", add, subs, loc }\''
    report = os.system(GitCommand)

def DimensionNum(Data):
    if isinstance(Data, torch.Tensor):
        return len(list(Data.size()))
    elif isinstance(Data, np.ndarray):
        return len(Data.shape)
    else:
        raise Exception(type(Data))
GetDimensionNum = DimensionNum

def ToLowerStr(Str):
    return Str.lower()

def GetSavePathFromName(Name, Suffix=""):
    if not Suffix.startswith("."):
        Suffix = "." + Suffix
    FilePath = DLUtils.GetMainSaveDir() + Name + Suffix
    FilePath = DLUtils.file.RenameIfFileExists(FilePath)
    return FilePath

def Float2StrDisplay(Float):
    if np.isinf(Float):
        return "inf"
    if np.isneginf(Float):
        return "-inf"
    if np.isnan(Float):
        return "NaN"

    if Float==0.0:
        return "0.0"

    Positive = Float < 0.0
    if not Positive:
        Float = - Float
        Sign = - 1.0
    else:
        Sign = 1.0

    Base, Exp = DLUtils.math.Float2BaseAndExponent(Float)
    TicksStr = []
    if 1 <= Exp <= 2:
        FloatStr = _str(int(Float))
    elif Exp == 0:
        FloatStr = '%.1f'%Float
    elif Exp == -1:
        FloatStr = '%.2f'%Float
    elif Exp == -2:
        FloatStr = '%.3f'%Float
    else:
        FloatStr = '%.2e'%Float
    return FloatStr * Sign

def Floats2StrDisplay(Floats):
    Floats = ToNpArray(Floats)
    Base, Exp = DLUtils.math.FloatsBaseAndExponent(Floats)

def Floats2StrWithEqualLength(Floats):
    Floats = DLUtils.ToNpArray(Floats)
    Base, Exp = DLUtils.math.Floats2BaseAndExponent(Floats)
    # to be implemented

def MountDictOnObj(Obj, Dict):
    Obj.__dict__.update(Dict)

ExternalMethods = None
ExternalClasses = None
def InitExternalMethods():
    global ExternalMethods, ExternalClasses
    ExternalMethods = DLUtils.utils.EmptyPyObj()
    ExternalClasses = DLUtils.utils.EmptyPyObj()

def RegisterExternalMethods(Name, Method):
    setattr(ExternalMethods, Name, Method)

def RegisterExternalClasses(Name, Class):
    setattr(ExternalClasses, Name, Class)

from ._string import Str2Bytes, Bytes2Str

def Unzip(Lists):
    return zip(*Lists)

def Zip(*Lists):
    return zip(*Lists)

def EnsurePyObj(Obj):
    if isinstance(Obj, argparse.Namespace):
        return Namespace2PyObj(Obj)
    elif isinstance(Obj, dict) or isinstance(Obj, list):
        return DLUtils.PyObj(Obj)
    else:
        raise Exception(type(Obj))

from collections import defaultdict
def CreateDefaultDict(GetDefaultMethod):
    return defaultdict(GetDefaultMethod)
GetDefaultDict = CreateDefaultDict

from .format import *

def RandomImage(Height=512, Width=512, ChannelNum=None, 
        BatchNum=10, DataType="TorchTensor"):
    if ChannelNum is None:
        Shape = [Height, Width]
    else:
        Shape = [Height, Width, ChannelNum]
    if BatchNum is not None:
        if isinstance(BatchNum, float):
            BatchNum = round(BatchNum)
        assert isinstance(BatchNum, int)
        Shape = [BatchNum] + Shape
    Image = DLUtils.SampleFromUniformDistribution(Shape, -1.0, 1.0)
    if DataType in ["np", "numpy"]:
        return Image
    elif DataType in ["TorchTensor"]:
        return DLUtils.ToTorchTensor(Image)
    else:
        raise Exception()

NoiseImage = RandomImage

def NormWithinNStd2Range(Data, Min, Max, N=1.0, Clip=True):
    Mean0 = Data.mean()
    Std0 = Data.std()
    Mean1 = (Min + Max) / 2.0
    Std1 = (Max - Min) / 2.0
    Data1 = (Data - Mean0) / (N * Std0) * Std1 + Std1
    if Clip:
        Data1 = np.clip(Data1, Min, Max)
    return Data1

from datetime import datetime
def DataTimeObj2Str(Obj, Format='%Y-%m-%d %H:%M:%S.%f'):
    return datetime.strftime(Obj, Format)

def Str2DataTimeObj(Str, Format='%Y-%m-%d %H:%M:%S.%f'):
    return datetime.strptime(Str, Format)
    
import functools
NormWithinStd2Range = functools.partial(NormWithinNStd2Range, N=1.0)
NormWithin1Std2Range = NormWithinStd2Range

# import DLUtils.utils.network as network
from ..backend.torch.format import ToTorchTensor, ToTorchTensorOrNum, NpArray2Tensor, NpArray2TorchTensor
from ..backend.torch import GetTensorByteNum, GetTensorElementNum

import DLUtils.utils.network as network
try:
    import DLUtils.utils.image as image
except Exception:
    warnings.warn("failed to import DLUtils.utils.image")
    pass
try:
    import DLUtils.utils.timer as timer
except Exception:
    pass
try:
    import DLUtils.utils.sql as sql
except Exception:
    pass
try:
    import DLUtils.utils.video as video
except Exception:
    pass

if GetSystemType() in ["Windows", "win"]:
    import DLUtils.backend.win as win
