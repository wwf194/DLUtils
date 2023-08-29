import datetime
import time
import re
import os
import warnings
import DLUtils

TimeStampBase = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

def TimeStr2Second(TimeStr):
    if isinstance(TimeStr, int):
        return TimeStr
    elif isinstance(TimeStr, float):
        return round(TimeStr)
    
    Pattern = r"(\d*\.\d*)([days|d|])"

    Result = re.match(Pattern, TimeStr)
    
    if Result is not None:
        NumStr = Result.group(1)
        UnitStr = Result.group(2)
    else:
        raise Exception()

try:
    import tzlocal # pip install tzlocal
except Exception:
    warnings.warn("lib tzlocal not found.")

def GMTTime2LocalTime(DateTimeObj, TimeZone=None):
    if TimeZone is None:        
        return DateTimeObj.astimezone(tzlocal.get_localzone())
    else:
        raise NotImplementedError()

def DateTimeObj2TimeStampFloat(DataTimeObj):
    # return time.mktime(DataTimeObj.timetuple())
    # TimeStamp = DataTimeObj.timestamp() # >= Python 3.3
    TimeDiff = DataTimeObj - TimeStampBase
    TimeStamp = TimeDiff.total_seconds()
    return TimeStamp
    
def DateTimeObj2TimeStampInt(DataTimeObj):
    return round(time.mktime(DataTimeObj.timetuple()))

def DateTimeObj2LocalTimeStr(DateTimeObjUTC, Format="%Y-%m-%d %H-%M-%S", TimeZone=None):
    DateTimeObjLocal = GMTTime2LocalTime(DateTimeObjUTC, TimeZone=TimeZone)
    return DateTimeObjLocal.strftime(Format)

DateTimeObj2Str = DateTimeObj2LocalTimeStr

def TimeStamp2DateTimeObj(TimeStamp):
    # TimeStamp: float or int. unit: second.
    # millisecond is supported, and will not be floored.
    DateTimeObj = TimeStampBase + datetime.timedelta(seconds=TimeStamp)
    return DateTimeObj
    # might throw error for out of range time stamp.
    # DateTimeObj = date.fromtimestamp(TimeStamp)

def GetCurrentTimeStampFloat():
    # return type: float, with millisecond precision.
    return DateTimeObj2TimeStampFloat(
        # datetime.datetime.now()
        datetime.datetime.utcnow() # caution. should use Greenwich Mean Time(GMT) here.
    )
GetCurrentTimeStamp = GetCurrentTimeStamp = GetCurrentTimeStampFloat

def GetCurrentTimeStampInt():
    return round(GetCurrentTimeStampFloat())
CurrentTimeStampInt = GetCurrentTimeStampInt

def GetCurrentTimeStampInt():
    return round(DateTimeObj2TimeStampFloat(
        # datetime.datetime.now()
        datetime.datetime.utcnow() # caution. should use Greenwich Mean Time(GMT) here.
    ))

def TimeStamp2Type(TimeStamp, Type):
    if Type in ["UnixTimeStamp", "TimeStamp"]:
        return TimeStamp
    elif Type in ["Str", "LocalTimeStr"]:
        return DLUtils.time.TimeStamp2Str(TimeStamp)
    else:
        raise Exception()

def FileCreateTime(FilePath):
    FilePath = DLUtils.file.CheckFileExists(FilePath)
    if DLUtils.system.IsWin():
        return os.path.getctime(FilePath)
    else:
        raise NotImplementedError()

def FolderLastModifiedTime(DirPath, ReturnType="LocalTimeStr"):
    # following operation will update a folder's last modified time:
        # delete child file or folder.
        # create(include by paste) child file or folder.
    DirPath = DLUtils.CheckDirExists(DirPath)
    LastModifiedTimeStamp = os.path.getmtime(DirPath)
    return TimeStamp2Type(LastModifiedTimeStamp, ReturnType)
DirLastModifiedTime = FolderLastModifiedTime

def FileLastModifiedTime(FilePath, ReturnType="LocalTimeStr"):
    # following operation will update a folder's last modified time:
        # delete child file or folder.
        # create(include by paste) child file or folder.
    FilePath = DLUtils.CheckFileExists(FilePath)
    LastModifiedTimeStamp = os.path.getmtime(FilePath)
    return TimeStamp2Type(LastModifiedTimeStamp, ReturnType)

def LastModifiedTime(Path, ReturnType="TimeStamp"):
    if DLUtils.file._ExistsFile(Path):
        # Path = DLUtils.file.StandardizeFilePath(Path)
        return FolderLastModifiedTime(Path, ReturnType)
    elif DLUtils.file._ExistsDir(Path):
        # Path = DLUtils.file.StandardizeDirPath(Path)
        return FolderLastModifiedTime(Path, ReturnType)
    else:
        raise Exception("non-existent path: %s"%Path)

def TimeStamp2LocalTimeStr(TimeStamp, Format="%Y-%m-%d %H-%M-%S"):
    if isinstance(TimeStamp, int):
        TimeStamp = TimeStamp * 1.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeStamp)
    return DateTimeObj2Str(DateTimeObj, Format=Format)
TimeStamp2Str = TimeStamp2LocalTimeStr

def CurrentTimeStr(Format=None, Verbose=False):
    if Format is None:
        Format = "%Y-%m-%d %H-%M-%S"
    TimeStr = time.strftime(Format, time.localtime()) # time display style: 2016-03-20 11:45:39
    if Verbose:
        print(TimeStr)
    return TimeStr
GetCurrentTime = GetTime = CurrentTimeStr

try:
    import dateutil
    def GetTimeDifferenceFromStr(TimeStr1, TimeStr2):
        Time1 = dateutil.parser.parse(TimeStr1)
        Time2 = dateutil.parser.parse(TimeStr2)

        TimeDiffSeconds = (Time2 - Time1).total_seconds()
        TimeDiffSeconds = round(TimeDiffSeconds)

        _Second = TimeDiffSeconds % 60
        Minute = TimeDiffSeconds // 60
        _Minute = Minute % 60
        Hour = Minute // 60
        TimeDiffStr = "%d:%02d:%02d"%(Hour, _Minute, _Second)
        return TimeDiffStr
except Exception:
    warnings.warn("lib dateutil not found")