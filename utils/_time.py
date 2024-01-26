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

# import tzlocal # pip install tzlocal
    # tzlocal is not a standard package
    # //2024.01 import failure despited installed

def GetLocalTimeZone():
    LocalTimeZone = datetime.datetime.utcnow().astimezone().tzinfo
    return LocalTimeZone # <class 'datetime.timezone'>

def GMTTime2LocalTime(DateTimeObj, TimeZone=None):
    # assert IsTzLocalImported
    if TimeZone is None:        
        return DateTimeObj.astimezone(
            # tzlocal.get_localzone()
            GetLocalTimeZone()
        )
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

def TimeStampToType(TimeStamp, Type, Format=None):
    if Type in ["UnixTimeStamp", "TimeStamp"]:
        return TimeStamp
    elif Type in ["Str", "LocalTimeStr"]:
        return DLUtils.time.TimeStampToStr(
            TimeStamp, TimeZone="Local", Format=Format
        )
    else:
        raise Exception()

def TimeStampToLocalTimeStr(TimeStamp, Format=None):
    if Format is None:
        Format = "%Y-%m-%d %H-%M-%S"
    if isinstance(TimeStamp, int):
        TimeStamp = TimeStamp * 1.0
    DateTimeObj = TimeStamp2DateTimeObj(TimeStamp)
    return DateTimeObj2Str(DateTimeObj, Format=Format)

def TimeStampToStr(TimeStamp, TimeZone="Local", Format=None):
    if TimeZone in ["Local"]:
        return TimeStampToLocalTimeStr(TimeStamp, Format=Format)
    elif TimeZone in ["UTC", "Greenwich", "UTC+0", "UTC0"]:
        return TimeStampToUTCTimeStr(TimeZone, Format=Format)
    else:
        raise Exception()

def TimeStampToUTCTimeStr(TimeZone, Format=None):
    return NotImplementedError()

def CurrentTimeStr(Format=None, Verbose=False):
    if Format is None:
        Format = "%Y-%m-%d %H-%M-%S"
    _TimeStr = time.strftime(Format, time.localtime()) # time display style: 2016-03-20 11:45:39
    if Verbose:
        print(_TimeStr)
    return _TimeStr
GetCurrentTime = GetTime = TimeStr = CurrentTimeStr

try:
    import dateutil
except Exception:
    if DLUtils.Verbose:
        warnings.warn("lib dateutil not found")
    IsDateUtilImported = False
else:
    IsDateUtilImported = True

def GetTimeDifferenceFromStr(TimeStr1, TimeStr2):
    assert IsDateUtilImported
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