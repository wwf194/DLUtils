def UpdateDict(DictSource, DictTarget, KeyPrefix=None):
    if KeyPrefix is None:
        for Key, Value in DictSource.keys():
            DictTarget[Key] = Value
    else:
        assert isinstance(KeyPrefix, str)
        for Key, Value in DictSource.keys():
            DictTarget[KeyPrefix + Key] = Value 

def PrintDict(Dict, Out="Std"):
    StrList = []
    
    for Key, Value in Dict.items():
        StrList.append('%s=%s'%(str(Key), str(Value)), end=' ')
    StrList.append('\n')

    Str = "".join(StrList)
    if Out in ["Std", "std"]:
        print(Str)
    elif Out in ["Str", "str"]:
        return Str
    else:
        raise Exception()