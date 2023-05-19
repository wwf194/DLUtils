
class DLUtilsDict(dict):
    def update(self, *List):
        dict.update(self, *List)
        return self

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

def IterableKeyToKeys(Dict):
    for Key, Value in dict(Dict).items():
        if isinstance(Key, tuple) or isinstance(Key, set):
            for _Key in Key:
                Dict[_Key] = Value
            Dict.pop(Key)

    if isinstance(Dict, dict):
        Dict = DLUtilsDict(Dict)
    return Dict

ExpandIterableKey = IterableKeyToElement = IterableKeyToKeys