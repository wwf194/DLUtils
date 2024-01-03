class Dict(dict):
    def __init__(self, Dict=None, **_Dict):
        super().__init__()
        if _Dict is None:
            assert isinstance(Dict, dict)
        if Dict is not None:
            self.update(Dict)
        self.update(_Dict)
    # def update(self, *List):
    #     dict.update(self, *List)
    #     return self
    def hasattr(self, Key):
        return Key in self
    def __getattr__(self, Key, Default=None):
        return self.get(Key, Default)
    getattr = __getattr__
def UpdateDict(DictSource, DictTarget, KeyPrefix=None):
    if KeyPrefix is None:
        for Key, Value in DictSource.keys():
            DictTarget[Key] = Value
    else:
        assert isinstance(KeyPrefix, str)
        for Key, Value in DictSource.keys():
            DictTarget[KeyPrefix + Key] = Value 

def ToDict(**Dict):
    return Dict

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

def IterableKeyToKeys(_Dict):
    for Key, Value in dict(_Dict).items():
        if isinstance(Key, tuple) or isinstance(Key, set):
            for _Key in Key:
                _Dict[_Key] = Value
            _Dict.pop(Key)

    if isinstance(_Dict, dict):
        _Dict = Dict(_Dict)
    return _Dict

ExpandIterableKey = IterableKeyToElement = IterableKeyToKeys

def GetFromKeyList(Dict, *KeyList, Default=None):
    for Key in KeyList:
        if Key in Dict:
            return Dict[Key]
    return Default

class MultiToOneMap(dict):
    def From(self, *KeyList):
        self.KeyList = KeyList
        return self
    def To(self, Value):
        for Key in self.KeyList:
            self[Key] = Value
        delattr(self, "KeyList")
        return self