import DLUtils

import warnings
from enum import Enum

from .json.parse import _JsonStr2Tree
from .json.parse import NODE_TYPE as NODE_PARSE_TYPE, COMMENT_TYPE

def ToParam(Obj):
    if isinstance(Obj, Param):
        return Obj
    elif isinstance(Obj, dict):
        return Param(Obj)
    else:
        raise Exception()

class NODE_TYPE(Enum):
    LEAF = 0
    SPINE = 1
    SPINE_WITH_LEAF = 2
    COMMENT = 3 # Temporal Node

class NODE_FORMAT(Enum):
    NULL = 0
    DICT = 1
    LIST = 2
    COMMENT = 3

class NODE_CHILD_TYPE(Enum):
    NO_SUBNODE = 0
    NO_SUBNODE_WITH_LEAF = 1
    LEAF = 1
    SINGLE_SUBNODE = 2
    SINGLE_SUBNODE_WITH_LEAF = 3
    MULTI_SUBNODE = 4
    MULTI_SUBNODE_WITH_LEAF = 5

def NewParam(**Dict):
    return Param(Dict)

def new_param(**Dict):
    return param(Dict)

class param:
    """
    NODE_TYPE.DICT
        requires all key names be str type.
    NODE_TYPE.LIST
        requires all key names be int type.
    """
    def __init__(self, _CONTENT=None, Dict=None, _PATH_FROM_ROOT=["ROOT"], _TYPE=NODE_TYPE.SPINE, _FORMAT=NODE_FORMAT.DICT, IsSuper=False):  
        self.SetAttr("_LIST", [])
        self.SetAttr("_DICT", {})
        if Dict is not None:
            self.SetAttr("_FORMAT", Dict["_FORMAT"])
            self.SetAttr("_LEAF", Dict.get("_LEAF"))
        else:
            self.SetAttr("_TYPE", _TYPE)
            self.SetAttr("_FORMAT", _FORMAT)
            self.SetAttr("_PATH_FROM_ROOT", _PATH_FROM_ROOT)
        
        if not IsSuper:
            if _CONTENT is not None:
                if isinstance(_CONTENT, param):
                    if _CONTENT._FORMAT == NODE_FORMAT.DICT:
                        Content = _CONTENT._DICT
                    elif _CONTENT._FORMAT == NODE_FORMAT.LIST:
                        Content = _CONTENT._LIST
                    else:
                        raise Exception()
                else:
                    Content = _CONTENT
                if isinstance(Content, dict):
                    self.SetAttr("_FORMAT", NODE_FORMAT.DICT)
                    self.from_dict(Content)
                elif isinstance(Content, list):
                    self.SetAttr("_FORMAT", NODE_FORMAT.LIST)
                    self.from_list(Content)
                else:
                    raise Exception()   
    def from_dict(self, Dict):
        self.SetAttr("_FORMAT", NODE_FORMAT.DICT)
        self.SetAttr("_DICT", {})
        self.absorb_dict(Dict)
        return self
    def absorb_dict(self, Dict):
        # assert isinstance(_DICT, dict)
        _DICT = self.GetAttr("_DICT")
        for key, item in Dict.items():
            if isinstance(item, dict):
                _DICT[key] = param(_FORMAT=NODE_FORMAT.DICT).from_dict(item)
            elif isinstance(item, list):
                _DICT[key] = param(_FORMAT=NODE_FORMAT.LIST).from_list(item)
            else:
                _DICT[key] = item
        return self
    def from_list(self, List):
        _LIST = self.GetAttr("_LIST")
        for Index, Item in enumerate(List):
            if isinstance(Item, dict):
                _LIST.append(param(_FORMAT=NODE_FORMAT.DICT).from_dict(Item))
            elif isinstance(Item, list):
                _LIST.append(param(_FORMAT=NODE_FORMAT.LIST).from_list(Item))
            else:
                _LIST.append(Item)
        return self
    def Get(self, Key):
        if Key in self.__dict__:
            return self.__dict__[Key]
        else:
            return None
    def Absorb(self, Node):
        assert Node._FORMAT == NODE_FORMAT.DICT
        for Key, SubNode in Node._DICT.items():
            if self.hasattr(Key):
                SubNodeExist = self.getattr(Key)
                if isinstance(SubNodeExist, DLUtils.param):
                    if isinstance(SubNode, DLUtils.param):
                        SubNodeExist.Absorb(SubNode)
                    else:
                        SubNodeExist._LEAF = SubNode
                        if SubNodeExist._TYPE == NODE_TYPE.SPINE:
                            SubNodeExist._TYPE = NODE_TYPE.SPINE_WITH_LEAF
                else:
                    if isinstance(SubNode, DLUtils.param):
                        if SubNode._TYPE == SubNode._SPINE:
                            SubNode._LEAF = SubNodeExist
                            SubNode._TYPE = NODE_TYPE.SPINE_WITH_LEAF
                        self._DICT[Key] = SubNode
                    else:
                        self._DICT[Key] = SubNode # overwrite leaf
            else:
                self.setattr(Key, SubNode)
        if Node.HasAttr("_LEAF"):
            self._LEAF = Node._LEAF
            if self._TYPE == NODE_TYPE.SPINE:
                self._TYPE = NODE_TYPE.SPINE_WITH_LEAF
        if Node.HasAttr("_COMMENT"):
            for Comment in Node._COMMENT:
                self.AddComment(Comment)
        return self
    def HasAttr(self, Key):
        return Key in self.__dict__
    def GetAttr(self, Key):
        return self.__dict__[Key]
    def GetAttrAndDelete(self, Key):
        return self.__dict__.pop(Key)
    def SetAttr(self, Key, Value):
        self.__dict__[Key] = Value
        return Value
    def DelAttr(self, Key):
        self.__dict__.pop(Key)
    def hasattr(self, Key):
        return Key in self._DICT
    def addattr(self, Key, Value):
        self._DICT[Key] = Value
    def setattr(self, Key, Value):
        self.GetAttr("_DICT")[Key] = Value
        return self
    def getattr(self, Key):
        return self._DICT[Key]
    def get(self, Key):
        if isinstance(Key, int):
            if len(self._LIST) < Key:
                return self._LIST[Key]
            else:
                return None
        else:
            return self._DICT.get(Key)
    def delattr(self, Key, AllowNonExist=True):
        if not Key in self._DICT:
            if AllowNonExist:
                return None
            else:
                raise Exception()
        else:
            return self._DICT.pop(Key) 
    def delattrifexists(self, Key):
        if not Key in self._DICT:
            return None
        else:
            return self._DICT.pop(Key) 
    def setdefault(self, Key, Value):
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_FORMAT.DICT) 
        if not Key in self._DICT:
            self._DICT[Key] = Value
        return self._DICT[Key]
    def getdefault(self, Key, DefaultValue):
        return self._DICT.get(Key, DefaultValue)
    def items(self):
        return self.GetAttr("_DICT").items()
    def append(self, Item):
        self._LIST.append(Item)
    def keys(self):
        return self._DICT.keys()
    def values(self):
        return self._DICT.values()
    def pop(self, Key):
        return self._DICT.pop(Key)
    def __hasattr__(self, Key):
        return Key in self._DICT
    def __setattr__(self, Key, Value):
        self._DICT[Key] = Value
    def __getattr__(self, Key):
        return self._DICT[Key]
    def __setitem__(self, Key, Value):
        if isinstance(Key, int):
            self._LIST[Key] = Value
        else:
            self._DICT[Key] = Value
        return self    
    def __getitem__(self, Key):
        if isinstance(Key, int):
            return self._LIST[Key]
        else:
            return self._DICT[Key]
    def __delattr__(self, Name):
        self._DICT.pop(Name)
        return self
    # For Serialization
    def __getstate__(self):
        return self.__dict__
    # For Unserialization
    def __setstate__(self, Dict):
        for Key, Item in Dict.items():
            self.SetAttr(Key, Item)
    def __getitem__(self, Key):
        if self._FORMAT == NODE_FORMAT.DICT:
            if isinstance(Key, str):
                return self._DICT[Key]
            elif isinstance(Key, int):
                return list(self._DICT.keys())[Key]  
            else:
                raise Exception()
        else:
            return self._LIST[Key]
    def __len__(self):
        if self._FORMAT == NODE_FORMAT.DICT:
            return len(list(self._DICT.keys()))
        else:
            return len(self._LIST)
    def ToJsonFile(self, FilePath):
        Param2JsonFile(self, FilePath)
        return self
    def ToFile(self, FilePath):
        DLUtils.file.Obj2BinaryFile(self, FilePath)
        return self

class Param(param):
    # tree node class for representing json-like structure.
    # content information is stored in attribute _DICT, or _LIST, _LEAF, depending on node type.
    # meta information is stored in other attributes such as _TYPE, _FORMAT etc.
    # viewed externally, node acts like an object with its attributes storing content information.
        # content could be accessed using . or [] operator
        # this is implemented using magic methods of python. 
    # method names with UpperCamelCase typically indicates functions to manipulate meta information
    # method names with   lowercases   typically indicates functions to manipulate content information.
    def __init__(self, _CONTENT=None, Dict=None, _PATH_FROM_ROOT=["ROOT"], _TYPE=NODE_TYPE.SPINE, _FORMAT=NODE_FORMAT.DICT):
        super().__init__(_CONTENT=_CONTENT, Dict=Dict, _FORMAT=_FORMAT, IsSuper=True)
        self.SetAttr("_DICT_TEMP", {})
        if Dict is not None:
            self.SetAttr("_TYPE", Dict["_TYPE"])
            self.SetAttr("_PATH_FROM_ROOT", Dict["_PATH_FROM_ROOT"])
        else:
            self.SetAttr("_TYPE", _TYPE)
            self.SetAttr("_PATH_FROM_ROOT", _PATH_FROM_ROOT)
        if _CONTENT is not None:
            if isinstance(_CONTENT, dict):
                self.SetAttr("_FORMAT", NODE_FORMAT.DICT)
                self.from_dict(_CONTENT)
            elif isinstance(_CONTENT, list):
                self.SetAttr("_FORMAT", NODE_FORMAT.LIST)
                self.from_list(_CONTENT)
            elif isinstance(_CONTENT, Param):
                self.FromParam(_CONTENT)
            else:
                raise Exception()   
    def FromParam(self, Node):
        for Key, Item in Node.__dict__.items():
            if Key == "_DICT":
                self.SetAttr("_DICT", dict(Item))
            else:
                self.SetAttr(Key, Item)
        return
    def AddComment(self, Comment, Type=None):
        CommentList = self.SetDefault("_COMMENT", [])
        if Type is None:
            if self._TYPE == NODE_TYPE.LEAF:
                Type = COMMENT_TYPE.LEAF
            else:
                if len(CommentList) == 0:
                    Type = COMMENT_TYPE.SPINE_BEFORE
                else:
                    IsBeforeEmpty, IsAfterEmpty = True, True
                    for Comment in CommentList:
                        _Type = Comment[1]
                        if _Type == COMMENT_TYPE.SPINE_BEFORE:
                            IsBeforeEmpty = False
                        elif _Type == COMMENT_TYPE.SPINE_AFTER:
                            IsAfterEmpty = False
                        else:
                            raise Exception()
                    if IsBeforeEmpty:
                        Type = COMMENT_TYPE.SPINE_BEFORE
                    elif IsAfterEmpty:
                        Type = COMMENT_TYPE.SPINE_AFTER
                    else:
                        Type = COMMENT_TYPE.SPINE_AFTER
        CommentList.append([Comment, Type])
    def AddCommentToChild(self, Key, Comment, Type=None):
        self.Get(Key)
    def from_dict(self, Dict):
        Paths = _JsonLike2Paths(Dict)
        Tree = _Paths2Tree(Paths)
        Node = _Tree2Param(Tree)
        self.FromParam(Node)
        return self
    def Exists(self):
        if self.Get("_IS_FALSE") is True:
            self.DestroyAlongSpine()
            return False
        else:
            return True
    def DestroyAlongSpine(self):
        Node = self
        while Node.Get("_IS_FALSE") is True:
            SupNode = Node.GetAttrAndDelete("_PARENT")
            SupNodeKey = Node.GetAttrAndDelete("_PARENT_KEY")
            SupNode._DICT_TEMP.pop(SupNodeKey)
            Node = SupNode
        return
    def SubstantiateAlongSpine(self, _TYPE, _FORMAT=NODE_FORMAT.DICT):
        Node = self
        Index = 0
        while Node.Get("_IS_FALSE") is True:
            Node.DelAttr("_IS_FALSE")
            if Index == 0:
                Node.SetAttr("_TYPE", _TYPE)
                Node.SetAttr("_FORMAT", _FORMAT)
            else:
                Node.SetAttr("_TYPE", NODE_TYPE.SPINE)
                Node.SetAttr("_FORMAT", NODE_FORMAT.DICT)
            SupNode = Node.GetAttrAndDelete("_PARENT")
            SupNodeKey = Node.GetAttrAndDelete("_PARENT_KEY")
            SupNode._DICT_TEMP.pop(SupNodeKey)
            SupNode._DICT[SupNodeKey] = Node
            Node = SupNode
            Index += 1
        return
    def SetParent(self, Parent, Key):
        self.SetAttr("_PARENT", Parent)
        self.SetAttr("_PARENT_KEY", Key)
    def SetDefault(self, Key, DefaultValue):
        if self.HasAttr(Key):
            return self.GetAttr(Key)
        else:
            return self.SetAttr(Key, DefaultValue)
    def FromFile(self, FilePath):
        Obj = DLUtils.file.BinaryFile2Obj(FilePath)
        self.FromParam(Obj)
        return self
    def absorb_dict(self, Dict):
        TreeSelf = Param2Tree(self)
        PathsSelf = Tree2Paths(TreeSelf, SplitKeyByDot=True)
        Paths = _JsonLike2Paths(Dict)
        Tree = _Paths2Tree(PathsSelf + Paths)
        
        Node = _Tree2Param(Tree)
        self.FromParam(Node)
        return self    
    def hasattr(self, Key):
        KeyList = Key.split(".")
        Base = self
        for Key in KeyList:
            if Key in Base._DICT:
                Base = Base.getattr(Key)
            else:
                return False
        if self.Get("_IS_FALSE") is True:
            self.DestroyAlongSpine()
        return True
    def setattr(self, Key, Value):
        if isinstance(Key, list):
            KeyList = Key
        elif isinstance(Key, str):
            KeyList = Key.split(".")
        else:
            raise Exception()
        
        if len(KeyList) == 1:
            self._DICT[KeyList[0]] = Value
            if self.Get("_IS_FALSE") is True:
                self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_FORMAT.DICT)
            return self
        else:
            KeyCurrent = KeyList[0]
            if self.hasattr(KeyCurrent):
                SubNode = self.getattr(KeyCurrent)
                if isinstance(SubNode, Param):
                    pass
                elif isinstance(SubNode, param):
                    if len(KeyList[1:]) > 1:
                        SubNode = Param().FromParam(SubNode)
                    else:
                        pass
                else: # LeafNode
                    Leaf = SubNode
                    SubNode = self.setemptyattr(KeyCurrent)
                    SubNode._LEAF = Leaf
            else:
                SubNode = self.setemptyattr(KeyCurrent)
            SubNode.setattr(KeyList[1:], Value)
    def getattr(self, Key):
        if isinstance(Key, list):
            KeyList = Key
        elif isinstance(Key, str):
            KeyList = Key.split(".")
        else:
            raise Exception()
        
        if len(KeyList) == 1:
            return self._DICT[KeyList[0]]
        else:
            KeyCurrent = KeyList[0]
            SubNode = self.getattr(KeyCurrent)
            assert isinstance(SubNode, param)
        return SubNode.getattr(KeyList[1:])
    def setemptyattr(self, Key):
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_FORMAT.DICT)
        SubNode = DLUtils.Param({})
        self._DICT[Key] = SubNode
        return SubNode
    def setdefault(self, Key, Value):
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_FORMAT.DICT) 
        if isinstance(Key, list):
            KeyList = Key
        elif isinstance(Key, str):
            KeyList = Key.split(".")
        else:
            raise Exception()
        
        Key = KeyList[0]
        if len(KeyList) == 1:
            if not Key in self._DICT:
                self._DICT[Key] = Value
            return self._DICT[Key]
        else:
            if self.hasattr(Key):
                SubNode = self.getattr(Key)
            else:
                SubNode = self.setemptyattr(Key)
            return SubNode.setdefault(KeyList[1:], Value)
    def append(self, Item):
        self._LIST.append(Item)
        if isinstance(Item, param):
            pass
        return self
    def __setattr__(self, Key, Value):
        self._DICT[Key] = Value
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_FORMAT.DICT)
    def __getattr__(self, Key):
        if Key in self._DICT:
            return self._DICT[Key]
        else:
            if Key in self._DICT_TEMP:
                return self._DICT_TEMP[Key]
            else:
                SubNode = Param()
                SubNode.SetAttr("_IS_FALSE", True)
                _PATH_FROM_ROOT = self.Get("_PATH_FROM_ROOT")
                if _PATH_FROM_ROOT is not None:
                    SubNode.SetAttr("_PATH_FROM_ROOT", self.GetAttr("_PATH_FROM_ROOT") + [Key])
                self._DICT_TEMP[Key] = SubNode
                SubNode.SetParent(self, Key)
                return SubNode




def _NewNode(
        _TYPE, 
        _FORMAT=NODE_FORMAT.NULL,
        _LEAF=None,
        _PATH_FROM_ROOT=None
    ):
    assert not isinstance(_PATH_FROM_ROOT, int)
    return {
        "_TYPE": _TYPE,
        "_FORMAT": _FORMAT,
        "_LEAF": _LEAF,
        "_LEAF_CONFLICT": [],
        "_DICT": {},
        "_LIST": {},
        "_PATH_FROM_ROOT": _PATH_FROM_ROOT,
    }

def _SetNodeKey(Node, Key, Value):
    if isinstance(Key, int):
        #Node["_LIST"] = Value
        Node["_LIST"][Key] = Value
    elif isinstance(Key, str):
        Node["_DICT"][Key] = Value
    else:
        raise Exception()
    return Value

def Param2JsonFile(Obj:Param, FilePath):
    JsonStr = Param2JsonStr(Obj)
    DLUtils.file.Str2TextFile(JsonStr, FilePath)
    return

def ToJsonElement(Obj):
    if isinstance(Obj, float) or \
        isinstance(Obj, int) or \
        isinstance(Obj, str):
        return Obj

def _NodeTypeDetail(Node):
    HAS_LEAF =  _HasLeaf(Node)
    CHILD_NUM = _SubNodeNum(Node)

    if CHILD_NUM == 0:
        if HAS_LEAF:
            return NODE_CHILD_TYPE.NO_SUBNODE_WITH_LEAF
        else:
            return NODE_CHILD_TYPE.NO_SUBNODE
    if CHILD_NUM == 1:
        if HAS_LEAF:
            return NODE_CHILD_TYPE.SINGLE_SUBNODE_WITH_LEAF
        else:
            return NODE_CHILD_TYPE.SINGLE_SUBNODE
    else:
        if HAS_LEAF:
            return NODE_CHILD_TYPE.MULTI_SUBNODE_WITH_LEAF
        else:
            return NODE_CHILD_TYPE.MULTI_SUBNODE

def _SubNodeNum(Node):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    if _TYPE==NODE_TYPE.LEAF:
        return 0
    else:
        if _FORMAT==NODE_FORMAT.DICT:
            return len(Node["_DICT"].keys())
        elif _FORMAT==NODE_FORMAT.LIST:
            NodeList = Node["_LIST"]
            if isinstance(Node["_LIST"], dict):
                return len(NodeList.keys())
            return len(Node["_LIST"])
        else:
            raise Exception()

def _HasLeaf(Node):
    return Node["_TYPE"] in [NODE_TYPE.LEAF, NODE_TYPE.SPINE_WITH_LEAF]

def _IsDict(Node):
    return Node["_FORMAT"]==NODE_FORMAT.DICT and Node["_TYPE"]!=NODE_TYPE.LEAF

def _IsList(Node):
    return Node["_FORMAT"]==NODE_FORMAT.LIST and Node["_TYPE"]!=NODE_TYPE.LEAF

def _IsSpine(Node):
    _TYPE = Node["_TYPE"]
    return _TYPE==NODE_TYPE.SPINE or _TYPE==NODE_TYPE.SPINE_WITH_LEAF

def _ChildNum(Node):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    # assert _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]
    if _TYPE == NODE_TYPE.LEAF:
        return 0
    if _FORMAT==NODE_FORMAT.DICT:
        return len(Node["_DICT"].keys())
    elif _FORMAT==NODE_FORMAT.LIST:
        return len(Node["_LIST"])
    else:
        warnings.warn("_FORMAT:{_FORMAT}")
        return 0

def Tree2JsonStr(RootNode):
    _DetectTreeCompress(RootNode)
    _CompressTree(RootNode)
    return _Tree2JsonStr(RootNode)

import json
import numpy as np
def ToJsonStr(Obj):
    if isinstance(Obj, set):
        Obj = list(Obj)

    if isinstance(Obj, str):
        Content = "\"%s\""%Obj
    elif isinstance(Obj, bool):
        if Obj:
            Content = "true"
        else:
            Content = "false"
    elif isinstance(Obj, int) or isinstance(Obj, float): # isinstance(True, int) -> True
        Content = str(Obj)
    elif Obj is None:
        Content = "null"
    elif isinstance(Obj, _SymbolEmptyDict) or isinstance(Obj, dict):
        Content = "{}"
    elif isinstance(Obj, _SymbolEmptyList) or isinstance(Obj, list):
        #Content = "[]"
        #Content = str(Obj) # avoid single quotation marks
        Content = json.dumps(Obj)
    elif isinstance(Obj, np.ndarray):
        if len(Obj.shape) == 1 and Obj.shape[0] < 20:
            Content = f"{str(DLUtils.NpArray2List(Obj))}"
        else:
            Content = f"\"np.ndarray. Type: {Obj.dtype}. Shape: {Obj.shape}\""
    else:
        Content = "\"_NON_BASIC_JSON_TYPE\""
    return Content

def _Tree2JsonStr(RootNode):
    StrList = []
    _Tree2JsonStrRecur(RootNode, StrList, 0, None)
    return "".join(StrList)

def _AppendWithIndent(StrList, IndentNum, Str):
    for _ in range(IndentNum):
        StrList.append("\t")
    StrList.append(Str)

def JsonDict2Str(Dict):
    StrList = []
    _JsonDict2StrRecur(Dict, StrList, 0)
    return "".join(StrList)

def _JsonDict2StrRecur(Obj, StrList, IndentNum, ParentKey=None):
    if isinstance(Obj, list):
        _AppendWithIndent(StrList, IndentNum, "[\n")
        IndexMax = len(Obj) - 1
        for Index, item in enumerate(Obj):
            _JsonDict2StrRecur(item, StrList, IndentNum + 1)
            if Index < IndexMax:
                StrList.append(",\n")
            else:
                StrList.append("\n")
        _AppendWithIndent(StrList, IndentNum, "]\n")
    elif isinstance(Obj, dict):
        _AppendWithIndent(StrList, IndentNum, "{\n")
        IndexMax = len(list(Obj.keys())) - 1
        Index = 0
        for Key, Item in Obj.items():
            _AppendWithIndent(StrList, IndentNum + 1, "\"%s\":"%Key)
            _JsonDict2StrRecur(Item, StrList, IndentNum + 1, Key)
            if Index < IndexMax:
                StrList.append(",\n")
            else:
                StrList.append("\n")
            Index += 1
        _AppendWithIndent(StrList, IndentNum, "}\n")
    else:
        Content = ToJsonStr(Obj)
        if isinstance(ParentKey, str):
            StrList.append(Content)
        elif isinstance(ParentKey, int):
            _AppendWithIndent(StrList, IndentNum, Content)   
        else:
            raise Exception()

def _AddComment(Node, StrList, _TYPE=None):
    # Add Comment
    CommentList = Node.get("_COMMENT")
    AddedComment = False
    if CommentList is not None:
        assert isinstance(CommentList, list)
        if _TYPE is None:
            for Comment in CommentList:
                StrList.append("\t")
                StrList += Comment[0]
                AddedComment = True
        else:
            for Comment in CommentList:
                if Comment[1] == _TYPE:
                    StrList.append("\t")
                    StrList += Comment[0]
                    AddedComment = True
    return AddedComment

def _Tree2JsonStrRecur(Node, StrList, IndentNum, Key, NoIndent=False):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    if _TYPE==NODE_TYPE.LEAF:
        if isinstance(Key, int) and not NoIndent:
            _AppendWithIndent(StrList, IndentNum, ToJsonStr(Node["_LEAF"]))
        else:
            StrList.append(ToJsonStr(Node["_LEAF"]))
        
    elif _TYPE==NODE_TYPE.SPINE:
        if _FORMAT==NODE_FORMAT.DICT:
            IndexMax = len(Node["_DICT"].keys()) - 1
            if IndexMax == -1: # Empty Dict
                if isinstance(Key, int):
                    _AppendWithIndent(StrList, IndentNum, "{}")
                else:
                    _AppendWithIndent(StrList, 0, "{}") # include root node.
                _AddComment(Node, StrList)
                return
            if isinstance(Key, int):
                _AppendWithIndent(StrList, IndentNum, "{")
            else:
                _AppendWithIndent(StrList, 0, "{") # include root node.
            _AddComment(Node, StrList, COMMENT_TYPE.SPINE_BEFORE)
            StrList.append("\n")
            Index = 0
            for SubKey, SubNodes in Node["_DICT"].items():
                if SubKey != "_LEAF":
                    if isinstance(SubNodes, dict):
                        SubNodes = [SubNodes]
                    assert isinstance(SubNodes, list)
                    SubIndexMax = len(SubNodes) - 1
                    for SubIndex, SubNode in enumerate(SubNodes):
                        _AppendWithIndent(StrList, IndentNum + 1, f"\"{SubKey}\": ")
                        _Tree2JsonStrRecur(SubNode, StrList, IndentNum + 1, SubKey)
                        
                        if Index < IndexMax or SubIndex < SubIndexMax:
                            StrList.append(",")
                        else:
                            pass

                        if SubNode["_TYPE"] == NODE_TYPE.LEAF:
                            _AddComment(SubNode, StrList)
                        else:
                            _AddComment(SubNode, StrList, COMMENT_TYPE.SPINE_AFTER)
                        
                        StrList.append("\n")
                else:
                    _AppendWithIndent(StrList, IndentNum + 1, f"\"{SubKey}\": ")
                    StrList.append(ToJsonStr(SubNodes))
                    if Index < IndexMax:
                        StrList.append(",")
                    else:
                        pass
                    StrList.append("\n")
                Index += 1
            _AppendWithIndent(StrList, IndentNum, "}")
        elif _FORMAT==NODE_FORMAT.LIST:
            AllLeafNode = True
            for SubIndex, SubNode in Node["_LIST"].items():
                if SubNode["_TYPE"] != NODE_TYPE.LEAF:
                    AllLeafNode = False
            SubIndexMax = len(Node["_LIST"]) - 1
            if AllLeafNode:
                # to be implemented: if any leaf node has comment
                # to be implemented: too many leaf nodes
                _AppendWithIndent(StrList, 0, "[") # Include root node.
                for SubIndex, SubNode in Node["_LIST"].items():
                    _Tree2JsonStrRecur(SubNode, StrList, IndentNum + 1, SubIndex, NoIndent=True)
                    if SubIndex < SubIndexMax:
                        StrList.append(", ")
                    else:
                        pass
                _AppendWithIndent(StrList, 0, "]") # Include root node.
                return

            if SubIndexMax == -1:
                if isinstance(Key, int):
                    _AppendWithIndent(StrList, IndentNum, "[]")
                else:
                    StrList.append("[]")
                _AddComment(Node, StrList)
                return
            if isinstance(Key, int):
                _AppendWithIndent(StrList, IndentNum, "[")
            else:
                _AppendWithIndent(StrList, 0, "[") # Include root node.

            _AddComment(Node, StrList, COMMENT_TYPE.SPINE_BEFORE)
            StrList.append("\n")

            for SubIndex, SubNode in Node["_LIST"].items():
                _Tree2JsonStrRecur(SubNode, StrList, IndentNum + 1, SubIndex)
                if SubIndex < SubIndexMax:
                    StrList.append(",")
                else:
                    pass
                if SubNode["_TYPE"] == NODE_TYPE.LEAF:
                    AddedComment = _AddComment(SubNode, StrList)
                else:
                    AddedComment = _AddComment(SubNode, StrList, COMMENT_TYPE.SPINE_AFTER)
                StrList.append("\n")
            _AppendWithIndent(StrList, IndentNum, "]")
        else:
            raise Exception()
    elif _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
        raise Exception()
    else:
        raise Exception()

def _CompressTree(RootNode):
    _CompressTreeRecur(RootNode, None, None, None)

def _CompressTreeRecur(Node, SupNode, SupKey, SupKeyIndex=None):
    if SupKey=="A":
        a = 1

    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    if _TYPE == NODE_TYPE.LEAF:
        return
    if Node["_TYPE"] == NODE_TYPE.SPINE_WITH_LEAF:
        if SupKey=="A.B":
            a = 1
        if Node.get("_SPLIT_LEAF") is True:
            assert SupNode["_FORMAT"] == NODE_FORMAT.DICT
            assert Node["_TYPE"] == NODE_TYPE.SPINE_WITH_LEAF
            LeafNode = _NewNode(
                NODE_TYPE.LEAF, NODE_FORMAT.NULL,
                Node["_LEAF"], Node["_PATH_FROM_ROOT"]
            )

            NodeComments = Node.get("_COMMENT")
            if NodeComments is not None and len(NodeComments) > 0:
                LeafNodeComments = []
                _NodeComments = []
                for Comment in NodeComments:
                    if Comment[1] == COMMENT_TYPE.LEAF:
                        LeafNodeComments.append(Comment)
                    else:
                        _NodeComments.append(Comment)
                
                if len(LeafNodeComments) > 0:
                    LeafNode["_COMMENT"] = LeafNodeComments
                Node["_COMMENT"] = _NodeComments

            SupDict = SupNode["_DICT"]
            if isinstance(SupDict[SupKey], dict):
                SupDict[SupKey] = [SupDict[SupKey]]
            assert isinstance(SupDict[SupKey], list)
            SupDict[SupKey] = [LeafNode] + SupNode["_DICT"][SupKey]
        else:
            assert Node["_DICT"].get("_LEAF") is None
            Node["_DICT"]["_LEAF"] = Node["_LEAF"]

        Node["_TYPE"] = NODE_TYPE.SPINE
        Node["_LEAF"] = None
        
    if Node.get("_MERGE_SUBNODE") is True:
        assert _ChildNum(Node) == 1
        assert isinstance(SupKey, str)
        SubKey = list(Node["_DICT"].keys())[0]
        SubNodes = Node["_DICT"][SubKey]
        if isinstance(SubNodes, dict):
            SubNodes = [SubNodes]
        assert isinstance(SubNodes, list)
        assert SupKeyIndex is not None

        SupKeyMerge = SupKey + "." + SubKey
        SupDict = SupNode["_DICT"]
        if SupDict.get(SupKeyMerge) is None: # Insert key SupKeyMerge just after SupKey.
            SubDictItems = list(SupDict.items())
            SubDictItems.insert(
                list(SupDict.keys()).index(SupKey), 
                (SupKeyMerge, [])
            )
            SupNode["_DICT"] = dict(SubDictItems)

        SubIndexBase = len(SupNode["_DICT"][SupKeyMerge])
        
        if isinstance(SupNode["_DICT"][SupKeyMerge], dict):
            SupNode["_DICT"][SupKeyMerge] = [SupNode["_DICT"][SupKeyMerge]]
        SupNode["_DICT"][SupKeyMerge] += SubNodes
        if isinstance(SupNode["_DICT"][SupKey], dict):
            SupNode["_DICT"][SupKey] = [SupNode["_DICT"][SupKey]]
        SupNode["_DICT"][SupKey].remove(Node)
        if not SupNode["_DICT"][SupKey]: # Empty List
            SupNode["_DICT"].pop(SupKey, None)
        
        for SubIndex, SubNode in enumerate(SubNodes):
            _CompressTreeRecur(SubNode, SupNode, SupKeyMerge, SubIndexBase + SubIndex)
        return
    else:
        pass

    if _FORMAT==NODE_FORMAT.DICT:
        for SubKey, SubNodes in dict(Node["_DICT"]).items():
            if SubKey != "_LEAF":
                if isinstance(SubNodes, dict):
                    SubNodes = [SubNodes]
                assert isinstance(SubNodes, list)
                SubNodesBak = list(SubNodes)
                for SubIndex, SubNode in enumerate(SubNodesBak): # In-Loop deletion might occur.
                    _CompressTreeRecur(SubNode, Node, SubKey, SubIndex)
                if not SubNodes: # Empty List
                    Node["_DICT"].pop(SubKey, None)
    elif _FORMAT==NODE_FORMAT.LIST:
        for index, SubNode in dict(Node["_LIST"]).items():
            _CompressTreeRecur(SubNode, SupNode, index, None)
    else:
        raise Exception()
    return

def Param2JsonStr(Obj):
    Tree = Param2Tree(Obj)
    #JsonDict = _Tree2JsonDict(Tree)
    Str = Tree2JsonStr(Tree)
    return Str

def _GetOnlyChild(Node):
    Dict = Node["_DICT"]
    if isinstance(Dict, dict):
        assert len(Dict.values()) == 1
        return list(Dict.values())[0]
    else:
        assert len(Dict) == 1
        return list(Dict.values())[0][0]

def _DetectTreeCompress(RootNode):
    _DetectTreeCompressRecur(RootNode, None)

def _DetectTreeCompressRecur(Node, SupKey):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    _CHILD_NUM = _ChildNum(Node)
    if _TYPE == NODE_TYPE.LEAF:
        return Node
    if _TYPE == NODE_TYPE.SPINE_WITH_LEAF and isinstance(SupKey, str):
        Node["_SPLIT_LEAF"] = True

    if _CHILD_NUM == 1 and _FORMAT == NODE_FORMAT.DICT and isinstance(SupKey, str):
        if _GetOnlyChild(Node)["_TYPE"] in [NODE_TYPE.SPINE, NODE_TYPE.LEAF]:
            Node["_MERGE_SUBNODE"] = True
    
    if _FORMAT == NODE_FORMAT.DICT:
        for SubKey, SubNode in Node["_DICT"].items():
            _DetectTreeCompressRecur(SubNode, SubKey)
    elif _FORMAT == NODE_FORMAT.LIST:
        for SubIndex, SubNode in Node["_LIST"].items():
            _DetectTreeCompressRecur(SubNode, SubIndex)
    else:
        raise Exception()

def _Tree2JsonDictRecur(Node):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]
    if _TYPE == NODE_TYPE.LEAF:
        return ToJsonElement(Node["_LEAF"])
    else:
        if _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
            return
        if _TYPE==NODE_TYPE.SPINE:
            if _FORMAT==NODE_FORMAT.DICT:
                Dict = {}
                for key, SubNode in Node["_DICT"].items():
                    if _HasLeaf(SubNode):
                        keys = ".".join(keys)
                        Dict[keys] = SubNode["_LEAF"]
                        break
                    keys.append(key)
                
            elif _FORMAT==NODE_FORMAT.LIST:
                List = []
                for index, SubNode in enumerate(Node["_LIST"]):
                    List.append(_Tree2JsonDictRecur(SubNode))
                return List
            else:
                raise Exception()
            return
        raise Exception()
    return

def Param2Tree(Obj):
    assert isinstance(Obj, param)
    Tree = Param2TreeRecur(Obj, None, None)
    return Tree

def Param2TreeRecur(Obj, SupNode, SupNodeKey):
    if isinstance(Obj, list) or isinstance(Obj, dict):
        Obj = DLUtils.Param(Obj)
        # SupNode["_DICT"].pop(SupNodeKey)
        # SupNode["_DICT"][SupNodeKey] = Obj

    if isinstance(Obj, param):
        _TYPE = Obj.GetAttr("_TYPE")
        _FORMAT = Obj.GetAttr("_FORMAT")
        _PATH_FROM_ROOT = Obj.GetAttr("_PATH_FROM_ROOT")
        Node = _NewNode(
            _TYPE, _FORMAT, None, _PATH_FROM_ROOT
        )
        if Obj.HasAttr("_COMMENT"):
            Node["_COMMENT"] = Obj.GetAttr("_COMMENT")
        
        if _TYPE in [NODE_TYPE.LEAF, NODE_TYPE.SPINE_WITH_LEAF]:
            _LEAF = Obj.GetAttr("_LEAF")
            Node["_LEAF"] = _LEAF
        if _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
            if _FORMAT==NODE_FORMAT.DICT:
                for Key, SubNode in Obj.GetAttr("_DICT").items():
                    Node["_DICT"][Key] = Param2TreeRecur(SubNode, Node, Key)
            elif _FORMAT==NODE_FORMAT.LIST:
                for Index, SubNode in enumerate(Obj.GetAttr("_LIST")):
                    Node["_LIST"][Index] = Param2TreeRecur(SubNode, Node, Index)
            else:
                raise Exception(_FORMAT)
        
        if Obj.HasAttr("_COMMENT_DICT"):
            CommentDict = Obj.GetAttr("_COMMENT_DICT")
            if _FORMAT == NODE_FORMAT.DICT:
                for Key, Comment in CommentDict.items():       
                    Node["_DICT"][Key]["_COMMENT"] = Comment
            else:
                for Index, Comment in CommentDict.items():
                    Node["_LIST"][Index]["_COMMENT"] = Comment

        return Node

    else:
        return _NewNode(
            NODE_TYPE.LEAF,
            NODE_FORMAT.NULL,
            Obj,
            SupNode["_PATH_FROM_ROOT"] + [SupNodeKey]
        )

def JsonStyleObj2Param(Obj):
    PathTable = _JsonLike2Paths(Obj)

    if isinstance(Obj, dict):
        Tree = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _FORMAT=NODE_FORMAT.DICT,
            _PATH_FROM_ROOT=["ROOT"]
        )
    elif isinstance(Obj, list):
        Tree = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _FORMAT=NODE_FORMAT.LIST,
            _PATH_FROM_ROOT=["ROOT"]
        )
    Tree = _Paths2Tree(PathTable, Tree)
    return _Tree2Param(Tree)

class _SymbolEmptyList():
    pass

class _SymbolEmptyDict():
    pass

def _JsonLike2Paths(Obj):
    PathTable = []
    _JsonLike2PathsRecur(Obj, PathTable, ["ROOT"])
    return PathTable

def Tree2Paths(Root, SplitKeyByDot=True, SplitKeyException=[]):
    PathTable = []
    _Tree2PathsRecur(Root, PathTable, ["ROOT"], SplitKeyByDot=SplitKeyByDot, SplitKeyException=[])
    return PathTable

class PATH_TYPE(Enum):
    LEAF, COMMENT = range(2)

def _Tree2PathsRecur(Node, PathTable, PathCurrent, **Dict):
    SplitKeyByDot = Dict.setdefault("SplitKeyByDot", True)
    SplitKeyException = Dict.setdefault("SplitKeyException", [])
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]

    Comment = Node.get("_COMMENT")
    if Comment is not None:
        assert isinstance(Comment, list)
        if len(Comment) > 0:
            PathTable.append([PathCurrent, Comment, PATH_TYPE.COMMENT])
    if _TYPE == NODE_TYPE.SPINE:
        if _FORMAT == NODE_FORMAT.DICT:
            _DICT = Node["_DICT"]
            if len(_DICT.keys()) == 0:
                PathTable.append(
                    [PathCurrent, _SymbolEmptyDict(), PATH_TYPE.LEAF]
                )
                return
            for Keys, SubNodes in _DICT.items():
                if SplitKeyByDot:
                    KeyList = Keys.split(".")
                else:
                    KeyList = [Keys]
                if isinstance(SubNodes, dict):
                    SubNodes = [SubNodes]
                for Index, SubNode in enumerate(SubNodes):
                    # assert isinstance(Key, str)
                    _Tree2PathsRecur(
                        SubNode,
                        PathTable,
                        PathCurrent = PathCurrent + KeyList,
                        **Dict
                    )
        elif _FORMAT == NODE_FORMAT.LIST:
            _LIST = Node["_LIST"]
            if len(_LIST) == 0:
                PathTable.append([PathCurrent, _SymbolEmptyList(), PATH_TYPE.LEAF])
                return
            if isinstance(_LIST, list):
                for Index, SubNode in enumerate(_LIST):
                    _Tree2PathsRecur(SubNode, PathTable, PathCurrent + [Index], **Dict)
            elif isinstance(_LIST, dict):
                for Index, SubNode in _LIST.items():
                    _Tree2PathsRecur(SubNode, PathTable, PathCurrent + [Index], **Dict)
            else:
                raise Exception()
        else:
            raise Exception()
    elif _TYPE == NODE_TYPE.LEAF:
        PathTable.append(
            [PathCurrent, Node["_LEAF"], PATH_TYPE.LEAF]
        )
    else:
        raise Exception()

def _JsonLike2PathsRecur(Obj, PathTable, PathCurrent):
    if isinstance(Obj, dict):
        if len(Obj.keys()) == 0:
            PathTable.append(
                [PathCurrent, _SymbolEmptyDict(), PATH_TYPE.LEAF]
            )
            return
        for Keys, Value in Obj.items():
            assert isinstance(Keys, str)
            KeyList = Keys.split(".")
            _JsonLike2PathsRecur(
                Obj[Keys],
                PathTable,
                PathCurrent = PathCurrent + KeyList
            )
    elif isinstance(Obj, list):
        if len(Obj) == 0:
            PathTable.append(
                [PathCurrent, _SymbolEmptyList(), PATH_TYPE.LEAF]
            )
        for Index, Value in enumerate(Obj):
            _JsonLike2PathsRecur(
                Obj[Index], PathTable, PathCurrent + [Index]
            )
    else:
        PathTable.append(
            [PathCurrent, Obj, PATH_TYPE.LEAF]
        )

def _GetNodeKey(Node, Key):
    if isinstance(Key, int):
        assert Node["_FORMAT"] in [NODE_FORMAT.LIST, NODE_FORMAT.COMMENT]
        Node["_FORMAT"] == NODE_FORMAT.LIST
        if Key in Node["_LIST"].keys():
            return Node["_LIST"][Key]
    elif isinstance(Key, str):
        assert Node["_FORMAT"] in [NODE_FORMAT.DICT, NODE_FORMAT.COMMENT], Node
        Node["_FORMAT"] = NODE_FORMAT.DICT
        if Key in Node["_DICT"].keys():
            return Node["_DICT"][Key]
    else:
        raise Exception()
    return None

def _Paths2Tree(PathTable, RootNode=None):
    if RootNode is None:
        if len(PathTable) == 1:
            if isinstance(PathTable[0][1], _SymbolEmptyDict)or isinstance(PathTable[0][1], str):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _FORMAT=NODE_FORMAT.DICT,
                    _PATH_FROM_ROOT=["ROOT"]
                )
            elif isinstance(PathTable[0][1], _SymbolEmptyList) or isinstance(PathTable[0][1], int):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _FORMAT=NODE_FORMAT.LIST,
                    _PATH_FROM_ROOT=["ROOT"]
                )
            else:
                raise Exception()
        elif len(PathTable) > 1:
            PathItem = PathTable[0][0]
            KeyExample = PathItem[1]
            if isinstance(KeyExample, str):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _FORMAT=NODE_FORMAT.DICT,
                    _PATH_FROM_ROOT=["ROOT"]
                )
            elif isinstance(KeyExample, int):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _FORMAT=NODE_FORMAT.LIST,
                    _PATH_FROM_ROOT=["ROOT"]
                )
            else:
                raise Exception()
        else:
            pass # to be implemented
    for Path, Leaf, _PATH_TYPE in PathTable:
        Node = RootNode
        PathLength = len(Path)
        # Dealing with keys that are not last.
        if PathLength == 1:
            assert Path[0] == 'ROOT'
            if _PATH_TYPE == PATH_TYPE.COMMENT:
                RootNode["_COMMENT"] = Leaf
                continue
            else:
                assert isinstance(Leaf, _SymbolEmptyDict) or isinstance(Leaf, _SymbolEmptyList)
                continue
        Index = 1
        for Key in Path[1:-1]: #Path with ROOT.
            NextKey = Path[Index + 1]
            SubNode = _GetNodeKey(Node, Key)
            if SubNode is not None:
                # if(SubNode["_PATH_FROM_ROOT"]==['ROOT', 0, 'A', 'B', 'C']):
                #     a = 1
                _TYPE = SubNode["_TYPE"]
                if _TYPE == NODE_TYPE.LEAF:
                    SubNode["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
                    SubNode["_FORMAT"] = NODE_FORMAT.LIST if isinstance(NextKey, int) else NODE_FORMAT.DICT
                elif _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
                    pass
                elif _TYPE == NODE_TYPE.COMMENT:
                    if _PATH_TYPE == PATH_TYPE.COMMENT:
                        pass
                    elif _PATH_TYPE == PATH_TYPE.LEAF:
                        SubNode["_TYPE"] = NODE_TYPE.SPINE
                    else:
                        raise Exception()
                else:
                    raise Exception()
                Node = SubNode
            else:
                if _PATH_TYPE == PATH_TYPE.LEAF:
                    NodeType = NODE_TYPE.SPINE
                elif _PATH_TYPE == PATH_TYPE.COMMENT:
                    NodeType = NODE_TYPE.COMMENT
                else:
                    raise Exception()
                SubNodeType = NODE_FORMAT.LIST if isinstance(NextKey, int) else NODE_FORMAT.DICT
                Node = _SetNodeKey(
                    Node, Key,
                    _NewNode(
                        NodeType,
                        SubNodeType,
                        _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key]
                    )
                )

            Index += 1
        # Dealing with last key.   
        Index, Key = PathLength - 1, Path[-1]
        SubNode = _GetNodeKey(Node, Key)

        if SubNode is not None:
            _TYPE = SubNode["_TYPE"]
            if _PATH_TYPE == PATH_TYPE.LEAF:
                if _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
                    if _TYPE == NODE_TYPE.SPINE_WITH_LEAF:
                        #WarningStr = f"Overwriting: Path: {SubNode['_PATH_FROM_ROOT']} Value:{SubNode['_LEAF']} with {Leaf}\n"
                        #warnings.warn(WarningStr)
                        pass
                    SubNode["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
                    SubNode["_LEAF"] = Leaf
                elif _TYPE == NODE_TYPE.LEAF:
                    #WarningStr = f"Overwriting: Path: {SubNode['_PATH_FROM_ROOT']} Value:{SubNode['_LEAF']} with {Leaf}\n"
                    #warnings.warn(WarningStr)
                    pass
                    SubNode["_LEAF"] = Leaf
                elif _TYPE == NODE_TYPE.COMMENT:
                    SubNode["_TYPE"] = NODE_TYPE.LEAF
                    SubNode["_LEAF"] = Leaf
                else:
                    raise Exception()
            elif _PATH_TYPE == PATH_TYPE.COMMENT:
                SubNode.setdefault("_COMMENT", [])
                SubNode["_COMMENT"] += Leaf
            else:
                raise Exception()
        else:
            if _PATH_TYPE == PATH_TYPE.LEAF:
                if isinstance(Leaf, _SymbolEmptyDict):
                    _SetNodeKey(
                        Node, Key,
                        _NewNode(
                            NODE_TYPE.SPINE,
                            NODE_FORMAT.DICT,
                            _LEAF=Leaf,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                        )
                    )
                elif isinstance(Leaf, _SymbolEmptyList):
                    _SetNodeKey(
                        Node, Key,
                        _NewNode(
                            NODE_TYPE.SPINE,
                            NODE_FORMAT.LIST,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                            _LEAF=Leaf
                        )
                    )
                else:
                    _SetNodeKey(
                        Node, Key,
                        _NewNode(
                            NODE_TYPE.LEAF,
                            NODE_FORMAT.NULL,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                            _LEAF=Leaf
                        )
                    )
            elif _PATH_TYPE == PATH_TYPE.COMMENT:
                    NewNode = _NewNode(
                            NODE_TYPE.COMMENT,
                            NODE_FORMAT.COMMENT,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                            _LEAF=None
                        )
                    NewNode["_COMMENT"] = Leaf
                    _SetNodeKey(Node, Key, NewNode) 
            else:
                raise Exception()
    return RootNode

def _Tree2Param(Tree):
    return _Tree2ParamRecur(Tree)

def _Tree2ParamRecur(Node):
    _TYPE = Node["_TYPE"]
    _FORMAT = Node["_FORMAT"]

    if _TYPE==NODE_TYPE.LEAF:
        return Node["_LEAF"]
    elif _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
        Obj = Param(Dict=Node)
        if _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
            Obj.SetAttr("_LEAF", Node["_LEAF"])
        
        Comment = Node.get("_COMMENT")
        if Comment is not None:
            Obj.SetAttr("_COMMENT", Node["_COMMENT"])

        if _FORMAT==NODE_FORMAT.NULL:
            raise Exception()
        
        elif _FORMAT==NODE_FORMAT.DICT:
            CommentDict = {}
            for Key, SubNode in Node["_DICT"].items():
                SubNodeComment = SubNode.get("_COMMENT")
                
                if SubNode["_TYPE"] == NODE_TYPE.LEAF and SubNodeComment is not None:
                    CommentDict[Key] = SubNodeComment
                setattr(    
                    Obj, Key, _Tree2ParamRecur(SubNode)  
                )
            if len(CommentDict) > 0:
                Obj.SetAttr("_COMMENT_DICT", CommentDict)
            return Obj

        elif _FORMAT==NODE_FORMAT.LIST:
            List = Node["_LIST"]
            if isinstance(List, dict):
                IndexList = Node["_LIST"].keys()
                Obj.SetAttr("_LIST", [None for Index in range(len(IndexList))])
                ObjList = Obj.GetAttr("_LIST")

                for Index in IndexList:
                    SubNode = Node["_LIST"][Index]
                    SubNodeComment = SubNode.get("_COMMENT")
                    CommentDict = {}
                    if SubNode["_TYPE"]==NODE_TYPE.LEAF and SubNodeComment is not None:
                        CommentDict[Index] = SubNodeComment
                    if len(CommentDict) > 0:
                        Obj.SetAttr("_COMMENT_DICT", CommentDict)

                    ObjList[Index] = _Tree2ParamRecur(SubNode)
            elif isinstance(List, list):
                ObjList = []
                Obj.SetAttr("_LIST", ObjList)
                CommentDict = {}
                for Index, SubNode in enumerate(List):
                    SubNodeComment = SubNode.get("_COMMENT")
                    if SubNode["_TYPE"]==NODE_TYPE.LEAF and SubNodeComment is not None:
                        CommentDict[Index] = SubNodeComment
                    ObjList.append(SubNode)
                if len(CommentDict) > 0:
                    Obj.SetAttr("_COMMENT_DICT", CommentDict)
            else:
                raise Exception()
        else:
            raise Exception()
        return Obj
    else:
        raise Exception()

def _AnalyzeTree(Root):
    Root["_PATH_FROM_ROOT"] = ["ROOT"]
    _AnalyzeTreeRecur(Root)

def _AnalyzeTreeRecur(Node, Key=None):
    _PARSE_TYPE = Node["_TYPE"]
    if _PARSE_TYPE == NODE_PARSE_TYPE.DICT:
        Node["_FORMAT"] = NODE_FORMAT.DICT
        Dict = Node["_DICT"]
        Node["_TYPE"] = NODE_TYPE.SPINE
        for Key, SubNodes in Dict.items():
            for Index, SubNode in enumerate(SubNodes):
                SubNode["_PATH_FROM_ROOT"] = Node["_PATH_FROM_ROOT"] + [Key]
                _AnalyzeTreeRecur(SubNode, Key)
    elif _PARSE_TYPE == NODE_PARSE_TYPE.LIST:
        Node["_FORMAT"] = NODE_FORMAT.LIST
        List = Node["_LIST"]
        Node["_TYPE"] = NODE_TYPE.SPINE
        if isinstance(List, dict):
            for Index, SubNode in List.items():
                SubNode["_PATH_FROM_ROOT"] = Node["_PATH_FROM_ROOT"] + [Index]
                _AnalyzeTreeRecur(SubNode, Index)
        elif isinstance(List, list):
            for Index, SubNode in enumerate(List):
                SubNode["_PATH_FROM_ROOT"] = Node["_PATH_FROM_ROOT"] + [Index]
                _AnalyzeTreeRecur(SubNode, Index)
        else:
            raise Exception()
    elif _PARSE_TYPE == NODE_PARSE_TYPE.LEAF:
        Node["_TYPE"] = NODE_TYPE.LEAF
        Node["_FORMAT"] = NODE_FORMAT.NULL
    else:
        raise Exception()

def JsonFile2Tree(FilePath):
    JsonStr = DLUtils.file.File2Str(FilePath)
    if not JsonStr.endswith("\n"):
        JsonStr += "\n"
    Tree = _JsonStr2Tree(JsonStr)
    if Tree is None:
        raise Exception("Failed to parse json file.")
    _AnalyzeTree(Tree)    
    return Tree

def JsonFile2Param(FilePath, SplitKeyByDot=True, SplitKeyException=[]):
    Tree = JsonFile2Tree(FilePath)
    assert Tree["_TYPE"] == NODE_TYPE.SPINE
    PathTable = Tree2Paths(Tree, SplitKeyByDot=SplitKeyByDot, SplitKeyException=[])

    if Tree["_FORMAT"] == NODE_FORMAT.DICT:
        RootNode = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _FORMAT=NODE_FORMAT.DICT,
            _PATH_FROM_ROOT=["ROOT"]
        )
    elif Tree["_FORMAT"] == NODE_FORMAT.LIST:
        RootNode = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _FORMAT=NODE_FORMAT.LIST,
            _PATH_FROM_ROOT=["ROOT"]
        )   
    else:
        raise Exception()

    Tree = _Paths2Tree(PathTable, RootNode)
    return _Tree2Param(Tree)