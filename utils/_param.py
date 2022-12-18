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

class NODE_SUBTYPE(Enum):
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

class param():
    def __init__(self, _CONTENT=None, Dict=None, _PATH_FROM_ROOT=["ROOT"], _TYPE=NODE_TYPE.SPINE, _SUBTYPE=NODE_SUBTYPE.DICT, IsSuper=False):  
        self.SetAttr("_LIST", [])
        self.SetAttr("_DICT", {})
        if Dict is not None:
            self.SetAttr("_SUBTYPE", Dict["_SUBTYPE"])
            self.SetAttr("_LEAF", Dict.get("_LEAF"))
        else:
            self.SetAttr("_TYPE", _TYPE)
            self.SetAttr("_SUBTYPE", _SUBTYPE)
            self.SetAttr("_PATH_FROM_ROOT", _PATH_FROM_ROOT)
        
        if not IsSuper:
            if _CONTENT is not None:
                if isinstance(_CONTENT, dict):
                    self.SetAttr("_SUBTYPE", NODE_SUBTYPE.DICT)
                    self.from_dict(_CONTENT)
                elif isinstance(_CONTENT, list):
                    self.SetAttr("_SUBTYPE", NODE_SUBTYPE.LIST)
                    self.from_list(_CONTENT)
                else:
                    raise Exception()   
    def from_dict(self, Dict):
        self.SetAttr("_SUBTYPE", NODE_SUBTYPE.DICT)
        self.SetAttr("_DICT", {})
        self.absorb_dict(Dict)
        return self
    def absorb_dict(self, Dict):
        # assert isinstance(_DICT, dict)
        _DICT = self.GetAttr("_DICT")
        for key, item in Dict.items():
            if isinstance(item, dict):
                _DICT[key] = param(_SUBTYPE=NODE_SUBTYPE.DICT).from_dict(item)
            elif isinstance(item, list):
                _DICT[key] = param(_SUBTYPE=NODE_SUBTYPE.LIST).from_list(item)
            else:
                _DICT[key] = item
        return self
    def from_list(self, List):
        _LIST = self.GetAttr("_LIST")
        for index, item in enumerate(List):
            if isinstance(item, dict):
                _LIST.append(param(_SUBTYPE=NODE_SUBTYPE.DICT).from_dict(item))
            elif isinstance(item, list):
                _LIST[index].append(param(_SUBTYPE=NODE_SUBTYPE.LIST).from_list(item))
            else:
                _LIST[index].append(item)
        return self
    def Get(self, Key):
        if Key in self.__dict__:
            return self.__dict__[Key]
        else:
            return None
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
        if not Key in self._DICT:
            self._DICT[Key] = Value
        return self._DICT[Key]
    def getdefault(self, Key, DefaultValue):
        return self._DICT.get(Key)
    def items(self):
        return self.GetAttr("_DICT").items()
    def append(self, Item):
        self._LIST.append(Item)
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
    def __getitem__(self, Index):
        return self._LIST[Index]
    def ToJsonFile(self, FilePath):
        Param2JsonFile(self, FilePath)
        return self
    def ToFile(self, FilePath):
        DLUtils.file.Obj2BinaryFile(self, FilePath)
        return self

class Param(param):
    # tree node class for representing json-like structure.
    # content information is stored in attribute _DICT, or _LIST, _LEAF, depending on node type.
    # meta information is stored in other attributes such as _TYPE, _SUBTYPE etc.
    # viewed externally, node acts like an object with its attributes storing content information.
        # content could be accessed using . or [] operator
        # this is implemented using magic methods of python. 
    # method names with UpperCamelCase typically indicates functions to manipulate meta information
    # method names with   lowercases   typically indicates functions to manipulate content information.
    def __init__(self, _CONTENT=None, Dict=None, _PATH_FROM_ROOT=["ROOT"], _TYPE=NODE_TYPE.SPINE, _SUBTYPE=NODE_SUBTYPE.DICT):
        super().__init__(_CONTENT=_CONTENT, Dict=Dict, _SUBTYPE=_SUBTYPE, IsSuper=True)
        self.SetAttr("_DICT_TEMP", {})
        if Dict is not None:
            self.SetAttr("_TYPE", Dict["_TYPE"])
            self.SetAttr("_PATH_FROM_ROOT", Dict["_PATH_FROM_ROOT"])
        else:
            self.SetAttr("_TYPE", _TYPE)
            self.SetAttr("_PATH_FROM_ROOT", _PATH_FROM_ROOT)
        if _CONTENT is not None:
            if isinstance(_CONTENT, dict):
                self.SetAttr("_SUBTYPE", NODE_SUBTYPE.DICT)
                self.from_dict(_CONTENT)
            elif isinstance(_CONTENT, list):
                self.SetAttr("_SUBTYPE", NODE_SUBTYPE.LIST)
                self.from_list(_CONTENT)
            else:
                raise Exception()   
    def FromParam(self, Node):
        for Key, Item in Node.__dict__.items():
            self.SetAttr(Key, Item)
        #self.SetAttr("__dict__", Node.__dict__)
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
    def SubstantiateAlongSpine(self, _TYPE, _SUBTYPE=NODE_SUBTYPE.DICT):
        Node = self
        Index = 0
        while Node.Get("_IS_FALSE") is True:
            Node.DelAttr("_IS_FALSE")
            if Index == 0:
                Node.SetAttr("_TYPE", _TYPE)
                Node.SetAttr("_SUBTYPE", _SUBTYPE)
            else:
                Node.SetAttr("_TYPE", NODE_TYPE.SPINE)
                Node.SetAttr("_SUBTYPE", NODE_SUBTYPE.DICT)
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
    def setattr(self, Key, Value):
        self._DICT[Key] = Value
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_SUBTYPE.DICT)
        return self
    def append(self, Item):
        self._LIST.append(Item)
        if isinstance(Item, param):
            pass
        return self
    def __setattr__(self, Key, Value):
        self._DICT[Key] = Value
        if self.Get("_IS_FALSE") is True:
            self.SubstantiateAlongSpine(NODE_TYPE.SPINE, NODE_SUBTYPE.DICT)
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
        _SUBTYPE=NODE_SUBTYPE.NULL,
        _LEAF=None,
        _PATH_FROM_ROOT=None
    ):
    assert not isinstance(_PATH_FROM_ROOT, int)
    return {
        "_TYPE": _TYPE,
        "_SUBTYPE": _SUBTYPE,
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
    _SUBTYPE = Node["_SUBTYPE"]
    if _TYPE==NODE_TYPE.LEAF:
        return 0
    else:
        if _SUBTYPE==NODE_SUBTYPE.DICT:
            return len(Node["_DICT"].keys())
        elif _SUBTYPE==NODE_SUBTYPE.LIST:
            NodeList = Node["_LIST"]
            if isinstance(Node["_LIST"], dict):
                return len(NodeList.keys())
            return len(Node["_LIST"])
        else:
            raise Exception()

def _HasLeaf(Node):
    return Node["_TYPE"] in [NODE_TYPE.LEAF, NODE_TYPE.SPINE_WITH_LEAF]

def _IsDict(Node):
    return Node["_SUBTYPE"]==NODE_SUBTYPE.DICT and Node["_TYPE"]!=NODE_TYPE.LEAF

def _IsList(Node):
    return Node["_SUBTYPE"]==NODE_SUBTYPE.LIST and Node["_TYPE"]!=NODE_TYPE.LEAF

def _IsSpine(Node):
    _TYPE = Node["_TYPE"]
    return _TYPE==NODE_TYPE.SPINE or _TYPE==NODE_TYPE.SPINE_WITH_LEAF

def _ChildNum(Node):
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]
    # assert _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]
    if _TYPE == NODE_TYPE.LEAF:
        return 0
    if _SUBTYPE==NODE_SUBTYPE.DICT:
        return len(Node["_DICT"].keys())
    elif _SUBTYPE==NODE_SUBTYPE.LIST:
        return len(Node["_LIST"])
    else:
        warnings.warn("_SUBTYPE:{_SUBTYPE}")
        return 0

def Tree2JsonStr(RootNode):
    _DetectTreeCompress(RootNode)
    _CompressTree(RootNode)
    return _Tree2JsonStr(RootNode)

import json
def ToJsonStr(Obj):
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
    if CommentList is not None:
        assert isinstance(CommentList, list)
        if _TYPE is None:
            for Comment in CommentList:
                StrList.append("\t")
                StrList += Comment[0]
        else:
            for Comment in CommentList:
                if Comment[1] == _TYPE:
                    StrList.append("\t")
                    StrList += Comment[0]

def _Tree2JsonStrRecur(Node, StrList, IndentNum, Key):
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]
    if _TYPE==NODE_TYPE.LEAF:
        if isinstance(Key, int):
            _AppendWithIndent(StrList, IndentNum, ToJsonStr(Node["_LEAF"]))
        else:
            StrList.append(ToJsonStr(Node["_LEAF"]))
        
    elif _TYPE==NODE_TYPE.SPINE:
        if _SUBTYPE==NODE_SUBTYPE.DICT:
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
        elif _SUBTYPE==NODE_SUBTYPE.LIST:
            SubIndexMax = len(Node["_LIST"]) - 1
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
                    _AddComment(SubNode, StrList)
                else:
                    _AddComment(SubNode, StrList, COMMENT_TYPE.SPINE_AFTER)
                
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
    _SUBTYPE = Node["_SUBTYPE"]
    if _TYPE == NODE_TYPE.LEAF:
        return
    if Node["_TYPE"] == NODE_TYPE.SPINE_WITH_LEAF:
        if SupKey=="A.B":
            a = 1
        if Node.get("_SPLIT_LEAF") is True:
            assert SupNode["_SUBTYPE"] == NODE_SUBTYPE.DICT
            assert Node["_TYPE"] == NODE_TYPE.SPINE_WITH_LEAF
            LeafNode = _NewNode(
                NODE_TYPE.LEAF, NODE_SUBTYPE.NULL,
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
        if SupKey == "B":
            a = 1
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

    if _SUBTYPE==NODE_SUBTYPE.DICT:
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
    elif _SUBTYPE==NODE_SUBTYPE.LIST:
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
    _SUBTYPE = Node["_SUBTYPE"]
    _CHILD_NUM = _ChildNum(Node)
    if _TYPE == NODE_TYPE.LEAF:
        return Node
    if _TYPE == NODE_TYPE.SPINE_WITH_LEAF and isinstance(SupKey, str):
        Node["_SPLIT_LEAF"] = True

    if _CHILD_NUM == 1 and _SUBTYPE == NODE_SUBTYPE.DICT and isinstance(SupKey, str):
        if _GetOnlyChild(Node)["_TYPE"] in [NODE_TYPE.SPINE, NODE_TYPE.LEAF]:
            Node["_MERGE_SUBNODE"] = True
    
    if _SUBTYPE == NODE_SUBTYPE.DICT:
        for SubKey, SubNode in Node["_DICT"].items():
            _DetectTreeCompressRecur(SubNode, SubKey)
    elif _SUBTYPE == NODE_SUBTYPE.LIST:
        for SubIndex, SubNode in Node["_LIST"].items():
            _DetectTreeCompressRecur(SubNode, SubIndex)
    else:
        raise Exception()

def _Tree2JsonDictRecur(Node):
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]
    if _TYPE == NODE_TYPE.LEAF:
        return ToJsonElement(Node["_LEAF"])
    else:
        if _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
            return
        if _TYPE==NODE_TYPE.SPINE:
            if _SUBTYPE==NODE_SUBTYPE.DICT:
                Dict = {}
                for key, SubNode in Node["_DICT"].items():
                    if _HasLeaf(SubNode):
                        keys = ".".join(keys)
                        Dict[keys] = SubNode["_LEAF"]
                        break
                    keys.append(key)
                
            elif _SUBTYPE==NODE_SUBTYPE.LIST:
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
    if isinstance(Obj, param):
        _TYPE = Obj.GetAttr("_TYPE")
        _SUBTYPE = Obj.GetAttr("_SUBTYPE")
        _PATH_FROM_ROOT = Obj.GetAttr("_PATH_FROM_ROOT")
        Node = _NewNode(
            _TYPE, _SUBTYPE, None, _PATH_FROM_ROOT
        )
        if Obj.HasAttr("_COMMENT"):
            Node["_COMMENT"] = Obj.GetAttr("_COMMENT")
        
        if _TYPE in [NODE_TYPE.LEAF, NODE_TYPE.SPINE_WITH_LEAF]:
            _LEAF = Obj.GetAttr("_LEAF")
            Node["_LEAF"] = _LEAF
        if _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
            if _SUBTYPE==NODE_SUBTYPE.DICT:
                for Key, SubNode in Obj.GetAttr("_DICT").items():
                    Node["_DICT"][Key] = Param2TreeRecur(SubNode, Node, Key)
            elif _SUBTYPE==NODE_SUBTYPE.LIST:
                for Index, SubNode in enumerate(Obj.GetAttr("_LIST")):
                    Node["_LIST"][Index] = Param2TreeRecur(SubNode, Node, Index)
            else:
                raise Exception(_SUBTYPE)
        
        if Obj.HasAttr("_COMMENT_DICT"):
            CommentDict = Obj.GetAttr("_COMMENT_DICT")
            if _SUBTYPE == NODE_SUBTYPE.DICT:
                for Key, Comment in CommentDict.items():       
                    Node["_DICT"][Key]["_COMMENT"] = Comment
            else:
                for Index, Comment in CommentDict.items():
                    Node["_LIST"][Index]["_COMMENT"] = Comment

        return Node
    else:
        return _NewNode(
            NODE_TYPE.LEAF,
            NODE_SUBTYPE.NULL,
            Obj,
            SupNode["_PATH_FROM_ROOT"] + [SupNodeKey]
        )

def JsonStyleObj2Param(Obj):
    PathTable = _JsonLike2Paths(Obj)

    if isinstance(Obj, dict):
        Tree = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.DICT,
            _PATH_FROM_ROOT=["ROOT"]
        )
    elif isinstance(Obj, list):
        Tree = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.LIST,
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

def Tree2Paths(Root, SplitKeyByDot=True):
    PathTable = []
    _Tree2PathsRecur(Root, PathTable, ["ROOT"], SplitKeyByDot=SplitKeyByDot)
    return PathTable

class PATH_TYPE(Enum):
    LEAF, COMMENT = range(2)

def _Tree2PathsRecur(Node, PathTable, PathCurrent, **Dict):
    SplitKeyByDot = Dict.setdefault("SplitKeyByDot", True)
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]

    Comment = Node.get("_COMMENT")
    if Comment is not None:
        assert isinstance(Comment, list)
        if len(Comment) > 0:
            PathTable.append([PathCurrent, Comment, PATH_TYPE.COMMENT])
    if _TYPE == NODE_TYPE.SPINE:
        if _SUBTYPE == NODE_SUBTYPE.DICT:
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
        elif _SUBTYPE == NODE_SUBTYPE.LIST:
            _LIST = Node["_LIST"]
            if len(_LIST) == 0:
                PathTable.append(
                    [PathCurrent, _SymbolEmptyList(), PATH_TYPE.LEAF]
                )
                return
            for Index, SubNode in enumerate(_LIST):
                _Tree2PathsRecur(
                    SubNode, PathTable, PathCurrent + [Index], **Dict
                )
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
        assert Node["_SUBTYPE"] in [NODE_SUBTYPE.LIST, NODE_SUBTYPE.COMMENT]
        Node["_SUBTYPE"] == NODE_SUBTYPE.LIST
        if Key in Node["_LIST"].keys():
            return Node["_LIST"][Key]
    elif isinstance(Key, str):
        assert Node["_SUBTYPE"] in [NODE_SUBTYPE.DICT, NODE_SUBTYPE.COMMENT], Node
        Node["_SUBTYPE"] = NODE_SUBTYPE.DICT
        if Key in Node["_DICT"].keys():
            return Node["_DICT"][Key]
    else:
        raise Exception()
    return None

def _Paths2Tree(PathTable, RootNode=None):
    if RootNode is None:
        if len(PathTable) > 0:
            KeyExample = PathTable[0][0][1]
            if isinstance(KeyExample, str):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _SUBTYPE=NODE_SUBTYPE.DICT,
                    _PATH_FROM_ROOT=["ROOT"]
                )
            elif isinstance(KeyExample, int):
                RootNode = _NewNode(
                    _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
                    _SUBTYPE=NODE_SUBTYPE.LIST,
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
            assert _PATH_TYPE == PATH_TYPE.COMMENT
            RootNode["_COMMENT"] = Leaf
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
                    SubNode["_SUBTYPE"] = NODE_SUBTYPE.LIST if isinstance(NextKey, int) else NODE_SUBTYPE.DICT
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
                SubNodeType = NODE_SUBTYPE.LIST if isinstance(NextKey, int) else NODE_SUBTYPE.DICT
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
                            NODE_SUBTYPE.DICT,
                            _LEAF=Leaf,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                        )
                    )
                elif isinstance(Leaf, _SymbolEmptyList):
                    _SetNodeKey(
                        Node, Key,
                        _NewNode(
                            NODE_TYPE.SPINE,
                            NODE_SUBTYPE.LIST,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                            _LEAF=Leaf
                        )
                    )
                else:
                    _SetNodeKey(
                        Node, Key,
                        _NewNode(
                            NODE_TYPE.LEAF,
                            NODE_SUBTYPE.NULL,
                            _PATH_FROM_ROOT=Node["_PATH_FROM_ROOT"] + [Key],
                            _LEAF=Leaf
                        )
                    )
            elif _PATH_TYPE == PATH_TYPE.COMMENT:
                    NewNode = _NewNode(
                            NODE_TYPE.COMMENT,
                            NODE_SUBTYPE.COMMENT,
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
    _SUBTYPE = Node["_SUBTYPE"]

    if _TYPE==NODE_TYPE.LEAF:
        return Node["_LEAF"]
    elif _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
        Obj = Param(Dict=Node)
        if _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
            Obj.SetAttr("_LEAF", Node["_LEAF"])
        
        Comment = Node.get("_COMMENT")
        if Comment is not None:
            Obj.SetAttr("_COMMENT", Node["_COMMENT"])

        if _SUBTYPE==NODE_SUBTYPE.NULL:
            raise Exception()
        
        elif _SUBTYPE==NODE_SUBTYPE.DICT:
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

        elif _SUBTYPE==NODE_SUBTYPE.LIST:
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
        Node["_SUBTYPE"] = NODE_SUBTYPE.DICT
        Dict = Node["_DICT"]
        Node["_TYPE"] = NODE_TYPE.SPINE
        for Key, SubNodes in Dict.items():
            for Index, SubNode in enumerate(SubNodes):
                SubNode["_PATH_FROM_ROOT"] = Node["_PATH_FROM_ROOT"] + [Key]
                _AnalyzeTreeRecur(SubNode, Key)
    elif _PARSE_TYPE == NODE_PARSE_TYPE.LIST:
        Node["_SUBTYPE"] = NODE_SUBTYPE.LIST
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
        Node["_SUBTYPE"] = NODE_SUBTYPE.NULL
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

def JsonFile2Param(FilePath, SplitKeyByDot=True):
    Tree = JsonFile2Tree(FilePath)
    assert Tree["_TYPE"] == NODE_TYPE.SPINE
    PathTable = Tree2Paths(Tree, SplitKeyByDot=SplitKeyByDot)

    if Tree["_SUBTYPE"] == NODE_SUBTYPE.DICT:
        RootNode = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.DICT,
            _PATH_FROM_ROOT=["ROOT"]
        )
    elif Tree["_SUBTYPE"] == NODE_SUBTYPE.LIST:
        RootNode = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.LIST,
            _PATH_FROM_ROOT=["ROOT"]
        )   
    else:
        raise Exception()

    Tree = _Paths2Tree(PathTable, RootNode)
    return _Tree2Param(Tree)