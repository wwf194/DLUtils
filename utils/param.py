import DLUtils

import warnings
from enum import Enum

class NODE_TYPE(Enum):
    LEAF = 0
    SPINE = 1
    SPINE_WITH_LEAF = 2
    COMMENT = 3 # Temporal Node

class NODE_SUBTYPE(Enum):
    NULL = 0
    DICT = 1
    LIST = 2

class NODE_CHILD_TYPE(Enum):
    NO_SUBNODE = 0
    NO_SUBNODE_WITH_LEAF = 1
    LEAF = 1
    SINGLE_SUBNODE = 2
    SINGLE_SUBNODE_WITH_LEAF = 3
    MULTI_SUBNODE = 4
    MULTI_SUBNODE_WITH_LEAF = 5


class Param():
    def __init__(self, Dict=None):
        self.Init(Dict)
    def Init(self, Dict=None):
        self.SetAttr("_LIST", [])
        self.SetAttr("_DICT", {})
        if Dict is not None:
            self.SetAttr("_TYPE", Dict["_TYPE"])
            self.SetAttr("_SUBTYPE", Dict["_SUBTYPE"])
            self.SetAttr("_PATH_FROM_ROOT", Dict["_PATH_FROM_ROOT"])
            self.SetAttr("_LEAF", Dict.get("_LEAF"))
    def __getattr__(self, Value):
        return self._DICT[Value]
    def __setattr__(self, Key, Value):
        self._DICT[Key] = Value
    def __delattr__(self, Name):
        self._DICT.pop(Name)
    def __getitem__(self, Index):
        return self._LIST[Index]
    def SetAttr(self, Key, Value):
        self.__dict__[Key] = Value
    def GetAttr(self, Key):
        return self.__dict__[Key]
    def ToJsonFile(self, FilePath):
        Dict = {}
    def ToFile(self, FilePath):
        return
    def FromFile(self, FilePath):
        return

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

def Param2JsonFile(Obj):
    assert isinstance(Obj, Param)
    Tree = Param2Tree(Obj)
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

def _Tree2JsonStr(RootNode):
    _DetectTreeCompress(RootNode)
    _CompressTree(RootNode)
    return _CompressedTree2JsonStr(RootNode)

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
    else:
        Content = "\"_NONE_BASIC_JSON_TYPE\""

    #Content += "\n"
    return Content

def _CompressedTree2JsonStr(RootNode):
    StrList = []
    _Tree2JsonStrRecur(RootNode, StrList, 0, None)
    return "".join(StrList)

def _AppendWithIndent(StrList, IndentNum, Str):
    for _ in range(IndentNum):
        StrList.append("\t")
    StrList.append(Str)

def _JsonDict2StrRecur(Obj, StrList, IndentNum, ParentKey=None):
    if isinstance(Obj, list):
        _AppendWithIndent(StrList, IndentNum, "[\n")
        for index, item in enumerate(Obj):
            _JsonDict2StrRecur(item, StrList, IndentNum + 1)
            StrList.append(",")
        _AppendWithIndent(StrList, IndentNum, "]\n")
    
    elif isinstance(Obj, dict):
        _AppendWithIndent(StrList, IndentNum, "{\n")
        for key, item in Obj.items():
            _AppendWithIndent(StrList, IndentNum + 1, "\"%s\":"%key)
            _JsonDict2StrRecur(item, StrList, IndentNum + 1, key)
            StrList.append("\n")
        _AppendWithIndent(StrList, IndentNum, "}\n")
    
    else:
        Content = ToJsonStr(Obj)
        if isinstance(ParentKey, str):
            StrList.append(Content)
        elif isinstance(ParentKey, int):
            _AppendWithIndent(StrList, IndentNum, Content)   
        else:
            raise Exception()

def _AddComment(Node, StrList):
    # Add Comment
    Comment = Node.get("_COMMENT")
    if Comment is not None:
        # assert isinstance(Comment, list)
        StrList.append("\t")
        StrList += Comment

def _Tree2JsonStrRecur(Node, StrList, IndentNum, Key):
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]
    if _TYPE==NODE_TYPE.LEAF:
        if isinstance(Key, int):
            _AppendWithIndent(StrList, IndentNum, ToJsonStr(Node["_LEAF"]))
        else:
            StrList.append(ToJsonStr(Node["_LEAF"]))
        _AddComment(Node, StrList)
    elif _TYPE==NODE_TYPE.SPINE:
        if _SUBTYPE==NODE_SUBTYPE.DICT:
            IndexMax = len(Node["_DICT"].keys()) - 1
            if IndexMax == -1: # Empty Dict
                if isinstance(Key, int):
                    _AppendWithIndent(StrList, IndentNum, "{}")
                else:
                    _AppendWithIndent(StrList, 0, "{}") # Include root node.
                _AddComment(Node, StrList)
                return
            if isinstance(Key, int):
                _AppendWithIndent(StrList, IndentNum, "{")
            else:
                _AppendWithIndent(StrList, 0, "{") # Include root node.
            _AddComment(Node, StrList)
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
                            StrList.append(",\n")
                        else:
                            StrList.append("\n")
                        
                else:
                    _AppendWithIndent(StrList, IndentNum + 1, f"\"{SubKey}\": ")
                    StrList.append(ToJsonStr(SubNodes))
                    if Index < IndexMax:
                        StrList.append(",\n")
                    else:
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

            _AddComment(Node, StrList)
            StrList.append("\n")

            for SubIndex, SubNode in Node["_LIST"].items():
                _Tree2JsonStrRecur(SubNode, StrList, IndentNum + 1, SubIndex)
                if SubIndex < SubIndexMax:
                    StrList.append(",\n")
                else:
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
        if SupKey=="B":
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
    Str = _Tree2JsonStr(Tree)
    return Str

def JsonDict2Str(Dict):
    StrList = []
    _JsonDict2StrRecur(Dict, [], 0)
    return "".join(StrList)

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
                    SubNode
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
    assert isinstance(Obj, Param)
    Tree = Param2TreeRecur(Obj, None, None)
    return Tree

def Param2TreeRecur(Obj, SupNode, SupNodeKey):
    if isinstance(Obj, Param):
        _TYPE = Obj.GetAttr("_TYPE")
        _SUBTYPE = Obj.GetAttr("_SUBTYPE")
        _PATH_FROM_ROOT = Obj.GetAttr("_PATH_FROM_ROOT")
        Node = _NewNode(
            _TYPE, _SUBTYPE, None, _PATH_FROM_ROOT
        )
        if _TYPE in [NODE_TYPE.LEAF, NODE_TYPE.SPINE_WITH_LEAF]:
            _LEAF = Obj.GetAttr("_LEAF")
            Node["_LEAF"] = _LEAF
        if _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
            if _SUBTYPE==NODE_SUBTYPE.DICT:
                for key, value in Obj.GetAttr("_DICT").items():
                    Node["_DICT"][key] = Param2TreeRecur(value, Node, key)
            elif _SUBTYPE==NODE_SUBTYPE.LIST:
                for index, value in enumerate(Obj.GetAttr("_LIST")):
                    Node["_LIST"][index] = Param2TreeRecur(value, Node, index)
            else:
                raise Exception(_SUBTYPE)
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

def _Tree2Paths(Root):
    PathTable = []
    _Tree2PathsRecur(Root, PathTable, ["ROOT"])
    return PathTable

class PATH_TYPE(Enum):
    LEAF, COMMENT = range(2)

def _Tree2PathsRecur(Node, PathTable, PathCurrent):
    _TYPE = Node["_TYPE"]
    _SUBTYPE = Node["_SUBTYPE"]

    Comment = Node.get("_COMMENT")
    if Comment is not None:
        assert isinstance(Comment, list)
        if len(Comment) > 0:
            PathTable.append([PathCurrent, Comment, PATH_TYPE.COMMENT])
    if _TYPE == NODE_TYPE.SPINE:
        if _SUBTYPE == NODE_SUBTYPE.DICT:
            Dict = Node["_DICT"]
            if len(Dict.keys()) == 0:
                PathTable.append(
                    [PathCurrent, _SymbolEmptyDict(), PATH_TYPE.LEAF]
                )
                return
            for Keys, SubNodes in Dict.items():
                KeyList = Keys.split(".")
                for Index, SubNode in enumerate(SubNodes):
                    # assert isinstance(Key, str)
                    _Tree2PathsRecur(
                        SubNode,
                        PathTable,
                        PathCurrent = PathCurrent + KeyList
                    )
        elif _SUBTYPE == NODE_SUBTYPE.LIST:
            List = Node["_LIST"]
            if len(List) == 0:
                PathTable.append(
                    [PathCurrent, _SymbolEmptyList(), PATH_TYPE.LEAF]
                )
                return
            for Index, SubNode in enumerate(List):
                _Tree2PathsRecur(
                    SubNode, PathTable, PathCurrent + [Index]
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
        assert Node["_SUBTYPE"] == NODE_SUBTYPE.LIST
        Node["_SUBTYPE"] == NODE_SUBTYPE.LIST
        if Key in Node["_LIST"].keys():
            return Node["_LIST"][Key]
    elif isinstance(Key, str):
        assert Node["_SUBTYPE"] == NODE_SUBTYPE.DICT, Node
        Node["_SUBTYPE"] = NODE_SUBTYPE.DICT
        if Key in Node["_DICT"].keys():
            return Node["_DICT"][Key]
    else:
        raise Exception()
    return None

def _Paths2Tree(PathTable, RootNode=None):
    if RootNode is None:
        pass
    for Path, Leaf, _PATH_TYPE in PathTable:
        if Path == ['ROOT', 0, 'A', 'B', 'C']:
            a = 1
        Node = RootNode
        PathLength = len(Path)
        # Dealing with keys that are not last.
        Index = 1
        for Key in Path[1:-1]: #Path with ROOT.
            NextKey = Path[Index + 1]
            SubNode = _GetNodeKey(Node, Key)
            if SubNode is not None:
                if(SubNode["_PATH_FROM_ROOT"]==['ROOT', 0, 'A', 'B', 'C']):
                    a = 1
                _TYPE = SubNode["_TYPE"]
                if _TYPE == NODE_TYPE.LEAF:
                    SubNode["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
                    SubNode["_SUBTYPE"] = NODE_SUBTYPE.LIST if isinstance(NextKey, int) else NODE_SUBTYPE.DICT
                elif _TYPE in [NODE_TYPE.SPINE, NODE_TYPE.SPINE_WITH_LEAF]:
                    pass
                else:
                    raise Exception()
                Node = SubNode
            else:
                SubNodeType = NODE_SUBTYPE.LIST if isinstance(NextKey, int) else NODE_SUBTYPE.DICT
                Node = _SetNodeKey(
                    Node, Key,
                    _NewNode(
                        NODE_TYPE.SPINE,
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
                    SubNode["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
                    SubNode["_LEAF"] = Leaf
                    if _TYPE == NODE_TYPE.SPINE_WITH_LEAF:
                        warnings.warn("Overwriting")
                elif _TYPE == NODE_TYPE.LEAF:
                    SubNode["_LEAF"] = Leaf
                    warnings.warn("Overwriting")
                elif _TYPE == NODE_TYPE.COMMENT:
                    SubNode["_TYPE"] = NODE_TYPE.LEAF
                    SubNode["_LEAF"] = Leaf
                else:
                    raise Exception()
            elif _PATH_TYPE == PATH_TYPE.COMMENT:
                a = 1
                SubNode.setdefault("_COMMENT", []).append(Leaf)
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
                            NODE_SUBTYPE.NULL,
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
        Obj = Param(Node)

        if _TYPE==NODE_TYPE.SPINE_WITH_LEAF:
            Obj.SetAttr("_LEAF", Node["_LEAF"])
        
        if _SUBTYPE==NODE_SUBTYPE.NULL:
            raise Exception()
        elif _SUBTYPE==NODE_SUBTYPE.DICT:
            for Key, SubNode in Node["_DICT"].items():
                setattr(    
                    Obj, Key, _Tree2ParamRecur(SubNode)  
                )
            return Obj

        elif _SUBTYPE==NODE_SUBTYPE.LIST:
            List = Node["_LIST"]
            if isinstance(List, dict):
                IndexList = Node["_LIST"].keys()
                Obj.SetAttr("_LIST", [None for Index in range(len(IndexList))])
                for Index in IndexList:
                    Obj.GetAttr("_LIST")[Index] = _Tree2ParamRecur(Node["_LIST"][Index])
            elif isinstance(List, list):
                ObjList = []
                Obj.SetAttr("_LIST", ObjList)
                for Index, Item in enumerate(List):
                    ObjList.append(Item)
            else:
                raise Exception()
        else:
            raise Exception()
        return Obj

from .json.parse import _JsonStr2Tree
from .json.parse import NODE_TYPE as NODE_PARSE_TYPE

def _AnalyzeTree(Root):
    Root["_PATH_FROM_ROOT"] = ["ROOT"]
    _AnalyzeTreeRecur(Root)

def _AnalyzeTreeRecur(Node, Key=None):
    _PARSE_TYPE = Node["_TYPE"]
    if Key == "_DICT":
        a = 1
    if _PARSE_TYPE == NODE_PARSE_TYPE.DICT:
        Node["_SUBTYPE"] = NODE_SUBTYPE.DICT
        Dict = Node["_DICT"]
        Node["_TYPE"] = NODE_TYPE.SPINE
        # if Node.get("_LEAF") is not None:
        #     Node["_LEAF"] = Dict.pop("_LEAF")
        #     Node["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
        for Key, SubNodes in Dict.items():
            for Index, SubNode in enumerate(SubNodes):
                SubNode["_PATH_FROM_ROOT"] = Node["_PATH_FROM_ROOT"] + [Key]
                _AnalyzeTreeRecur(SubNode, Key)
    elif _PARSE_TYPE == NODE_PARSE_TYPE.LIST:
        Node["_SUBTYPE"] = NODE_SUBTYPE.LIST
        List = Node["_LIST"]
        Node["_TYPE"] = NODE_TYPE.SPINE
        # if Node.get("_LEAF") is not None:
        #     Node["_TYPE"] = NODE_TYPE.SPINE_WITH_LEAF
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
    Tree = _JsonStr2Tree(JsonStr)
    if Tree is None:
        raise Exception("Failed to parse json file.")
    _AnalyzeTree(Tree)    
    return Tree

def JsonFile2Param(FilePath):
    Tree = JsonFile2Tree(FilePath)
    assert Tree["_TYPE"] == NODE_TYPE.SPINE
    if Tree["_SUBTYPE"] == NODE_SUBTYPE.DICT:
        Root = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.DICT,
            _PATH_FROM_ROOT=["ROOT"]
        )
    elif Tree["_SUBTYPE"] == NODE_SUBTYPE.LIST:
        Root = _NewNode(
            _TYPE=NODE_TYPE.SPINE, # Root node must be list / dict.
            _SUBTYPE=NODE_SUBTYPE.LIST,
            _PATH_FROM_ROOT=["ROOT"]
        )   
    else:
        raise Exception()

    PathTable = _Tree2Paths(Tree)
    Tree = _Paths2Tree(PathTable, Root)
    return _Tree2Param(Tree)