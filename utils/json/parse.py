# Ref: https://github.com/akashb95/pyjsonparse

import warnings
import DLUtils
from DLUtils.utils.json.ply import lex

tokens = (
    "BRACKET_CURLY_LEFT",
    "BRACKET_CURLY_RIGHT",
    "BRACKET_SQUARE_LEFT",
    "BRACKET_SQUARE_RIGHT",
    #"QUOTATION_MARK_SINGLE",
    #"QUOTATION_MARK_DOUBLE",
    "COMMA",
    "COLON",
    "FLOAT", "INT",
    "BOOLEAN", "NULL",
    "STR_SINGLE_QUOTATION_MARK",
    "STR_DOUBLE_QUOTATION_MARK",
    "ENDL",
    "LINE_COMMENT"
)

t_BRACKET_CURLY_LEFT = r'\{'
t_BRACKET_CURLY_RIGHT = r'\}'
t_BRACKET_SQUARE_LEFT = r'\['
t_BRACKET_SQUARE_RIGHT = r'\]'
# t_QUOTATION_MARK_SINGLE = r'\''
# t_QUOTATION_MARK_DOUBLE = r'\"'
t_COMMA = r'\,'
t_COLON = r'\:'
t_BOOLEAN = r'(true|false)'
t_NULL = "null"

# single characters to be ignored.
# 忽略空格和制表符\t(就是TAB)
t_ignore = ' \t\r'

def t_LINE_COMMENT(t):
    r'\/\/.*\n'
    return t

def t_INT(t):
    r'(0[xX][0-9a-fA-F]+)|([0-9]+)|([01]+[bB])'
    # 必须返回 t. 可以给t设置属性，方便后续语法分析
    return t

# 函数定义的TOKEN匹配规则 > 正则表达式字符串定义的TOKEN匹配规则
def t_FLOAT(t):
    r'[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)'
    return t

def t_STR_DOUBLE_QUOTATION_MARK(t):
    # string - escaped chars and all but unicode control characters
    r'"(\\[bfrnt"/\\]|[^\u0022\u005C\u0000-\u001F\u007F-\u009F]|\\u[0-9a-fA-F]{4})*"'
    #r'\"(.*)\"'
    return t

def t_STR_SINGLE_QUOTATION_MARK(t):
    r'\'(\\[bfrnt"/\\]|[^\u0022\u005C\u0000-\u001F\u007F-\u009F]|\\u[0-9a-fA-F]{4})*\''
    return t

def t_ENDL(t):
    r"""\n+"""
    t.lexer.lineno += len(t.value)
    return t

# Compute column.
#     input is the input text string
#     token is a token instance
def find_column(input, token):
    line_start = input.rfind('\n', 0, token.lexpos) + 1
    return (token.lexpos - line_start) + 1

# This method will be called when faced with unmatchable TOKEN
def t_error(t):
    # t是以下list
    # [type, value, lineno, lexpos]
    # [符号名称(如t_FLOAT), 符号内容(如1.234), 行号, 列号]
    print(f'Lex error {t.value[0]} at line {t.lineno}, illegal character {t.value[0]}')

def LexerTest(Input=None):
    if Input is None:
        Input = '{,}[]\t 1.234"ABC{,}[]()\n\t"'
    lexer = lex.lex()
    lexer.input(Input)
    while True:
        token = lexer.token()
        print(token)
        if not token:
            # 返回的token为None，表示到达末尾
            break

from DLUtils.utils.json.ply import yacc
from enum import Enum

class _NODE_TYPE(Enum):
    EMPTY = 0
    NODES = 1
    NODE = 2
    DICT = 3
    DICT_NODE = 4
    DICT_NODES = 5
    LIST = 6
    LEAF = 10
    ENDL = 11
    LINE_COMMENT = 12

class NODE_TYPE(Enum):
    EMPTY, NODES, NODE, \
    DICT, DICT_NODE, DICT_NODES, \
    LIST, LEAF, ENDL, COMMA,\
    LINE_COMMENT \
    = range(11)

class NODE_LEAF_TYPE:
    INT = 0
    FLOAT = 1
    STR = 2
    BOOLEAN = 3
    NULL = 4

def dict_nodes_template():
    return {
        "_TYPE": NODE_TYPE.DICT_NODES,
        "_DICT": {}
    }

def dict_node_template():
    return {
        "_TYPE": NODE_TYPE.DICT_NODE
    }
def empty_template():
    return  {"_TYPE": NODE_TYPE.EMPTY}

def list_node_template():
    return {
        "_TYPE": NODE_TYPE.NODES,
        "_LIST": []
    }
def list_template():
    return {
        "_TYPE": NODE_TYPE.LIST,
        "_LIST": []
    }

def dict_template():
    return {
        "_TYPE": NODE_TYPE.DICT,
        "_DICT": {}
    }

def endl_template():
    return {"_TYPE": NODE_TYPE.ENDL}

def leaf_template():
    return {"_TYPE": NODE_TYPE.LEAF}

def line_comment_template():
    return {"_TYPE": NODE_TYPE.LINE_COMMENT}

def comma_template():
    return {"_TYPE": NODE_TYPE.COMMA}

def add_comment(node, comment_node):
    if isinstance(comment_node, dict) and comment_node.get("_COMMENT") is not None:
        node["_COMMENT"] = comment_node["_COMMENT"]

start = "root"

def p_root(p):
    '''
        root : dict
             | list
    '''
    p[0] = p[1]
    return

def p_list_nodes(p):
    '''
        list_nodes : list_nodes_non_empty
                   | list_nodes_non_empty comma
    '''
    p[0] = p[1]
    if len(p) == 3:
        add_comment(p[0], p[2])

def p_list_nodes_non_empty(p):
    '''
        list_nodes_non_empty : node
                             | list_nodes_non_empty comma node
    '''
    # allows trailing empty dict, empty list, comma.
    # if len(p) == 3 and isinstance(p[2], dict):
    #     p[0] = p[1]
    if len(p) == 2:
        Node = list_node_template()
        Node["_LIST"] = [p[1]]
        p[0] = Node
    elif len(p) == 4:
        Node = p[1]
        add_comment(Node["_LIST"][-1], p[2])
        Node["_LIST"].append(p[3])
        p[0] = Node
    else:
        raise Exception()

def p_node(p):
    '''
        node : leaf
             | dict
             | list
             | node comment
    '''
    Node = p[1]
    if len(p) == 3:
        add_comment(Node, p[2])
    p[0] = Node

def p_dict(p):
    '''
        dict : BRACKET_CURLY_LEFT endl_optional dict_nodes BRACKET_CURLY_RIGHT
             | BRACKET_CURLY_LEFT endl_optional BRACKET_CURLY_RIGHT
    '''
    if len(p) == 4:
        p[0] = dict_template()
    else:
        p[0] = p[3]
        p[0]["_TYPE"] = NODE_TYPE.DICT

def p_dict_node(p):
    '''
        dict_node : str COLON node
    '''

    p[0] = dict_node_template()
    if p[1] == "C":
        a = 1
    if p[1] == "A.B":
        a = 1
    p[0]["_KEY"] = p[1]["_LEAF"]
    p[0]["_VALUE"] = p[3]

def p_dict_nodes(p):
    '''
        dict_nodes : dict_nodes_non_empty
                   | dict_nodes_non_empty comma
    '''
    p[0] = p[1]

def p_dict_nodes_non_empty(p):
    '''
        dict_nodes_non_empty : dict_node
                             | dict_nodes_non_empty comma dict_node
    '''
    # allows trailing empty dict, empty list, comma.
    # if len(p) == 2 and p[1]["_TYPE"] == NODE_TYPE.EMPTY:
    #     p[0] = dict(node_dict_nodes)
    # elif len(p) == 3 and isinstance(p[2], dict) and p[2]["_TYPE"] == NODE_TYPE.ENDL:
    #     p[0] = p[1]
    if len(p) == 2:
        SubNode = p[1]
        Node = dict_nodes_template()
        Node["_DICT"][SubNode["_KEY"]] = [SubNode["_VALUE"]]
        p[0] = Node
    elif len(p) == 4:
        Node = p[1]
        SubNode = p[3]
        SubNodeKey = SubNode["_KEY"]
        if Node["_DICT"].get(SubNodeKey) is not None:
            Node["_DICT"][SubNodeKey].append(SubNode["_VALUE"])
        else:
            Node["_DICT"][SubNodeKey] = [SubNode["_VALUE"]]
        p[0] = Node
    else:
        raise Exception()

def p_list(p):
    '''
        list : BRACKET_SQUARE_LEFT endl_optional list_nodes BRACKET_SQUARE_RIGHT
             | BRACKET_SQUARE_LEFT endl_optional BRACKET_SQUARE_RIGHT
    '''
    if len(p) == 4:
        p[0] = list_template()
    elif len(p) == 5:  
        p[0] = p[3]
        p[0]["_TYPE"] = NODE_TYPE.LIST
    else:
        raise Exception()

def p_endl_optional(p):
    '''
        endl_optional : endl
                      | empty
    '''
    p[0] = p[1]

def p_endl(p):
    '''
        endl : ENDL
    '''
    p[0] = endl_template()

def p_leaf(p):
    '''
        leaf : str
             | int
             | float
             | null
             | boolean
    '''
    p[0] = p[1]

def p_int(p):
    '''
        int : INT
    '''   
    Node = leaf_template()
    Node["_LEAF"] = int(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.INT
    p[0] = Node

def p_float(p):
    '''
        float : FLOAT
    '''   
    Node = leaf_template()
    Node["_LEAF"] = float(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.FLOAT
    p[0] = Node

def p_boolean(p):
    '''
        boolean : BOOLEAN
    '''   
    Node = leaf_template()
    Node["_LEAF"] = bool(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.BOOLEAN
    p[0] = Node

def p_null(p):
    '''
        null : NULL
    '''   
    Node = leaf_template()
    Node["_LEAF"] = None
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.NULL
    p[0] = Node

def p_str(p):
    '''
        str : STR_SINGLE_QUOTATION_MARK 
            | STR_DOUBLE_QUOTATION_MARK
    '''

    Node = leaf_template()
    Node["_LEAF"] = p[1][1:-1] # remove quotation mark
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.STR
    p[0] = Node

def p_comment(p):
    '''
        comment : endl
                | line_comment
    '''
    p[0] = p[1]

def p_line_comment(p):
    '''
        line_comment : LINE_COMMENT
    '''
    p[0] = line_comment_template()
    p[0]["_COMMENT"] = p[1]

def p_comma(p):
    '''
        comma : COMMA
              | COMMA comment
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        return comma_template()

def p_empty(p):
    '''empty : '''
    p[0] = empty_template()

def p_error(p):
    #print(f"Syntax error at '{p.value}'. Line:{p.lineno} Pos:{p.lexpos}")
    raise Exception(f"Syntax error at '{p.value}'. Line:{p.lineno}")

def YaccTest():
    from pathlib import Path
    import os
    JsonStr = Path(os.path.dirname(os.path.realpath(__file__)) + "/" + "1 - JsonTest.jsonc").read_text()
    lexer = lex.lex()
    lexer.input(JsonStr)
    while True:
        token = lexer.token()
        print(token)
        if token is None:
            break
    lexer = lex.lex()
    parser = yacc.yacc(debug=True)
    
    # returns p[0] in method p_root(p)
    Tree = parser.parse(JsonStr, lexer=lexer)

def _JsonStr2Tree(JsonStr, Verbose=True, Debug=True):
    lexer = lex.lex()
    lexer.input(JsonStr)
    if Verbose:
        while True:
            token = lexer.token()
            print(token)
            if token is None:
                break

    parser = yacc.yacc(debug=True)
    Tree = None
    if Debug:
        Tree = parser.parse(JsonStr, lexer=lexer)
    else:
        try:
            Tree = parser.parse(JsonStr, lexer=lexer)
        except Exception:
            raise Exception()
    return Tree


if __name__ == '__main__':
    YaccTest()
    


