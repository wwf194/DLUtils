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
    "ENDL"
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

#要忽略的符号. 忽略空格和制表符\t(就是TAB)
t_ignore = ' \t\r'

# 函数定义的TOKEN匹配规则 > 正则表达式字符串定义的TOKEN匹配规则
def t_FLOAT(t):
    r'[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)'
    return t

def t_INT(t):
    r'(0[xX][0-9a-fA-F]+)|([0-9]+)|([01]+[bB])'
    # 必须返回 t. 可以给t设置属性，方便后续语法分析
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

# 遇到无法匹配的TOKEN时会进入此方法
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

class NODE_TYPE:
    EMPTY = 0
    NODES = 1
    NODE = 2
    DICT = 3
    DICT_NODE = 4
    DICT_NODES = 5
    LIST = 6
    STR = 7
    INT = 8
    FLOAT = 9
    LEAF = 10
    ENDL = 11

class NODE_LEAF_TYPE:
    INT = 0
    FLOAT = 1
    STR = 2
    BOOLEAN = 3
    NULL = 4

node_dict_nodes = {
    "_TYPE": NODE_TYPE.DICT_NODES,
    "_DICT": {}
}
node_dict_node = {
    "_TYPE": NODE_TYPE.DICT_NODE
}
node_empty = {"_TYPE": NODE_TYPE.EMPTY}
node_nodes = {
    "_TYPE": NODE_TYPE.NODES,
    "_LIST": []
}
node_list = {
    "_TYPE": NODE_TYPE.LIST
}
node_dict = {
    "_TYPE": NODE_TYPE.DICT
}
node_leaf = {"_TYPE": NODE_TYPE.LEAF}
node_endl = {"_TYPE": NODE_TYPE.ENDL}

start = "root"

def p_root(p):
    '''
        root : dict
             | list
    '''
    p[0] = p[1]

def p_list_nodes(p):
    '''
        list_nodes : list_nodes_non_empty
              | list_nodes_non_empty comma
    '''  
    p[0] = p[1]

def p_list_nodes_non_empty(p):
    '''
        list_nodes_non_empty : node
                             | list_nodes_non_empty comma node
    '''
    # allows trailing empty dict, empty list, comma.
    if len(p) == 3 and isinstance(p[2], dict):
        p[0] = p[1]
    elif len(p) == 2 or len(p) == 3:
        Node = dict(node_nodes)
        Node["_LIST"] = [p[1]]
        p[0] = Node
    elif len(p) == 4:
        Node = p[1]
        Node["_LIST"].append(p[3])  
        p[0] = Node
    else:
        raise Exception()

def p_node(p):
    '''
        node : leaf
             | dict
             | list
             | node ENDL
    '''
    p[0] = p[1]

# def p_comma_trailing(p):
#     'comma_trailing : COMMA '

def p_dict(p):
    '''
        dict : BRACKET_CURLY_LEFT endl dict_nodes BRACKET_CURLY_RIGHT
             | BRACKET_CURLY_LEFT endl BRACKET_CURLY_RIGHT
    '''

    if len(p) == 4:
        p[0] = dict(node_dict)
    else:
        p[0] = p[3]
        p[0]["_TYPE"] = NODE_TYPE.DICT

def p_dict_node(p):
    '''
        dict_node : str COLON node
    '''
    if len(p) == 3:
        p[0] = p[1]
    else:
        p[0] = dict(node_dict_node)
        p[0]["_KEY"] = p[1]
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
    if len(p) == 2 and p[1]["_TYPE"] == NODE_TYPE.EMPTY:
        p[0] = dict(node_dict_nodes)
    elif len(p) == 3 and isinstance(p[2], dict) and p[2]["_TYPE"] == NODE_TYPE.ENDL:
        p[0] = p[1]
    elif len(p) == 2 or len(p) == 3:
        SubNode = p[1]
        Node = dict(node_dict_nodes)
        Node["_DICT"][SubNode["_KEY"]] = [SubNode["_VALUE"]]
        p[0] = Node
    elif len(p) == 4 and p[3]["_TYPE"] == NODE_TYPE.DICT_NODE:
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
        list : BRACKET_SQUARE_LEFT endl list_nodes BRACKET_SQUARE_RIGHT
             | BRACKET_SQUARE_LEFT endl BRACKET_SQUARE_RIGHT
    '''
    if len(p) == 4:
        p[0] = dict(node_list)
    elif len(p) == 5:  
        p[0] = p[3]
        p[0]["_TYPE"] = NODE_TYPE.LIST
    else:
        raise Exception()

def p_endl(p):
    '''
        endl : ENDL
             | empty
    '''
    p[0] = dict(node_endl)

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
    Node = dict(node_leaf)
    Node["_LEAF"] = int(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.INT
    p[0] = Node

def p_float(p):
    '''
        float : FLOAT
    '''   
    Node = dict(node_leaf)
    Node["_LEAF"] = float(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.FLOAT
    p[0] = Node

def p_boolean(p):
    '''
        null : BOOLEAN
    '''   
    Node = dict(node_leaf)
    Node["_LEAF"] = bool(p[1])
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.BOOLEAN
    p[0] = Node

def p_null(p):
    '''
        null : NULL
    '''   
    Node = dict(node_leaf)
    Node["_LEAF"] = None
    Node["_LEAF_TYPE"] = NODE_LEAF_TYPE.NULL
    p[0] = Node

def p_str(p):
    '''
        str : STR_SINGLE_QUOTATION_MARK 
            | STR_DOUBLE_QUOTATION_MARK
    '''
    p[0] = p[1]

def p_comma(p):
    '''
        comma : COMMA
              | COMMA ENDL
    '''
def p_empty(p):
    '''empty : '''
    p[0] = dict(node_empty)

def p_error(p):
    #print(f"Syntax error at '{p.value}'. Line:{p.lineno} Pos:{p.lexpos}")
    print(f"Syntax error at '{p.value}'. Line:{p.lineno}")

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

def _JsonStr2Tree(FilePath):
    lexer = lex.lex()
    lexer.input(JsonStr)
    parser = yacc.yacc(debug=True)
    Tree = None
    try:
        Tree = parser.parse(JsonStr, lexer=lexer)
    except Exception:
        pass
    return Tree


if __name__ == '__main__':
    YaccTest()
    


