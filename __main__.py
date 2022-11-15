
# for running scripts inside this package
import sys
sys.path.append("../")

import argparse
parser = argparse.ArgumentParser(description="命令行参数解析") # description可选
parser.add_argument(
        "-t", "--task", # 数量>=1即可                
        dest="task", # 这个参数将被存储在args.dest属性中
        default="NULL", # 之后的参数必须是 xxx=xxx 的形式 
        help="directory to save trained models"
    )
args = parser.parse_args() # parser输入的命令行参数

from DLUtils import *
if __name__=="__main__":
    obj = utils.EmptyObj()
    print(obj.__dict__)


import os
def TestJsonUtils():
    FolderPath = DLUtils.file.FolderPathOfFile(__file__)
    print(FolderPath)
    JsonDict = DLUtils.JsonFile2JsonDict(FolderPath + "test/JsonTestFile.jsonc")
    Obj = DLUtils.param.ToParamObj(JsonDict)
    print(Obj[0].A.B.C.D.E)
    Str = DLUtils.param.Param2JsonStr(Obj)
    DLUtils.Str2File(Str, "test/JsonTestFile - Reproduce.jsonc")
    return
#net = NewNetwork("Transformer")

if __name__=="__main__":
    if args.task in ["json", "TestJsonUtils"]:
        TestJsonUtils()
