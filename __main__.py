
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

def TestJsonUtils():
    FileName = "test/JsonTestFile.jsonc"
    FilePath = DLUtils.file.FolderPathOfFile(__file__) + FileName
    # JsonDict = DLUtils.JsonFile2JsonDict(FilePath)
    # Obj = DLUtils.param.JsonStyleObj2Param(JsonDict)
    # Str = DLUtils.param.Param2JsonStr(Obj)
    # DLUtils.Str2File(Str, DLUtils.file.AddSuffixToFileWithFormat(FilePath, " - Reproduce"))
    Param = DLUtils.param.JsonFile2Param(FilePath)
    Param[0].A.B.I.J.K = 5
    print(Param.I.J.K)
    Str = DLUtils.param.Param2JsonStr(Param)
    DLUtils.Str2File(Str, DLUtils.file.AddSuffixToFileWithFormat(FilePath, " - Reproduce"))
    return

if __name__=="__main__":
    if args.task in ["json", "TestJsonUtils"]:
        TestJsonUtils()
    elif args.task in ["BuildMLP", "mlp"]:
        import DLUtils
        MLP = DLUtils.NN.ModuleSequence(
            [
                DLUtils.NN.LinearLayer().SetWeight(
                    DLUtils.SampleFromKaimingUniform((100, 50), "ReLU")
                ).SetMode("Wx").SetBias(0.0)
            ]
        )
        MLP.ToFile("./test/mlp - param.dat")
        MLP.ToJsonFile("./test/mlp - param.json")
        MLP = DLUtils.NN.ModuleSequence().FromFile("./test/mlp - param.dat")
        MLP.ToJsonFile("./test/mlp - param - 2.json")
        MLP.PlotWeight("./test/")
    else:
        raise Exception()
