# type: ignore
# makes pylance ignore this file.

# for running scripts inside this package
import sys
import traceback
sys.path.append("../")

import argparse
parser = argparse.ArgumentParser(description="命令行参数解析") # description可选
parser.add_argument(
        "-t", "--task", # 数量>=1即可                
        dest="task", # 这个参数将被存储在args.dest属性中
        nargs="+",
        default=None, # 之后的参数必须是 xxx=xxx 的形式 
        help="task type"
    )
parser.add_argument(
        "-p", "--path",               
        dest="path",
        default=None,
        help="file or directory path required by task"
    )
args = parser.parse_args() # parser输入的命令行参数

import DLUtils

def TestJsonUtils():
    FileName = "test/JsonTestFile.jsonc"
    FilePath = DLUtils.file.FolderPathOfFile(__file__) + FileName
    # JsonDict = DLUtils.JsonFile2JsonDict(FilePath)
    # Obj = DLUtils.param.JsonStyleObj2Param(JsonDict)
    # Str = DLUtils.param.Param2JsonStr(Obj)
    # DLUtils.Str2File(Str, DLUtils.file.AddSuffixToFileWithFormat(FilePath, " - Reproduce"))
    Param = DLUtils.utils.JsonFile2Param(FilePath)
    Param[0].A.B.I.J.K = 5
    print(Param.I.J.K)
    Str = DLUtils.utils.Param2JsonStr(Param)
    DLUtils.Str2File(Str, DLUtils.file.AddSuffixToFileWithFormat(FilePath, " - Reproduce"))
    return

def TestPlotData():
    import numpy as np
    data = np.ones((10, 10))
    data[0, 5] = float('NaN')
    data[1, 1] = float('Inf')
    dataMasked, data = DLUtils.plot.MaskOutInfOrNaN(data)
    print(dataMasked.shape) # [98]
    data = np.zeros((50, 100))
    for i in range(100):
        data[:, i] = 1.0 * i - 25.0
    DLUtils.plot.PlotData2D("data2D:50x100", data, "./test/data2D50x100.svg", XLabel="100", YLabel="50")
    data = np.zeros((98))
    for i in range(98):
        data[i] = 1.0 * i
    DLUtils.plot.PlotData1D("Data1D:98", data, "./test/data1D98.svg", XLabel="X Label", YLabel="Y Label")

if __name__=="__main__":
    TaskList = args.task
    Task = TaskList[0]
    if Task in ["json", "TestJsonUtils"]:
        TestJsonUtils()
    elif Task in ["example"]:
        SubTask = TaskList[1]
        if SubTask in ["mnist_mlp"]:
            import DLUtils
            #DLUtils.file.FolderConfig("~/Data/mnist").ToJsonFile("task/image/classification/mnist-folder-config.jsonc")
            DLUtils.example.mnist_mlp(
                SaveDir="./test/mnist/MLP/"
            )
        elif SubTask in ["cifar10"]:
            if TaskList[2] in ["conv"]:
                Task = DLUtils.Task("ImageClassification").SetType("cifar10").SetDataPath("~/Data/cifar-10-python.tar.gz")
            else:
                raise Exception()
        else:
            raise Exception(SubTask)
    elif Task in ["cifar10"]:
        SubTask = TaskList[1]
        if SubTask in ["generate_config"]:
            Path = args.path
            if Path is None:
                Path = "~/Data/cifar-10-batches-py"
            DLUtils.task.image.classification.cifar10.DataSetConfig(
                args.path,
                "./task/image/classification/cifar10/"
            )
    else:
        raise Exception()
