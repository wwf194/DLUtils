# type: ignore
# makes pylance ignore this file.
# for running scripts inside this package
import sys
import traceback
sys.path.append("../")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "-t", "--task", dest="task", # 这个参数将被存储在args.dest属性中
        nargs="+", # 1 or many
        default=None, # 之后的参数必须是 xxx=xxx 的形式 
        help="task type"
    )
parser.add_argument(
        "-p", "--path", dest="path", default=None,
        nargs=1,
        help="file or directory path required by task"
    )
parser.add_argument(
        "-a", "--analysis", dest="IsAnalysis",
        # nargs=0, # 与action冲突
        action="store_true", default=False,
        help="whre or not this is an analysis task"
    )
parser.add_argument("-nn", "--NodeNum", dest="NodeNum", type=int, default=1)
parser.add_argument("-ni", "--NodeIndex", dest="NodeIndex", type=int, default=0)
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
        Param1 = TaskList[1]
        if Param1 in ["mnist_mlp"]:
            import DLUtils
            #DLUtils.file.FolderConfig("~/Data/mnist").ToJsonFile("task/image/classification/mnist-folder-config.jsonc")
            DLUtils.example.mnist_mlp()
        elif Param1 in ["cifar10"]:
            Param2 = TaskList[2]
            if Param2 in ["conv"]:
                if args.IsAnal:
                    DLUtils.example.cifar10_conv_anal()
                else:
                    DLUtils.example.cifar10_conv()
            else:
                raise Exception()
        elif Param1 in ["vae_mnist", "mnist_vae"]:
            Param2 = TaskList[2]
            if Param2 in ["conv"]:
                DLUtils.example.vae_mnist_conv()
            elif Param2 in ["mlp"]:
                DLUtils.example.vae_mnist_mlp()
            else:
                raise Exception()
        elif Param1 in ["vit"]:
            DLUtils.example.vision_transformer_imagenet_1k()
        elif Param1 in ["parallel", "parallel_test"]:
            DLUtils.example.vision_transformer_imagenet_1k_parallel(
                NodeNum=args.NodeNum, NodeIndex=args.NodeIndex
            )
        else:
            raise Exception(SubTask)
    elif Task in ["cifar10", "cifar"]:
        Param1 = TaskList[1]
        if Param1 in ["generate_config"]:
            Path = args.path
            if Path is None:
                Path = "~/Data/cifar-10-batches-py"
            DLUtils.task.image.classification.cifar10.DataSetConfig(
                args.path,
                "./task/image/classification/cifar10/"
            )
    elif Task in ["ftp"]:
        SubTask = TaskList[1]
        if SubTask in ["test"]:
            DLUtils.utils.network.ftp.Test()
        else:
            raise Exception()
    elif Task in ["image"]:
        SubTask = TaskList[1]
        if SubTask in ["compress"]:
            SubTask2 = TaskList[2]
            if SubTask2 in ["test"]:
                DLUtils.utils.image.compress.Test()
            else:
                raise Exception()
        else:
            raise Exception()
    
    else:
        raise Exception()

def ParseCmdArgs():
    parser = argparse.ArgumentParser()
    # parser.add_argument("task", nargs="?", default="DoTasksFromFile")
    parser.add_argument("-t", "--task", dest="task",
        nargs="?", # 0 or 1
    default="CopyProject2DirAndRun")
    parser.add_argument("-t2", "--task2", dest="task2", default="DoTasksFromFile")
    parser.add_argument("-id", "--IsDebug", dest="IsDebug", default=True)

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default=None)
    # parser.add_argument("-sd", "--SaveDir", dest="SaveDir", default="./log/DoTasksFromFile-2021-10-16-16:04:16/")
    parser.add_argument("-tf", "--TaskFile", dest="TaskFile", default="./task.jsonc")
    parser.add_argument("-tn", "--TaskName", dest="TaskName", default="Main")

    # If Args.task in ['CopyProject2DirAndRun'], this argument will be used to designate file to be run.
    parser.add_argument("-ms", "--MainScript", dest="MainScript", default="main.py")
    


    CmdArgs = parser.parse_args()
    return Namespace2PyObj(CmdArgs) # CmdArgs is of type namespace