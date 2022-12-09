# type: ignore
# makes pylance ignore this file.

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
    if args.task in ["json", "TestJsonUtils"]:
        TestJsonUtils()
    elif args.task in ["BuildMLP", "mlp"]:
        import DLUtils
        DLUtils.file.FolderDesciprtion("~/Data/mnist").ToJsonFile("test/mnist-folder-description.jsonc")

        Log = DLUtils.log.SeriesLog()
        MLP = DLUtils.NN.ModuleSequence(
            [
                DLUtils.NN.NonLinearLayer().SetWeight(
                    DLUtils.SampleFromKaimingUniform((10, 100), "ReLU")
                ).SetMode("f(Wx+b)").SetBias(0.0).SetNonLinear("ReLU"),
                DLUtils.norm.LayerNorm().SetFeatureNum(100),
                DLUtils.NN.LinearLayer().SetWeight(
                    DLUtils.SampleFromXaiverUniform((100, 50))
                ).SetMode("Wx+b").SetBias("zeros"),
            ]
        ).SetLog(Log).Init()
        Log.ToJsonFile("./test/log.json")
        Log.ToFile("./test/log.dat")

        TrainableParam = MLP.ExtractTrainableParam()
        DLUtils.file.JsonDict2File(TrainableParam, "./test/trainable_param.json")
        
        SavePath = "./test/mlp - param.dat"
        MLP.ToFile(SavePath)
        MLP.ToJsonFile("./test/mlp - param.json")
        MLP = DLUtils.NN.ModuleSequence().FromFile("./test/mlp - param.dat")
        MLP.ToJsonFile("./test/mlp - param - 2.json")
        MLP.PlotWeight("./test/")
        MLP.SetLog(Log).FromFile(SavePath).Init()
        Input = DLUtils.SampleFromNormalDistribution((100, 10), 0.0, 1.0)
        Output = MLP.Receive(
            DLUtils.ToTorchTensor(Input)
        )
        DLUtils.plot.PlotDataAndDistribution2D(Input, SavePath="./test/Input.svg")
        DLUtils.plot.PlotBatchDataAndDistribution1D(Output, SavePath="./test/Output.svg")

        Optimizer = DLUtils.Optimizer("GradientDescend") \
            .SetSubType("Adam") \
            .Enable("Momentum") \
            .SetParam(Alpha=0.1, Beta=0.1)

        Loss = DLUtils.Loss("CrossEntropy")

        Evaluator = DLUtils.Evaluator("ImageClassification") \
            .SetLoss(Loss)

        Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist")
        TrainData = Task.TrainData()
        TestData = Task.TestData()

        DLUtils.TrainProcess("Epoch-Batch") \
            .SetLog(Log) \
            .SetParam(EpochNum=100, BatchNum=100) \
            .BindEvaluator(Evaluator) \
            .BindModel(MLP) \
            .BindTrainData(MNISTTrain) \
            .BindTestData(MNISTTest) \
            .BindOptimizer(Optimizer) \
            .Start()
        
    else:
        raise Exception()
