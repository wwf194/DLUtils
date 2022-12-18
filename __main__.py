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
        #DLUtils.file.FolderDesciprtion("~/Data/mnist").ToJsonFile("task/image/classification/mnist-folder-config.jsonc")

        Log = DLUtils.log.SeriesLog()
        Model = DLUtils.NN.ModuleSequence(
            [
                DLUtils.transform.Reshape(-1, 28 * 28),
                DLUtils.NN.NonLinearLayer().SetWeight(
                    DLUtils.SampleFromKaimingUniform((28 * 28, 100), "ReLU")
                ).SetMode("f(Wx+b)").SetBias(0.0).SetNonLinear("ReLU"),
                DLUtils.norm.LayerNorm().SetFeatureNum(100),
                DLUtils.NN.LinearLayer().SetWeight(
                    DLUtils.SampleFromXaiverUniform((100, 10))
                ).SetMode("Wx+b").SetBias("zeros"),
            ]
        ).SetLog(Log).Init()
        Log.ToJsonFile("./test/log.json")
        Log.ToFile("./test/log.dat")

        TrainParam = MLP.ExtractTrainParam()
        DLUtils.file.JsonDict2File(TrainParam, "./test/trainable_param.json")
        
        SavePath = "./test/mlp - param.dat"
        Model.ToFile(SavePath).ToJsonFile("./test/mlp - param.json")
        Model = DLUtils.NN.ModuleSequence().FromFile("./test/mlp - param.dat")
        Model.ToJsonFile("./test/mlp - param - 2.json")
        Model.PlotWeight("./test/")
        Model.SetLog(Log).FromFile(SavePath).Init()
        Input = DLUtils.SampleFromNormalDistribution((64, 28, 28), 0.0, 1.0)
        Output = MLP.Receive(
            DLUtils.ToTorchTensor(Input)
        )
        DLUtils.plot.PlotDataAndDistribution2D(Input[0], SavePath="./test/Input.svg")
        DLUtils.plot.PlotBatchDataAndDistribution1D(Output, SavePath="./test/Output.svg")

        Optimizer = DLUtils.Optimizer("GradientDescend") \
            .SetSubType("Adam") \
            .Enable("Momentum") \
            .SetTrainParam(Model.ExtractTrainParam()) \
            .SetParam(LearningRate=0.01, Alpha=0.1, Beta=0.1) \
            .Init()

        Loss = DLUtils.NN.ModuleGraph() \
            .AddSubModule(
                "ClassIndex2OneHot", DLUtils.transform.Index2OneHot(10)
            )\
            .AddSubModule(
                "Loss", DLUtils.Loss("SoftMaxAndCrossEntropy")
            )\
            .AddRoute(
                "ClassIndex2OneHot", ["OutputTarget"], "OutputTargetProb"
            )\
            .AddRoute(
                "Loss", ["Output", "OutputTargetProb"], "Loss"
            )\
            .SetOutput("Loss").Init()

        Evaluator = DLUtils.Evaluator("ImageClassification") \
            .SetLoss(Loss)

        Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist.zip")

        BatchSize = 64
        DLUtils.TrainProcess("Epoch-Batch") \
            .SetLog(Log) \
            .SetParam(EpochNum=100) \
            .SetParam(BatchNum=Task.TrainBatchNum(BatchSize)) \
            .BindEvaluator(Evaluator) \
            .BindModel(MLP) \
            .BindTrainData(Task.TrainData(BatchSize=BatchSize)) \
            .BindTestData(Task.TestData()) \
            .BindOptimizer(Optimizer) \
            .Start()
    else:
        raise Exception()
