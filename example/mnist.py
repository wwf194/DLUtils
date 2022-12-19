
import DLUtils
def MNIST_MLP():
    BatchSize = 64
    Device = "cuda:0"

    #DLUtils.file.FolderDesciprtion("~/Data/mnist").ToJsonFile("task/image/classification/mnist-folder-config.jsonc")
    Log = DLUtils.log.SeriesLog()
    Model = DLUtils.NN.ModuleSequence(
        [
            DLUtils.transform.Reshape(-1, 28 * 28),
            DLUtils.transform.Norm(0.0, 256.0, 0.0, 1.0),
            DLUtils.NN.NonLinearLayer().SetWeight(
                DLUtils.SampleFromKaimingUniform((28 * 28, 100), "ReLU")
            ).SetMode("f(Wx+b)").SetBias(0.0).SetNonLinear("ReLU"),
            #DLUtils.norm.LayerNorm().SetFeatureNum(100),
            DLUtils.NN.LinearLayer().SetWeight(
                DLUtils.SampleFromXaiverUniform((100, 10))
            ).SetMode("Wx+b").SetBias("zeros"),
        ]
    ).SetLog(Log).Init()
    Log.ToJsonFile("./test/log.json")
    Log.ToFile("./test/log.dat")

    TrainParam = Model.ExtractTrainParam()
    DLUtils.file.JsonDict2File(TrainParam, "./test/trainable_param.json")
    
    SavePath = "./test/mlp - param.dat"
    Model.ToFile(SavePath).ToJsonFile("./test/mlp - param.json")
    Model = DLUtils.NN.ModuleSequence().FromFile("./test/mlp - param.dat")
    Model.ToJsonFile("./test/mlp - param - 2.json")
    Model.PlotWeight("./test/")
    Model.SetLog(Log).FromFile(SavePath).Init()
    Input = DLUtils.SampleFromNormalDistribution((64, 28, 28), 0.0, 1.0)
    Output = Model.Receive(
        DLUtils.ToTorchTensor(Input)
    )
    DLUtils.plot.PlotDataAndDistribution2D(Input[0], SavePath="./test/Input.svg")
    DLUtils.plot.PlotBatchDataAndDistribution1D(Output, SavePath="./test/Output.svg")

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

    Optimizer = DLUtils.Optimizer("GradientDescend") \
        .SetSubType("Adam") \
        .Enable("Momentum") \
        .SetTrainParam(Model.ExtractTrainParam()) \
        .SetParam(LearningRate=0.0001, Alpha=0.1, Beta=0.1) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .Init()

    Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist.zip")


    
    DLUtils.TrainProcess("Epoch-Batch") \
        .SetLog(Log) \
        .SetParam(EpochNum=100) \
        .SetParam(BatchNum=Task.TrainBatchNum(BatchSize)) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .BindTrainData(Task.TrainData(BatchSize=BatchSize)) \
        .BindTestData(Task.TestData()) \
        .BindOptimizer(Optimizer) \
        .SetDevice(Device) \
        .Start()


def MNIST_CONV():
    BatchSize = 64
    Device = "cuda:0"

    #DLUtils.file.FolderDesciprtion("~/Data/mnist").ToJsonFile("task/image/classification/mnist-folder-config.jsonc")
    Log = DLUtils.log.SeriesLog()
    Model = DLUtils.NN.ModuleSequence(
        [
            DLUtils.transform.Reshape(-1, 28 * 28),
            DLUtils.transform.Norm(0.0, 256.0, 0.0, 1.0),
            DLUtils.NN.Conv
        ]
    ).SetLog(Log).Init()
    Log.ToJsonFile("./test/log.json")
    Log.ToFile("./test/log.dat")

    TrainParam = Model.ExtractTrainParam()
    DLUtils.file.JsonDict2File(TrainParam, "./test/trainable_param.json")
    
    SavePath = "./test/mlp - param.dat"
    Model.ToFile(SavePath).ToJsonFile("./test/mlp - param.json")
    Model = DLUtils.NN.ModuleSequence().FromFile("./test/mlp - param.dat")
    Model.ToJsonFile("./test/mlp - param - 2.json")
    Model.PlotWeight("./test/")
    Model.SetLog(Log).FromFile(SavePath).Init()
    Input = DLUtils.SampleFromNormalDistribution((64, 28, 28), 0.0, 1.0)
    Output = Model.Receive(
        DLUtils.ToTorchTensor(Input)
    )
    DLUtils.plot.PlotDataAndDistribution2D(Input[0], SavePath="./test/Input.svg")
    DLUtils.plot.PlotBatchDataAndDistribution1D(Output, SavePath="./test/Output.svg")

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

    Optimizer = DLUtils.Optimizer("GradientDescend") \
        .SetSubType("Adam") \
        .Enable("Momentum") \
        .SetTrainParam(Model.ExtractTrainParam()) \
        .SetParam(LearningRate=0.0001, Alpha=0.1, Beta=0.1) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .Init()

    Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist.zip")


    
    DLUtils.TrainProcess("Epoch-Batch") \
        .SetLog(Log) \
        .SetParam(EpochNum=100) \
        .SetParam(BatchNum=Task.TrainBatchNum(BatchSize)) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .BindTrainData(Task.TrainData(BatchSize=BatchSize)) \
        .BindTestData(Task.TestData()) \
        .BindOptimizer(Optimizer) \
        .SetDevice(Device) \
        .Start()