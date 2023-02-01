
import DLUtils
def mnist_mlp(SaveDir="test/mnist/mlp"):
    DLUtils.file.EnsureDir(SaveDir)
    BatchSize = 64
    EpochNum  = 100
    Device = "cuda:0"
    
    Log = DLUtils.log.SeriesLog()
    Model = DLUtils.network.ModuleSeries(
        [
            DLUtils.transform.Reshape(-1, 28 * 28),
            DLUtils.transform.Norm(0.0, 256.0, 0.0, 1.0),
            DLUtils.network.NonLinearLayer().SetWeight(
                DLUtils.SampleFromKaimingUniform((28 * 28, 100), "ReLU")
            ).SetMode("f(Wx+b)").SetBias(0.0).SetNonLinear("ReLU"),
            DLUtils.norm.LayerNorm().SetFeatureNum(100),
            DLUtils.network.LinearLayer().SetWeight(
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
    Model = DLUtils.network.ModuleSeries().FromFile("./test/mlp - param.dat")
    Model.ToJsonFile("./test/mlp - param - 2.json")
    Model.PlotWeight("./test/")
    Model.SetLog(Log).FromFile(SavePath).Init()
    Input = DLUtils.SampleFromNormalDistribution((64, 28, 28), 0.0, 1.0)
    
    Output = Model.Receive(
        DLUtils.ToTorchTensor(Input)
    )
    DLUtils.plot.PlotDataAndDistribution2D(Input[0], SavePath="./test/In.svg")
    DLUtils.plot.PlotBatchDataAndDistribution1D(Output, SavePath="./test/Out.svg")

    Loss = DLUtils.network.ModuleGraph() \
        .AddSubModule("ClassIndex2OneHot", DLUtils.transform.Index2OneHot(10)) \
        .AddSubModule("Loss", DLUtils.Loss("SoftMaxAndCrossEntropy"))\
        .AddRoute("ClassIndex2OneHot", ["OutputTarget"], "OutputTargetProb")\
        .AddRoute("Loss", ["Output", "OutputTargetProb"], "Loss")\
        .SetOutput("Loss").Init()

    Evaluator = DLUtils.Evaluator("ImageClassification") \
        .SetLoss(Loss)
    EvaluationLog = DLUtils.EvaluationLog("ImageClassification")

    Optimizer = DLUtils.Optimizer("GradientDescend") \
        .SetSubType("Adam") \
        .Enable("Momentum") \
        .SetTrainParam(Model.ExtractTrainParam()) \
        .SetParam(LearningRate=0.0001, Alpha=0.1, Beta=0.1) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .Init()

    Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist.zip")
    Save = DLUtils.train.EpochBatchTrain.Save().SetParam(TestNum=10, SaveDir=SaveDir)
    Test = DLUtils.train.EpochBatchTrain.Test().SetParam(TestNum="All")

    DLUtils.TrainSession("Epoch-Batch").SetLog(Log) \
        .SetParam(EpochNum=EpochNum, BatchSize=BatchSize) \
        .SetParam(BatchNum=Task.TrainBatchNum(BatchSize)) \
        .Bind(
            Evaluator=Evaluator, EvaluationLog=EvaluationLog,
            Model=Model, Task=Task, Optimizer=Optimizer, 
            Test=Test, Save=Save
        ) \
        .SetDevice(Device).Init().Start()

def mnist_conv():
    return