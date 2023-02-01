import DLUtils

def cifar10_conv_anal(SaveDir = None, IsTrain=True):
    # cifar10_conv()
    if SaveDir is None:
        SaveDir = DLUtils.PackageFolderPath + "example/cifar10/conv/"
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    Device = "cuda:0"  
    SaveFilePath = DLUtils.file.AfterTrainModelFile(SaveDir + "model/")
    Model = DLUtils.network.ModuleSeries().FromFile(SaveFilePath)
    Model.ToJsonFile(SaveDir + "model/config.jsonc")
    TrainSession = DLUtils.TrainSession().FromFile(SaveDir + "TrainSession.dat").Init()

    TrainSession.Anal.AfterTrain(DLUtils.param({
        "TrainSession": TrainSession
    }))

def cifar10_conv(SaveDir=None, IsAnal=False):
    if SaveDir is None:
        SaveDir = DLUtils.PackageFolderPath + "example/cifar10/conv/"
    else:
        SaveDir = DLUtils.file.StandardizePath(SaveDir)
    if IsAnal:
        cifar10_conv_anal(SaveDir)
    DLUtils.file.EnsureDir(SaveDir)
    BatchSize = 64
    EpochNum  = 5
    Device = "cuda:0"

    Task = DLUtils.Task("ImageClassification").SetType("cifar10").SetDataPath("~/Data/cifar-10-python.tar.gz")
    Log = DLUtils.log.SeriesLog()
    
    Model = DLUtils.network.ModuleSeries(
        [
            DLUtils.transform.Norm(0.0, 256.0, -1.0, 1.0),
            DLUtils.network.Conv2D(Padding=1).SetWeight(
                DLUtils.Conv2DKernel((3, 100, 3, 3), NonLinear="ReLU")
            ).SetBias("zeros"),
            DLUtils.network.AvgPool2D(2),
            DLUtils.network.Conv2D(Padding=1).SetWeight(
                DLUtils.Conv2DKernel((100, 5, 3, 3), NonLinear="ReLU")
            ).SetBias("zeros"),
            DLUtils.transform.Reshape(-1, 5 * 16 * 16),
            DLUtils.network.NonLinearLayer().SetWeight(
                DLUtils.SampleFromKaimingUniform((5 * 16 * 16, 100), "ReLU")
            ).SetMode("f(Wx+b)").SetBias(0.0).SetNonLinear("ReLU"),
            DLUtils.network.LinearLayer().SetWeight(
                DLUtils.SampleFromXaiverUniform((100, 10))
            ).SetMode("Wx+b").SetBias("zeros"),
        ]
    ).SetLog(Log).Init()
    Log.ToJsonFile(SaveDir + "log.json")
    Log.ToFile("./test/log.dat")
    TrainParam = Model.ExtractTrainParam()
    DLUtils.file.JsonDict2File(TrainParam, SaveDir + "trainable_param.json")
    SaveFilePath = SaveDir + "model/config(Init).dat"
    Model.ToFile(SaveFilePath).ToJsonFile(SaveDir + "model/config.json")
    Model = DLUtils.network.ModuleSeries().FromFile(SaveFilePath)
    Model.ToJsonFile(SaveDir + "model/config.jsonc")
    Model.PlotWeight(SaveDir + "model/weight/")
    Model.SetLog(Log).FromFile(SaveFilePath).Init()
    Input = DLUtils.SampleFromNormalDistribution((64, 3, 32, 32), -1.0, 1.0)
    Output = Model.Receive(DLUtils.ToTorchTensor(Input))
    
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
        .SetParam(LearningRate=0.0001, Alpha=0.9, Beta=0.9) \
        .BindEvaluator(Evaluator) \
        .BindModel(Model) \
        .Init()

    Save = DLUtils.train.EpochBatchTrain.Save().SetParam(EventNum=10, SaveDir=SaveDir)
    Test = DLUtils.train.EpochBatchTrain.Test().SetParam(EventNum="All")
    Anal = DLUtils.train.EpochBatchTrain.AnalysisAfterTrain().SetParam(SaveDir=SaveDir)

    DLUtils.TrainSession("Epoch-Batch").SetLog(Log) \
        .SetParam(EpochNum=EpochNum, BatchSize=BatchSize) \
        .Bind(
            Evaluator=Evaluator, EvaluationLog=EvaluationLog,
            Model=Model, Task=Task, Optimizer=Optimizer, 
            Test=Test, Save=Save, Anal=Anal
        ) \
        .AddConnectEvent("Model", "TensorMovement", "Optimizer", "ResetOptimizer") \
        .Init().SetDevice(Device).Start().ToFile(SaveDir + "TrainSession.dat")