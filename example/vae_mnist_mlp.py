import DLUtils
import DLUtils.network as network
import DLUtils.loss as loss
import DLUtils.train as train
import functools

from .vae_mnist import SampleImage
def vae_mnist_plot(SaveDir):
    TrainSession = DLUtils.FromFile(SaveDir + "TrainSession.dat").Init()
    Model=DLUtils.LoadMostRecentlySavedModelInDir(SaveDir + "model-saved").Init()
    TrainSession.Bind(
        Model=Model
    ).SetDevice("cuda:0")
    Anal = train.EpochBatchTrain.EventAfterTrain().SetEvent(
        functools.partial(SampleImage, LatentUnitNum=Model.Encoder.Mu.Param.Out.Num)
    )
    TrainSession.SimulateAfterTrain(Anal)

def vae_mnist_mlp(SaveDir="./example/vae_mnist/"): 
    Task = "Plot"
    if Task in ["Plot"]:
        Str = DLUtils.NpArray2Str(DLUtils.SampleFromUniformDistribution((10, 10)), Name="TestNpArray")
        DLUtils.Str2File(Str, "./example/np2str.txt")
        vae_mnist_plot(SaveDir)
        return
    SaveDir = DLUtils.file.ToStandardPathStr(SaveDir)
    ModelType = "Conv"
    DLUtils.file.ClearDir(SaveDir)
    XNum = 28 * 28
    LatentUnitNum = 20
    EpochNum = 10
    BatchSize = 64
    Device = "cuda:0"
    PreProcess = network.ModuleSeries().AddSubModule(Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0))

    Encoder = network.ModuleGraph().AddSubModule(
            Reshape=network.Reshape(-1, 28 * 28),
            MLP=network.MLP(
                (XNum, 100, 20),
                NonLinear="ReLU", NonLinearOnLastLayer=True
            ), # Output: [BatchSize, Mu:LogOfVariance]
            Mu=network.Linear(20, LatentUnitNum),
            LogOfVar=network.Linear(20, LatentUnitNum),
            Sampler=network.SampleFromNormalDistribution(InNum=LatentUnitNum, InType="LogOfVar")
        ).AddRoute(
            Reshape=("X", "Y"),
            MLP=("Y", "Y"),
            Mu=("Y", "Mu"),
            LogOfVar=("Y", "LogOfVar"),
            Sampler=(["Mu", "LogOfVar"], "Z")
        ).SetIn("X").SetOut("Mu", "LogOfVar", "Z").PlotWeight(SaveDir=SaveDir + "weight/")

    Decoder = network.ModuleList(
        network.MLP(
            (LatentUnitNum, 100, XNum), NonLinear="ReLU", NonLinearOnLastLayer=False
        ),
        network.Sigmoid(),
        network.Reshape(-1, 28, 28)
    )

    Model = network.ModuleGraph().AddSubModule(
            PreProcess=PreProcess,
            Encoder=Encoder,
            Decoder=Decoder
        ).AddRoute(
            PreProcess=["X", "XIn"],
            Encoder=["XIn", {"Mu", "LogOfVar", "Z"}],
            Decoder=["Z", "XPred"]
        # ).SetParam(OutType="AllInDict", OutName=["Z", "XPred"]) \
        ).SetDictOut("Z", "XPred", "Mu", "LogOfVar", "XIn").SetIn("X") \
        .Init().SetDevice(Device).SetRoot()
    
    # test input
    TestIn = DLUtils.ToTorchTensor(
        DLUtils.SampleFromUniformDistribution((10, 28, 28), -1.0, 1.0)
    ).to(Device)
    TestOut = Model(TestIn)

    Out = DLUtils.param(Out)
    Z, XPred = Out.Z, Out.XPred
    XPred = DLUtils.ToNpArray(XPred)
    DLUtils.plot.PlotGreyImage(XPred[0], SaveDir + "image-decode/" + "test.png")
    Loss = network.ModuleGraph() \
        .AddSubModule(
            #LossReconstruct=loss.CrossEntropy2Class(AfterOperation="Mean"),
            LossReconstruct=loss.MSELoss(AfterOperation="Mean"),
            LossKL=loss.KLNormMuSigmaAndNorm01().SetParam(AfterOperation="Mean"),
            Sum=network.WeightedSum(1.0, 1.0)
        ).AddRoute(
            AddDictItem="Out",
            LossKL=(("Mu","LogOfVar"), "LossKL"),
            LossReconstruct=(["XPred", "XIn"], "LossReconstruct"),
            Sum=(
                [
                    "LossKL",
                    "LossReconstruct"
                ], 
                "LossTotal"
            )
        ).SetOut(Loss="LossTotal", LossKL="LossKL", LossReconstruct="LossReconstruct") \
        .SetDictIn().Init()

    Evaluator = DLUtils.Evaluator("PredAndTarget").SetLoss(Loss)
    EvaluationLog = DLUtils.EvaluationLog("PredAndTarget")


    Optimizer = DLUtils.Optimizer().GradientDescend().SGD(
            LR=0.01, Momentum=0.9, Nesterov=True
        ).Bind(Model=Model, Evaluator=Evaluator)
    
    # Optimizer = DLUtils.Optimizer().GradientDescend().Adam(
    #     LearningRate=0.01, Alpha=0.9, Beta=0.9
    # ).Bind(Model=Model, Evaluator=Evaluator)
    
    Task = DLUtils.Task("ImageClassification").SetType("MNIST").SetDataPath("~/Data/mnist.zip")
    Save = train.EpochBatchTrain.Save(TestNum=10, SaveDir=SaveDir)
    Test = train.EpochBatchTrain.Test(TestNum=10)
    
    Log = DLUtils.log.SeriesLog()

    BatchNum = Task.TrainBatchNum(BatchSize)
    TrainSession = DLUtils.EpochBatchTrainSession(
            EpochNum=EpochNum, BatchSize=BatchSize,
            BatchNum=BatchNum, SaveDir=SaveDir
        ).SetLog(Log).AddSubModule(
            Evaluator=Evaluator, EvaluationLog=EvaluationLog,
            Task=Task, Optimizer=Optimizer, 
            Test=Test, Save=Save,
            OnlineMonitor = train.Select1FromN.OnlineReporterMultiLoss(BatchNum//5) \
                .AddLogAndReportItem("Loss", Type="Float") \
                .AddLogAndReportItem("LossKL", Type="Float") \
                .AddLogAndReportItem("LossReconstruct", Type="Float")
        ).Bind(
            Model=Model,
            AnalOnline = train.EpochBatchTrain.EventAfterEveryEpoch().SetEvent(
                functools.partial(SampleImage, LatentUnitNum=LatentUnitNum)
            )
        ).Init().SetDevice(Device).Start().ToFile(SaveDir + "TrainSession.dat")

    del TrainSession
    del Model

    TrainSession = DLUtils.FromFile(SaveDir + "TrainSession.dat").Init()
    TrainSession.Bind(
        Model=DLUtils.LoadMostRecentlySavedModelInDir(SaveDir + "model-saved")
    )
    Anal = train.EpochBatchTrain.AnalysisAfterTrain().AddItem("LossBatch")
    TrainSession.SimulateAfterTrain(Anal)

