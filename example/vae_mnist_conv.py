import torch
import functools

import DLUtils
import DLUtils.network as network
import DLUtils.loss as loss
import DLUtils.train as train

from .vae_mnist import SampleImage, PlotLabelClusters, PlotLatentSpace

def vae_conv_anal(SaveDir):
    TrainSession = DLUtils.FromFile(SaveDir + "TrainSession.dat").Init()
    Model=DLUtils.LoadMostRecentlySavedModelInDir(SaveDir + "model-saved").Init()
    TrainSession.Bind(
        Model=Model
    ).SetDevice("cuda:0")
    Anal = train.EpochBatchTrain.EventAfterTrain().SetEventList(
        PlotLatentSpace, PlotLabelClusters,
        functools.partial(SampleImage, LatentUnitNum=Model.Encoder.Mu.Param.Out.Num)
    )
    TrainSession.SimulateAfterTrain(Anal)

    AnalLoss = train.EpochBatchTrain.AnalysisAfterTrain().AddItem("LossBatch")
    TrainSession.SimulateAfterTrain(AnalLoss)

def vae_conv(SaveDir="./example/vae_conv/"): 
    Task = "Anal"
    if Task in ["Plot"]:
        Str = DLUtils.NpArray2Str(DLUtils.SampleFromUniformDistribution((10, 10)), Name="TestNpArray")
        DLUtils.Str2File(Str, "./example/np2str.txt")
        return
    if Task in ["Anal"]:
        vae_conv_anal(SaveDir)
        return

    SaveDir = DLUtils.file.ToStandardPathStr(SaveDir)
    ModelType = "Conv"
    DLUtils.file.ClearDir(SaveDir)
    XNum = 28 * 28
    LatentUnitNum = 2
    EpochNum = 30
    BatchSize = 128
    Device = "cuda:0"

    PreProcess = network.ModuleList().AddSubModule(
        #ToTorchTensor=network.NpArray2TorchTensor(Device=Device),
        # why norm to range (0.0, 1.0), not (-1.0, 1.0) ?
        # to make it like a probability, thus allowing binary cross entropy loss.
        Move2Device = network.MoveTensor2Device(),
        Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0),
    )
    # latent_dim = 2
    # encoder_inputs = keras.Input(shape=(28, 28, 1))
    # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Flatten()(x)
    # x = layers.Dense(16, activation="relu")(x)
    # z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    # encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    Encoder = network.ModuleGraph().AddSubModule(
            ExtractFeature=network.ModuleList().AddSubModule(
                Reshape1=network.Reshape(-1, 28, 28, 1),
                Permute = network.ChangeDimOrder(0, 3, 1, 2),
                Conv1=network.Conv2D(
                    InNum=1, OutNum=32, KernelSize=3, Stride=2,
                    Padding=(1, 1), NonLinear="ReLU"
                ), # Output: [BatchSize,14, 14, 32]
                Conv2=network.Conv2D(
                    InNum=32, OutNum=64, KernelSize=3, Stride=2,
                    Padding=(1, 1), NonLinear="ReLU"
                ), # Output: [BatchSize, 7, 7, 64]
                Reshape2 = network.Reshape(-1, 7 * 7 * 64),
                Linear1 = network.NonLinear(InNum=7 * 7 * 64, OutNum=16, NonLinear="ReLU")
            ),
            Mu=network.Linear(16, LatentUnitNum),
            LogOfVar=network.Linear(16, LatentUnitNum),
            Sampler=network.SampleFromNormalDistribution(InNum=LatentUnitNum, InType="LogOfVar")
        ).AddRoute(
            ExtractFeature=("X", "Y"),
            Mu=("Y", "Mu"),
            LogOfVar=("Y", "LogOfVar"),
            Sampler=(["Mu", "LogOfVar"], "Z")
        ).SetIn("X").SetOut("Mu", "LogOfVar", "Z").PlotWeight(SaveDir=SaveDir + "weight/")

    # latent_inputs = keras.Input(shape=(latent_dim,))
    # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    # x = layers.Reshape((7, 7, 64))(x)
    # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    # decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()

    Decoder = network.ModuleList().AddSubModule(
        Linear1=network.NonLinear(InNum=2, OutNum=64 * 7 * 7, NonLinear="ReLU"),
        Reshape1=network.Reshape(-1, 64, 7, 7),
        UpConv1=network.UpConv2D(
            InNum=64, OutNum=64, KernelSize=3,
            Stride=2, Padding=1, NonLinear="ReLU",
            OutputPadding=1
        ), # [BatchSize, 14, 14, 64]
        UpConv2=network.UpConv2D(
            InNum=64, OutNum=32, KernelSize=3, 
            Stride=2, Padding=1, NonLinear="ReLU",
            OutputPadding=1
        ), # [BatchSize, 28, 28, 32]
        Conv3=network.Conv2D(
            InNum=32, OutNum=1, KernelSize=3, 
            Stride=1, Padding=1, NonLinear="Sigmoid",
            OutputPadding=0
        ), # [BatchSize, 28, 28, 1]
        Reshape2=network.Reshape(-1, 28, 28)
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
    TestIn = DLUtils.SampleFromUniformDistribution((10, 28, 28), -1.0, 1.0)
    TestIn = DLUtils.ToTorchTensor(TestIn).to(Device)
    TestOut = Model(TestIn)

    Out = DLUtils.param(TestOut)
    Z, XPred = Out.Z, Out.XPred
    XPred = DLUtils.ToNpArray(XPred)
    DLUtils.plot.PlotGreyImage(XPred[0], SaveDir + "image-decode/" + "test.png")
    Loss = network.ModuleGraph() \
        .AddSubModule(
            LossReconstruct=loss.CrossEntropy2Class(AfterOperation="Mean"),
            #LossReconstruct=loss.MSELoss(AfterOperation="Mean"),
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

    # Optimizer = DLUtils.Optimizer().GradientDescend().SGD(
    #         LR=0.01, Momentum=0.9, Nesterov=True
    #     ).Bind(Model=Model, Evaluator=Evaluator)
    
    Optimizer = DLUtils.Optimizer().GradientDescend().Adam(
        LearningRate=0.001, Alpha=0.9, Beta=0.998
    ).Bind(Model=Model, Evaluator=Evaluator)
    
    Task = DLUtils.Task().ImageClassification().MNIST().SetDataPath("~/Data/mnist.zip")
    Save = train.EpochBatchTrain.Save(Num=10, SaveDir=SaveDir)
    Test = train.EpochBatchTrain.Test(TestNum="All")
    
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
                .AddLogAndReportItem("LossTotal", Type="Float") \
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

    torch.cuda.empty_cache()
    vae_conv_anal(SaveDir=SaveDir)

def SampleImage(Dict, LatentUnitNum):
    Model, SaveDir = Dict.Model, Dict.SaveDir
    # test input
    #Z = DLUtils.SampleFromGaussianDistribution((10, LatentUnitNum))
    import numpy as np
    Z = np.full((10, LatentUnitNum), 0.1)
    #Z = np.full((10, LatentUnitNum), 0.0)
    XPred = Model.Decoder(
        DLUtils.ToTorchTensor(Z).to(Model.Device)
    )
    XPred = DLUtils.ToNpArray(XPred)

    if len(Z[0]) < 4:
        ZStr = "(" + ", ".join(["%.2f"%ZElement for ZElement in Z[0]]) + ")"
    else:
        ZStr = "(" + ", ".join(["%.2f"%ZElement for ZElement in Z[0, 0:4]]) + "...)"
    DLUtils.plot.PlotGreyImage(XPred[0], SaveDir + "image-decode/" + "Epoch%d-Z=%s.png"
        %(Dict.EpochIndex, ZStr))

