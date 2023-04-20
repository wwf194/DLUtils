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



def init_ViT(
        # dataset setting
        ClassNum = 1000,
        ImageSize = 256,
        PatchSize = 32,
        # network setting
        LayerNum = 6,
        TokenFeatureNum = 1024,
        MSAHeadNum = 16,
        MLPHiddenUnitNum = 2048,
    ):
    assert ImageSize % PatchSize == 0
    
    MLPInputNum = TokenFeatureNum
    MLPOutputNum = TokenFeatureNum

    MultiHeadSelfAttentionLayerList = {}
    for LayerIndex in range(LayerNum):
        MultiHeadSelfAttentionLayer = network.ModuleGraph().AddSubModule(
            MSA = network.MultiHeadSelfAttention(   
            )
            , # multi-head self attention
            MLP = network.MLP(
                UnitNum = [
                    MLPInputNum,
                    MLPHiddenUnitNum,
                    MLPOutputNum,
                ],
                BiasOnLastLayer=False,
                NonLinearOnLastLayer=False
            )
        )
        MultiHeadSelfAttentionLayerList.append(MultiHeadSelfAttentionLayer)

    ViT = network.ModuleGraph()
    for LayerIndex in range(LayerNum):
        ViT.AddSubModule(
            "MultiHeadSelfAttentionLayer%d"%LayerIndex,
            MultiHeadSelfAttentionLayerList[LayerIndex]
        )
    return ViT, ViTClassificationHeader


def vision_transformer_imagenet_1k_patch_test(SaveDir, Preprocess, PatchNum, Device):
    TestImage = DLUtils.Jpg2NpArray(
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00011424.JPEG"
        "./example/vit_imagenet/test/test_image.png"
    )
    print("TestImage shape: %s. dtype: %s"%(str(TestImage.shape), str(TestImage.dtype)))
    # return
    PatchList = Preprocess.Receive(
        DLUtils.ToTorchTensor(TestImage).to(Device).permute(2, 0, 1).unsqueeze(0)
    )
    # DLUtils.DeleteAllFilesAndSubFolders("./example/vit_imagenet/test/")
    PatchList = DLUtils.ToNpArray(PatchList)
    DLUtils.NpArray2ImageFile(TestImage, "./example/vit_imagenet/test/patch_all.png")

    for Index in range(PatchNum * PatchNum):
        DLUtils.NpArray2ImageFile(
            PatchList[0][Index].reshape(224 // PatchNum, 224 // PatchNum, 3),
            "./example/vit_imagenet/test/patch_%d.png"%Index
        )
    return

def vision_transformer_imagenet_1k(SaveDir="./example/vit_imagenet/"):
    SaveDir = DLUtils.file.ToStandardPathStr(SaveDir)
    # DLUtils.file.ClearDir(SaveDir)
    XNum = 28 * 28
    LatentUnitNum = 2
    EpochNum = 30
    BatchSize = 128
    Device = "cuda:0"
    PatchNum = 4
    Preprocess = network.ModuleList().AddSubModule(
        # input: (BatchSize, ChannelNum, Height, Width)
        Move2Device=network.MoveTensor2Device(),
        Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0),
        Crop=network.CenterCrop(224, 224),
        Image2PatchList=network.Image2PatchList(
            PatchNumX=PatchNum, PatchNumY=PatchNum,
            ImageHeight=224, ImageWidth=224
        ) # (BatchSize, PatchListSize, Height * Width * ChannelNum)
    )
    
    HeadNum = 16
    QKVSize = 512
    PatchFeatureNum = 224 * 224 * 3 // (PatchNum * PatchNum)
    ViT = network.ModuleList().AddSubModule(
        Preprocess=Preprocess,
        MSA=network.MultiHeadSelfAttention(
            QKSize = QKVSize,
            VSize = QKVSize,
            HeadNum = HeadNum,
            InNum = PatchFeatureNum,
            OutNum = PatchFeatureNum
        )
    ).Init().SetDevice(Device)

    # test image2patch
    vision_transformer_imagenet_1k_patch_test(
        SaveDir, Preprocess, PatchNum, Device
    )

    TestImage = DLUtils.Jpg2NpArray(
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00011424.JPEG"
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/train/n01820546/n01820546_4054.JPEG"
        "./example/vit_imagenet/test/test_image.png"
    )

    Out = ViT(
        DLUtils.ToTorchTensor(TestImage).to(Device).permute(2, 0, 1).unsqueeze(0)
    )
    return
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
    
def AfterTrainAnalysis(Dict):
    PlotLatentSpace(Dict)
    PlotLabelClusters(Dict)