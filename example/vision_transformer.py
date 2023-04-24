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


class VisionTransformer(DLUtils.module.AbstractNetwork):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("LayerNum"): "Layer.Num",
        ("TokenSize", "FeatureSize"): ("Token.Size"),
        ("QKSize"): "MSA.Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "MSA.Attention.V.Size", # total size. not size of each head.
        ("HeadNum"): "MSA.Attention.Head.Num",
        ("MLPSize"): "MLP.HiddenLayer.Size",
        ("NonLinear", "MLPNonLinear"): "MLP.NonLinear",
        ("PatchNumX", "PatchXNum"): "Patch.NumX",
        ("PatchNumY", "PatchYNum"): "Patch.NumY",
        ("ClassNum", "NumClass"): "Task.Class.Num"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def Receive(self, X):
        X = self.Preprocess(X) # (BatchSize, TokenNum, TokenSize)
        X = torch.concat([self.ImageToken, X], dim=1) # (BatchSize, TokenNum + 1, TokenSize)
        X += self.PositionEmbedding
        X = self.TransformerEncoder(X) # multi-head self attention
        Y = self.ClassificationHead(X)
        return Y
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.TokenSize = Param.Token.Size
        self.PatchNumX = Param.Patch.NumX
        self.PatchNumY = Param.Patch.NumY
        if self.IsInit():
            self.AddSubModule(
                Preprocess=network.ModuleList().AddSubModule(
                    # input: (BatchSize, ChannelNum, Height, Width)
                    Move2Device=network.MoveTensor2Device(),
                    Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0),
                    Crop=network.CenterCrop(224, 224),
                    Image2PatchList=network.Image2PatchList(
                        PatchNumX=Param.Patch.NumX, PatchNumY=Param.Patch.NumY,
                        ImageHeight=224, ImageWidth=224
                    ) # (BatchSize, PatchListSize, Height * Width * ChannelNum)
                ),
                TransformerEncoder=network.TransformerEncoder(
                    LayerNum = 6,
                    TokenSize = Param.Token.Size,
                    QKSize = Param.MSA.Attention.QK.Size,
                    VSize = Param.MSA.Attention.V.Size,
                    HeadNum = Param.MSA.Attention.Head.Num,
                    MLPNonLinear = Param.MLP.NonLinear,
                    MLPSize = Param.MLP.HiddenLayer.Size
                )
            )
            Param.ImageToken.Data = DLUtils.SampleFromNormalDistribution(
                Shape=(1, 1, self.TokenSize),
                Mean=0.0, Std=1.0
            )
            self.RegisterTrainParam(
                Name="ImageToken", # token representing the entire imge.
                Path="ImageToken.Data",
            )
            Param.PositionEmbedding.Data = DLUtils.SampleFromNormalDistribution(
                Shape=(1, self.PatchNumX * self.PatchNumY + 1, self.TokenSize),
                Mean=0.0, Std=1.0
            )
            self.RegisterTrainParam(
                Name="PositionEmbedding",
                Path="PositionEmbedding.Data"
            )
            
            self.AddSubModule(
                ClassificationHead=network.ModuleList(
                    LayerNorm=network.LayerNorm(
                        NormShape=(self.TokenSize)
                    ),
                    LinearLayer=network.LinearLayer(
                        InSize=self.TokenSize,
                        OutSize = Param.Task.Class.Num
                    )
                )
            )
        return super().Init(IsSuper=True, IsRoot=IsRoot)

def vision_transformer_imagenet_1k_patch_test(SaveDir, PatchNum, Device):


    TestImage = DLUtils.Jpg2NpArray(
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00011424.JPEG"
        "./example/vit_imagenet/test/test_image.png"
    )
    print("TestImage shape: %s. dtype: %s"%(str(TestImage.shape), str(TestImage.dtype)))
    # return
    PatchNum = 4
    Preprocess=network.ModuleList().AddSubModule(
        # input: (BatchSize, ChannelNum, Height, Width)
        Move2Device=network.MoveTensor2Device(),
        Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0),
        Crop=network.CenterCrop(224, 224),
        Image2PatchList=network.Image2PatchList(
            PatchNumX=PatchNum, PatchNumY=PatchNum,
            ImageHeight=224, ImageWidth=224
        ) # (BatchSize, PatchListSize, Height * Width * ChannelNum)
    ).Init().SetDevice(Device)
    
    PatchList = Preprocess(
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
    LatentUnitNum = 2
    EpochNum = 30
    BatchSize = 128
    Device = "cuda:1"
    PatchNum = 16


    # load imagenet-1k / ILSVRC 2012
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm
    model = torchvision.models.resnet50(weights="DEFAULT")
    model.eval().to(Device) # Needs CUDA, don't bother on CPUs
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_val = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    from task import ImageNet1k
    Task = ImageNet1k(
        DataPath="~/Data/imagenet",
        Mode="Validation",
        Transform=transform_val
    ).Init()
    DataLoader = Task.DataLoader(
                BatchSize=64, # may need to reduce this depending on your GPU 
                ThreadNum=8, # may need to reduce this depending on your num of CPUs and RAM
                Shuffle=False,
                DropLast=False,
                PinMemory=True
            )
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(DataLoader):
            y_pred = model(x.to(Device))
            correct += (y_pred.argmax(axis=1) == y.to(Device)).sum().item()
            total += len(y)
    print(correct / total)

    QKVSize = 512
    HeadNum = 16
    MLPSize = 512
    TokenSize = 224 * 224 * 3 // (PatchNum * PatchNum)
    ViT = VisionTransformer(
        LayerNum = 6,
        TokenSize = TokenSize,
        QKSize = QKVSize,
        VSize = QKVSize,
        HeadNum = HeadNum,
        MLPNonLinear = "ReLU", 
        MLPSize = MLPSize,
        PatchNumX = PatchNum,
        PatchNumY = PatchNum,
        ClassNum=1000
    ).Init().SetDevice(Device)

    # test image2patch
    vision_transformer_imagenet_1k_patch_test(
        SaveDir, PatchNum, Device
    )

    TestImage = DLUtils.Jpg2NpArray(
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00011424.JPEG"
        # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/train/n01820546/n01820546_4054.JPEG"
        "./example/vit_imagenet/test/test_image.png"
    )

    # test input image
    TestOut = ViT(
        DLUtils.ToTorchTensor(TestImage).to(Device).permute(2, 0, 1).unsqueeze(0)
    )

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

    return

    Evaluator = DLUtils.Evaluator("PredAndTarget").SetLoss(Loss)
    EvaluationLog = DLUtils.EvaluationLog("PredAndTarget")
    
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