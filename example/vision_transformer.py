# nohup python __main__.py -t example vit imagenet &>./example/vit_imagenet/log.txt &

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
    # preprocess: divide image to patches, flatten each patch to 1D.
    # multiple transformer layer with same in and out size.
    ParamMap = DLUtils.IterableKeyToElement({
        ("LayerNum"): "Layer.Num",
        ("TokenSize", "FeatureSize"): ("Token.Size"),
        ("QKSize"): "MSA.Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "MSA.Attention.V.Size", # total size. not size of each head.
        ("HeadNum"): "MSA.Attention.Head.Num",
        ("MLPSize"): "MLP.HiddenLayer.Size",
        ("NonLinear", "MLPNonLinear"): "MLP.NonLinear",
        ("PatchNumX", "PatchXNum"): "Patch.NumX",
        ("PatchNumY", "PatchYNum"): "Patch.NumY",
        ("ClassNum", "NumClass"): "Task.Class.Num",
        ("DropOut"): "DropOut.Probability",
        ("DropOutInplace", "DropOutInPlace"): "DropOut.InPlace"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def Receive(self, X):
        # X: (BatchSize, ChannelNum, Height, Width)
        BatchSize = X.size(0)
        X = self.Preprocess(X) # (BatchSize, TokenNum, TokenSize)
        ImageToken = self.ImageToken.expand(BatchSize, -1, -1)
        X = torch.concat([ImageToken, X], dim=1) # (BatchSize, TokenNum + 1, TokenSize)
        X += self.PositionEmbedding
        X = self.DropOut(X)
        X = self.TransformerEncoder(X) # multi-head self attention
            # (BatchSize, TokenNum + 1, TokenSize)
        X = X[:, 0, :] # (BatchSize, TokenSize)
        Y = self.ClassificationHead(X)
        # (BatchSize, ClassNum)
        return {"Out": Y}
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.TokenSize = Param.Token.Size
        self.PatchNumX = Param.Patch.NumX
        self.PatchNumY = Param.Patch.NumY
        if self.IsInit():
            self.AddSubModule(
                Preprocess=network.ModuleList().AddSubModule(
                    # input: (BatchSize, ChannelNum, Height, Width)
                    # Move2Device=network.MoveTensor2Device(),
                    # Norm=network.ShiftRange(0.0, 256.0, 0.0, 1.0),
                    # Crop=network.CenterCrop(224, 224),
                    Image2PatchList=network.Image2PatchList(
                        PatchNumX=Param.Patch.NumX, PatchNumY=Param.Patch.NumY,
                        ImageHeight=224, ImageWidth=224
                    ) # (BatchSize, PatchListSize, Height * Width * ChannelNum)
                ),
                TransformerEncoder=network.TransformerEncoder(
                    LayerNum = Param.Layer.Num,
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
            
            # dropout setting
            self.SetDropOutInit()
        
        # dropout setting
        self.SetDropOut()
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
        # Move2Device=network.MoveTensor2Device(),
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
    BatchSize = 256
    Device = "cuda:1"
    
    # load imagenet-1k / ILSVRC 2012
    from torchvision import transforms
    import torch
    import torchvision
    from tqdm import tqdm
    # model = torchvision.models.resnet50(weights="DEFAULT")
    # model.eval().to(Device) # Needs CUDA, don't bother on CPUs
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
    ).Init().SetDevice(Device)

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for x, y in tqdm(DataLoader):
    #         y_pred = model(x.to(Device))
    #         correct += (y_pred.argmax(axis=1) == y.to(Device)).sum().item()
    #         total += len(y)
    # print(correct / total)

    QKVSize = 512
    HeadNum = 16
    MLPSize = 512
    PatchNum = 14
    TokenSize = 224 * 224 * 3 // (PatchNum * PatchNum)
    ViT = Model = VisionTransformer(
        LayerNum = 6,
        TokenSize = TokenSize,
        QKSize = QKVSize,
        VSize = QKVSize,
        HeadNum = HeadNum,
        MLPNonLinear = "ReLU", 
        MLPSize = MLPSize,
        PatchNumX = PatchNum,
        PatchNumY = PatchNum,
        ClassNum=1000,
        DropOut=0.1
    ).Init().SetDevice(Device).ReportModelSizeInFile(SaveDir + "./model-size.json")

    # test image2patch
    vision_transformer_imagenet_1k_patch_test(
        SaveDir, PatchNum, Device
    )

    # TestImage = DLUtils.Jpg2NpArray(
    #     # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00011424.JPEG"
    #     # "~/Data/imagenet/ILSVRC/Data/CLS-LOC/train/n01820546/n01820546_4054.JPEG"
    #     "./example/vit_imagenet/test/test_image.png"
    # )
    TrainData = Task.DataLoader(
                Type="Train",
                BatchSize=BatchSize, # may need to reduce this depending on your GPU 
                ThreadNum=8, # may need to reduce this depending on your num of CPUs and RAM
                Shuffle=True,
                DropLast=False,
                PinMemory=True
            )

    ValidationData = Task.DataLoader(
                Type="Validation",
                BatchSize=BatchSize, # may need to reduce this depending on your GPU 
                ThreadNum=8, # may need to reduce this depending on your num of CPUs and RAM
                DropLast=False,
            )

    # Image, ClassIndex = TrainData.GetNextBatch() # (BatchSize, ChannelNum, Height, Width)    
    # # test input image
    # TestOut = ViT(
    #     Image
    # )

    torch.cuda.empty_cache()

    Loss = network.ModuleGraph() \
        .AddSubModule(
            ClassIndex2OneHot=network.Index2OneHot(1000),
            LossClassification=loss.SoftMaxAndCrossEntropy(AfterOperation="Mean"),
            Sum=network.WeightedSum(1.0),
            # Move2Device=network.MoveTensor2Device(Device)
        ).AddRoute(
            AddDictItem="Out",
            # LossKL=(("Mu","LogOfVar"), "LossKL"),
            # LossReconstruct=(["XPred", "XIn"], "LossReconstruct"),
            # Move2Device=("OutTarget", "OutTarget"),
            ClassIndex2OneHot=("OutTarget", "OutTargetProb"),
            LossClassification=(("Out", "OutTargetProb"), "LossClassification"),
            Sum=(
                [
                    "LossClassification",
                ], 
                "LossTotal"
            )
        ).SetDictOut(Loss="LossTotal") \
        .SetDictIn().Init()

    Evaluator = DLUtils.Evaluator("ImageClassification").SetLoss(Loss)
    EvaluationLog = DLUtils.EvaluationLog("ImageClassification")
    
    Optimizer = DLUtils.Optimizer().GradientDescend().Adam(
        LearningRate=0.001, Alpha=0.9, Beta=0.999,
        WeightDecay=0.1
    ).Bind(Model=Model, Evaluator=Evaluator)
    
    Save = train.EpochBatchTrain.Save(Num=10, SaveDir=SaveDir)
    Validate = train.EpochBatchTrain.Validate(
        TestNum="All", TriggerEventBeforeTrain=True
    ).AddSubModule(
        OnlineMonitor = train.Select1FromN.OnlineReporterMultiLoss(
            BatchInterval=10
            # NumPerEpoch=5
        ) \
        .AddLogAndReportItem(WatchName="Loss", Type="Float") \
        .AddLogItem("NumTotal") \
        .AddLogItem("NumCorrectTop5") \
        .AddLogItem("NumCorrect") \
        .AddReportItem(LogName=("NumCorrect", "NumTotal"), ReportName="AccTop1", Type="Acc") \
        .AddReportItem(LogName=("NumCorrectTop5", "NumTotal"), ReportName="AccTop5", Type="Acc")
    )
    Log = DLUtils.log.SeriesLog()

    # BatchNum = DataLoader.BatchNum(BatchSize)
    TrainSession = DLUtils.EpochBatchTrainSession(
            EpochNum=EpochNum, BatchSize=BatchSize, SaveDir=SaveDir
        ).SetLog(Log).AddSubModule(
            Evaluator=Evaluator, EvaluationLog=EvaluationLog,
            Task=Task, Optimizer=Optimizer, 
            Validate=Validate, Save=Save,
            OnlineMonitor = train.Select1FromN.OnlineReporterMultiLoss(
                BatchInterval=10
                # NumPerEpoch=5
            ) \
            .AddLogAndReportItem(WatchName="Loss", Type="Float") \
            .AddLogItem("NumCorrect") \
            .AddLogItem("NumCorrectTop5") \
            .AddLogItem("NumTotal") \
            .AddReportItem(LogName=("NumCorrect", "NumTotal"), ReportName="AccTop1", Type="Acc") \
            .AddReportItem(LogName=("NumCorrectTop5", "NumTotal"), ReportName="AccTop5", Type="Acc")
        ).Bind(
            Model=Model,
            TrainData=TrainData.Reset(),
            ValidationData=ValidationData.Reset()
            # AnalOnline = train.EpochBatchTrain.EventAfterEveryEpoch().SetEvent(
            #     functools.partial(SampleImage, LatentUnitNum=LatentUnitNum)
            # )
        ).Init().SetDevice(Device).Start().ToFile(SaveDir + "TrainSession.dat")




    del TrainSession
    del Model

def AfterTrainAnalysis(Dict):
    PlotLatentSpace(Dict)
    PlotLabelClusters(Dict)