import DLUtils
import DLUtils.network as network

def vae_mnist(SaveDir="./example/vae_mnist/"):
    SaveDir = DLUtils.file.ToStandardPathStr(SaveDir)
    XNum = 28 * 28
    LatentUnitNum = 2
    Device = "cuda:0"
    Encoder = network.ModuleList(
        network.Reshape(-1, 28 * 28),
        network.MLP(
            (XNum, 100, LatentUnitNum * 2),
            NonLinear="ReLU", NonLinearOnLastLayer=False
        ), # Output: [BatchSize, Mu:LogOfVariance]
        network.SampleFromNormalDistribution(
            InNum = LatentUnitNum, InType="LogOfVariance"
        )
    )
    Encoder.PlotWeight(SaveDir=SaveDir + "weight/")
    
    Decoder = network.ModuleList(
        network.MLP(
            (LatentUnitNum, 100, XNum * 2), NonLinear="ReLU", NonLinearOnLastLayer=False
        ),
        network.SampleFromNormalDistribution(
            InNum = XNum, InType="LogOfVariance"
        ),
        network.Reshape(-1, 28, 28),
        network.Sigmoid()
    )

    VAE = network.ModuleSeries({
        "Encoder": Encoder, "Decoder": Decoder
    }).SetParam(OutType="All").Init().SetDevice(Device)
    
    # test input
    Z, XPred = VAE(
        DLUtils.ToTorchTensor(
            DLUtils.SampleFromUniformDistribution((10, 28, 28), -1.0, 1.0)
        ).to(Device)
    )
    XPred = DLUtils.ToNpArray(XPred)
    DLUtils.plot.PlotGreyImage(XPred[0], SaveDir + "image-decode/" + "test.png")
    
    
    
    Loss = network.ModuleSeries(
        network.ModuleParallel(
            
        ),
        network.Sum()
    )
    
    Task = DLUtils.Task("mnist")