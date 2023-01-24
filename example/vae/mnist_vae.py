import DLUtils
import DLUtils.network as network

def mnist_vae(SaveDir="./example/vae/mnist/"):
    SaveDir = DLUtils.file.ToStandardPathStr(SaveDir)

    LatentUnitNum = 2

    Encoder = network.ModuleList(
        network.MLP(
            (28 * 28, 100, LatentUnitNum * 2),
            NonLinear="ReLU"
        ),
        network.SampleFromNormalDistribution(
            InNum = LatentUnitNum, InType="LogOfVariance"
        )
    )
    

    Encoder.PlotWeight(SaveDir=SaveDir + "weight/")

    Decoder = network.MLP(
        (4, ), NonLinear="ReLU"
    )

    Out = Encoder(
        #DLUtils.NoiseImage(28, 28)
        DLUtils.ToTorchTensor(
            DLUtils.SampleFromUniformDistribution((10, 28 * 28), -1.0, 1.0)
        )
    )
    Task = DLUtils.Task("mnist")
    