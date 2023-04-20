import DLUtils
import matplotlib.pyplot as plt
import numpy as np
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

# https://keras.io/examples/generative/vae/
def PlotLatentSpace(Dict, n=30, figsize=15, Device="cuda:0", ShowImage=True):
    Model = Dict.Model
    # display a n*n 2D manifold of digits
    Model.SetDevice(Device)
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]]) # [1, (z0, z1)]
            z_sample = DLUtils.ToTorchTensor(z_sample).to(Device)
            x_decoded = Model.Decoder(z_sample)
            x_decoded = DLUtils.ToNpArray(x_decoded)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    fig = plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    Fig = plt.gcf()
    if ShowImage:
        plt.show(block=False)
    SaveDir = Dict.SaveDir
    DLUtils.plot.SaveFigForPlt(Fig=Fig, SavePath=SaveDir + "anal-visual/" + "latent-space.png")

# https://keras.io/examples/generative/vae/
def PlotLabelClusters(Dict, ShowImage=True):
    Model = Dict.Model
    # display a 2D plot of the digit classes in the latent space
    Data = DLUtils.Task().ImageClassification().MNIST().SetDataPath("~/Data/mnist.zip").TrainData(BatchSize=128)
    # TrainData = np.expand_dims(TrainData.Image, -1).astype("float32") / 255
    LabelList = []
    ZMeanList = []
    Device = Dict.getdefault("Device", "cuda:0")
    for BatchIndex in range(Data.BatchNum()):
        In, OutTarget = Data.Get(BatchIndex)
        In = DLUtils.ToTorchTensor(In).to(Device)
        In = Model.PreProcess(In)
        Z = Model.Encoder(In)
        ZMean = Z["Mu"]
        ZMean = DLUtils.ToNpArray(ZMean)
        LabelList.append(OutTarget)
        ZMeanList.append(ZMean)
    Label = np.concatenate(LabelList, axis=0)
    ZMean = np.concatenate(ZMeanList, axis=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(ZMean[:, 0], ZMean[:, 1], c=Label)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    SaveDir = Dict.SaveDir
    Fig = plt.gcf() # gcf: get current figure
    if ShowImage:
        plt.show(block=False)
    DLUtils.plot.SaveFigForPlt(Fig=Fig, SavePath=SaveDir + "anal-visual/" + "label-clusters.png")
