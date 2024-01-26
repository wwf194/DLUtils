import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
else:
    torch = DLUtils.GetLazyTorch()

class DataFetcherForEpochBatchTrain(torch.utils.data.Dataset):
    def __init__(self):
        self.Device = "cpu"
        super().__init__()
    def __len__(self):
        # must be overwritten
        raise Exception()
    def __getitem__(self, Index):
        # must be overwritten
        raise Exception()
    def SetDevice(self, Device):
        self.Device = Device

class DataLoaderForEpochBatchTrain(torch.utils.data.DataLoader, DLUtils.module.AbstractModule):
    ParamMap = DLUtils.IterableKeyToElement({
        ("BatchSize"): "Batch.Size",
        ("BatchNum", "NumBatch"): "Batch.Num",
        ("DropLast"): "Batch.DropLast",
        ("Shuffle"): "Batch.Shuffle",
        ("ThreadNum", "FetchThreadNum"): "Thread.Num",
        # for parallel distributed dataloader
        ("NodeIndex"): "Parallel.Node.Index",
        ("NodeNum"): "Parallel.Node.Num",
        ("Parallel"): "Parallel.Enable",
        ("DropLastWhenSplitDatasetForEachNode"): "Parallel.Dataset.DropLastWhenSplit"
        # when dataset sample num is not divisible by node num.
    })
    TorchSamplerParamMap = DLUtils.IterableKeyToElement({
        ("Seed"): "seed",
        ("Batch.Shuffle"): "shuffle",
        ("Parallel.Dataset.DropLastWhenSplit"): "drop_last",
        ("Parallel.Node.Num"): "num_replicas", # total node num
        ("Parallel.Node.Index"): "rank", # index of current node
    })
    TorchDataLoaderParamMap = DLUtils.IterableKeyToElement({
        ("Thread.Num"): "num_workers",
        ("PinMemory"): "pin_memory", # forbid caching data in memory on disk.
        ("Batch.Size"): "batch_size",
        ("Batch.Shuffle"): "shuffle",
        ("Batch.DropLast"): "drop_last",
    })
    def GetTorchDataLoaderParam(self):
        Dict = {}
        Param = self.Param
        for NameInParam, NameForTorch in self.TorchDataLoaderParamMap.items():
            if Param.hasattr(NameInParam):
                Dict[NameForTorch] = Param.getattr(NameInParam)
            else:
                pass
        return Dict
    def GetTorchSamplerParam(self):
        Dict = {}
        Param = self.Param
        for NameInParam, NameForTorch in self.TorchSamplerParamMap.items():
            if Param.hasattr(NameInParam):
                Dict[NameForTorch] = Param.getattr(NameInParam)
            else:
                pass
        return Dict 
    def __init__(self, DataFetcher=None, Sampler=None, **Dict):
        self.DataFetcher = DataFetcher
        DLUtils.module.AbstractModule.__init__(self, **Dict)
        # DataFetcher: get sample input and output according to index.
        if DataFetcher is not None:
            self.SetDataFetcher(DataFetcher)
        Param = self.Param
        Param.Parallel.setdefault("Enable", False)
        Param.Thread.setdefault("Num", 0)

        if Param.Parallel.Enable:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=self.DataFetcher,
                **self.GetTorchSamplerParam()
            )
        else:
            sampler = None

        DataLoaderDict = self.GetTorchDataLoaderParam()
        if sampler is not None:
            DataLoaderDict.pop("shuffle", None) # pop without throwing exeption
        torch.utils.data.DataLoader.__init__(
            self,
            dataset=self.DataFetcher,
            **DataLoaderDict,
            # Setting num_workers > 1 might severely slow down speed.
            sampler=sampler
        )
    def SetDataFetcher(self, DataFetcher):
        self.DataFetcher = DataFetcher
    def BeforeEpoch(self, Dict):
        self.Reset()
    def AfterEpoch(self, Dict):
        self.Reset()
    def GetBatch(self, BatchIndex=None):
        In, OutTarget = next(self.Iter)
        return In.to(self.Device), OutTarget.to(self.Device)
    Get = GetNextBatch = GetBatch
    # single device situation
    def SetDevice(self, Device, IsRoot=True):
        self.DataFetcher.SetDevice(Device)
        self.Device = Device
        return self
    def GetBatchNum(self):
        return self.BatchNum
    def Reset(self):
        self.Iter = iter(self)
        return self
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Thread.setdefault("Num", 1)
        assert isinstance(Param.Thread.Num, int)
        assert hasattr(self, "DataFetcher")

        assert Param.Batch.hasattr("Size")
        self.BatchSize = Param.Batch.Size

        # calculate batch num
        # if not Param.Batch.hasattr("Num"):
        #     DataNum = self.DataFetcher.DataNum
        #     Param.Batch.Num = DataNum // self.BatchSize
        #     if DataNum % self.BatchSize > 0:
        #         Param.Batch.Num += 1 
        Param.Batch.Num = len(iter(self))
        self.BatchNum = Param.Batch.Num
        
        # device setting
        if hasattr(self.DataFetcher, "Device"):
            self.Device = self.DataFetcher.Device

        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.Reset()
        self.Device = "cpu" # default
        return self