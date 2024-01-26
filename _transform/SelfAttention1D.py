import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")

class SelfAttention1D(DLUtils.module.AbstractModule):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data  = self.data
        cache = self.cache

        assert HasAttrs(param, "In.Num")
        assert HasAttrs(param, "Out.Num")
        EnsureAttrs(param, "Attention.Feature.Num", default=Param.In.Num)
        SetAttrs(param, "Weight.Input2Query.Size", value=[Param.In.Num, param.Attention.Feature.Num])
        SetAttrs(param, "Weight.Input2Key.Size",   value=[Param.In.Num, param.Attention.Feature.Num])
        if HasAttrs(param, "Out.Num"):
            SetAttrs(param, "Weight.Input2Value.Size", value=[Param.In.Num, Param.Out.Num])
        elif HasAttrs(param, "Weight.Input2Value.Size"):
            SetAttrs(param, "Out.Num", value=param.Weight.Input2Value.Size[1])
        else:
            raise Exception()

        if cache.IsInit:
            data.Input2Query = DLUtils.transform.CreateWeight2D(param.Weight.Input2Query)
            data.Input2Key   = DLUtils.transform.CreateWeight2D(param.Weight.Input2Key)
            data.Input2Value = DLUtils.transform.CreateWeight2D(param.Weight.Input2Value)
        else:
            data.Input2Query = DLUtils.ToTorchTensor(data.Input2Query)
            data.Input2Key   = DLUtils.ToTorchTensor(data.Input2Key)
            data.Input2Value = DLUtils.ToTorchTensor(data.Input2Value)

        cache.Tensor = DLUtils.EmptyPyObj()
        
        cache.TensorDict["Input2Query"] = cache.Input

        cache.TrainParam.append([data, "Input2Query", data.Input2Query])
        cache.TrainParam.append([data, "Input2Key",   data.Input2Key])
        cache.TrainParam.append([data, "Input2Value", data.Input2Value])
        cache.AttentionCoefficient = 1.0 / param.Attention.Feature.Num ** 0.5

        return self
    def SplitHead(self, Data, HeadNum):
        BatchSize = Data.shape[0]
        TokenNum = Data.shape[1]
        # FeatureNumTotal = Data.shape[2]
        Data = Data.view(BatchSize, TokenNum, HeadNum, -1)
        Data = Data.permute(0, 2, 1, 3)
        return Data
    def forward(self, Input, log):
        # Input: [BatchSize, TokenNum, InNum]
        data = self.data
        cache = self.cache
        Query = torch.mm(Input, data.Input2Query) # [BatchSize, TokenNum, AttentionFeatureNum]
        Key = torch.mm(Input, data.Input2Key)     # [BatchSize, TokenNum, AttentionFeatureNum]
        Value = torch.mm(Input, data.Input2Value) # [BatchSize, TokenNum, OutNum]

        self.LogCache("Attention.Key", Key, "Attention", log=log)
        self.LogCache("Attention.Query", Query, "Attention", log=log)
        self.LogCache("Attention.Value", Value, "Attention", log=log)

        if cache.MultiHead:
            QueryMultiHead = torch.mm(Query, data.Query2MultiHeadQuery) # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
            KeyMultiHead   = torch.mm(Key,   data.Query2MultiHeadKey)   # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
            ValueMultiHead = torch.mm(Value, data.Query2MultiHeadValue) # [BatchSize, TokenNum, OutputFeatureNum]

            QueryMultiHead = self.SplitHead(QueryMultiHead, cache.HeadNum) # [BatchSize, HeadNum, TokenNum, AttentionPerHeadFeatureNum]
            KeyMultiHead   = self.SplitHead(KeyMultiHead,   cache.HeadNum)
            ValueMultiHead = self.SplitHead(ValueMultiHead, cache.HeadNum)

            AttentionMultiHead = torch.einsum("bhtf,bhtf->bht", QueryMultiHead, KeyMultiHead) # [BatchSize, HeadNum, TokenNum, TokenNum]
            AttentionMultiHead *= cache.AttentionPerHeadFeatureNum
            AttentionMultiHead = F.softmax(AttentionMultiHead, axis=3)

            OutputMultiHead = torch.einsum("bhtu,bhuo->bhto", AttentionMultiHead, ValueMultiHead)
            
            self.LogCache("AttentionMultiHead",       AttentionMultiHead, "Attention", log=log)
            self.LogCache("AttentionMultiHead.Query", QueryMultiHead,     "Attention", log=log)
            self.LogCache("AttentionMultiHead.Key",   KeyMultiHead,       "Attention", log=log)
            self.LogCache("AttentionMultiHead.Value", ValueMultiHead,     "Attention", log=log)
            self.LogCache("OutputMultiHead",          OutputMultiHead,    "Activity",  log=log)
            return OutputMultiHead
        else:
            Attention = torch.bmm(Query, Key.permute(0, 2, 1)) ** cache.AttentionFeatureNum # [BatchSize, TokenNum, TokenNum]
            Attention = F.softmax(Attention, dim=2) # [BatchSize, TokenNum, TokenNum]
            Output = torch.bmm(Attention, Value) # [BatchSize, TokenNum, OutNum]

            self.LogCache("Attention", Attention, "Attention", log=log)
            self.LogCache("Output", Output, "Activity", log=log)
            return Output

__MainClass__ = SelfAttention1D
DLUtils.transform.SetMethodForTransformModule(__MainClass__)