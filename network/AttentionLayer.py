# import torch
# import math

# import DLUtils

# import DLUtils
# class Transformer(DLUtils.module.AbstractNetwork):
#     def __init__(self, SubModule=None):
#         super().__init__()
#         Param = self.Param
#         Param._CLASS = "DLUtils.network.Transformer"
#         if SubModule is None:
#             self.AddSubModule("SubModule", SubModule)
#     def Receive(self, X):
#         # X: [BatchSize, TokenNum, FeatureSize]
        
#         Y = self.Attention(X)
#         Y = X + Y # Residual Adding
#         Y = self.LayerNorm1(Y)

#         Z = self.MLP(Y)
#         Z = Y + Z
#         Z = self.LayerNorm2(Y)
#         return Y
#     def Init(self, IsSuper=False, IsRoot=True):
#         Param = self.Param
#         if not Param.hasattr("LayerNorm1"):
#             self.AddSubModule(
#                 "LayerNorm1", DLUtils.norm.LayerNorm()
#             )
#         if not Param.hasattr("LayerNomr2"):
#             self.AddSubModule(
#                 "LayerNorm2", DLUtils.norm.LayerNorm()
#             )
        
#         return super().Init(IsSuper=True, IsRoot=IsRoot)
    
# class SelfAttention(DLUtils.module.AbstractNetwork):
#     def __init__(self, SubModule=None):
#         super().__init__()
#         Param = self.Param
#         Param._CLASS = "DLUtils.network.SelfAttention"
#         if SubModule is None:
#             self.AddSubModule("SubModule", SubModule)
#     def SetParam(self, **Dict):
#         Param = self.Param
#         InNum = Dict.get("InNum")
#         if InNum is None:
#             Param.In.Num = InNum
#         QKSize = Dict.get("QKSize")
#         if QKSize is not None:
#             Param.QK.Size = QKSize

#         VSize = Dict.get("VSize")
#         if VSize is not None:
#             Param.V.Size = VSize
#     def SetTrainParam(self, **Dict):
#         for Name, TrainParam in Dict.items():
#             if Name in ["Q"]:
#                 self.SetTrainParam(Q=TrainParam)
#             elif Name in ["K"]:
#                 self.SetTrainParam(K=TrainParam)
#             elif Name in ["V"]:
#                 self.SetTrainParam(V=TrainParam)
#             else:
#                 raise Exception()
#         return self
#     def Receive(self, X:torch.Tensor):
#         QKSize = self.QKSize
#         HeadNum = self.HeadNum
#         QKHeadSize = self.QKHeadSize
#         VHeadSize = self.VHeadSize
#         # X: [BatchSize, TokenNum, FeatureNum]
#         BatchSize = X.size(0)
#         TokenNum = X.size(1)
#         FeatureNum = X.size(2)
#         X = X.reshape(BatchSize* TokenNum, FeatureNum)

#         V = torch.bmm(X, self.V) # [BatchSize * TokenNum, TokenNum * VSize]
#         K = torch.bmm(X, self.K) # [BatchSize * TokenNum, QKSize]
#         Q = torch.bmm(X, self.Q) # [BatchSize * TokenNum, QKSize]

#         VAttention = MultiHeadAttention(Q, K, V, BatchSize, TokenNum, HeadNum, VHeadSize, QKHeadSize)

#         return VAttention

#     def Init(self, IsSuper=False, IsRoot=True):
#         Param = self.Param
#         self.InNum = Param.In.Num
#         if not Param.QK.Head.hasattr("Num"):
#             Param.Head.Num = 1
#             Param.QK.Head.Size = Param.QK.Size // Param.QK.Head.Num
#         self.HeadNum = Param.Head.Num
#         self.QKHeadSize = Param.QK.Head.Size
#         self.QKSize = Param.QK.Size
#         self.OutNum = Param.V.Size
#         self.VSize = Param.V.Size
#         assert self.QKSize % self.HeadNum == 0
#         assert self.VSize % self.HeadNum == 0
#         return super().Init(IsSuper=True, IsRoot=IsRoot)

# def Attention(Q, K, V, QKSize=None):
#     # Q: [BatchSize, TokenNum,  QKSize]
#     # K: [BatchSize, TokenNum,  QKSize]
#     # V: [BatchSize, TokenNum,  VSize ]
#     QKSize = Q.size(2)
#     QK = torch.bmm(
#         Q,                  # [BatchSize, TokenNum,  QKSize]
#         K.permute(0, 2, 1)  # [BatchSize, QKSize,    TokenNum]
#     ) # [BatchSize, TokenNum, TokenNum]
#     QK = QK / math.sqrt(QKSize)
#     QKSoftmax = torch.softmax(QK, dim=2)  # [BatchSize, TokenNum, TokenNum]
#     VAttention  = torch.bmm(QKSoftmax, V) # [BatchSize, TokenNum, VSize]
#     return VAttention

# SingleHeadAttention = Attention

# def MultiHeadAttention(Q, K, V, BatchSize, TokenNum, HeadNum, VHeadSize, QKHeadSize):
#     # V: [BatchSize, HeadNum, TokenNum, VHeadSize]
#     V = V.reshape(BatchSize, TokenNum, HeadNum, VHeadSize).permute(0, 2, 1, 3)
#     # K: [BatchSize, HeadNum, TokenNum, QKHeadSize]
#     K = K.reshape(BatchSize, TokenNum, HeadNum, QKHeadSize).permute(0, 2, 1, 3)
#     # Q: [BatchSize, HeadNum, TokenNum, QKHeadSize]
#     Q = Q.reshape(BatchSize, TokenNum, HeadNum, QKHeadSize).permute(0, 2, 1, 3)

#     QK = torch.bmm(
#         Q.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize), # [BatchSize * HeadNum, TokenNum, QKHeadSize]
#         K.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize).permute(0, 2, 1) # [BatchSize * HeadNum, QKHeadSize, TokenNum]
#     ).reshape(BatchSize, HeadNum, TokenNum, TokenNum) # [BatchSize * HeadNum, TokenNum, TokenNum]
    
#     QK = QK / math.sqrt(QKHeadSize)
#     QKSoftmax = torch.softmax(QK, dim=2) # [BatchSize * HeadNum, TokenNum, TokenNum]
    
#     VAttention  = torch.bmm(
#         QKSoftmax, # [BatchSize * HeadNum, TokenNum, TokenNum]
#         V.reshape(BatchSize * HeadNum, TokenNum, VHeadSize)
#     ).reshape(BatchSize, HeadNum, TokenNum, VHeadSize) # [BatchSize, HeadNum, TokenNum, VHeadSize]
#     VAttention = VAttention.permute(0, 2, 1, 3) # [BatchSize, TokenNum, HeadNum, VHeadSize]
#     #VAttention = VAttention.reshape(BatchSize, TokenNum, HeadNum, VHeadSize) 
#     VAttention = VAttention.reshape(BatchSize, TokenNum, HeadNum * VHeadSize)
# AttentionMultiHead = MultiHeadAttention

# from ..AbstractModule import AbstractNetwork
# class Transformer(AbstractNetwork):
#     def __init__(param):
#         return
#     def forward(X):
#         # X: [BatchSize, TokenNum, EmbeddingNum]
#         pass
#     def LayerNorm(X):
#         return
#     def Init():
#         return

# class SelfAttention1D(DLUtils.module.AbstractModule):
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         return
#     def Build(self, IsLoad=False):
#         self.BeforeBuild(IsLoad)
#         param = self.param
#         data  = self.data
#         cache = self.cache

#         assert HasAttrs(param, "In.Num")
#         assert HasAttrs(param, "Out.Num")
#         EnsureAttrs(param, "Attention.Feature.Num", default=Param.In.Num)
#         SetAttrs(param, "Weight.Input2Query.Size", value=[Param.In.Num, param.Attention.Feature.Num])
#         SetAttrs(param, "Weight.Input2Key.Size",   value=[Param.In.Num, param.Attention.Feature.Num])
#         if HasAttrs(param, "Out.Num"):
#             SetAttrs(param, "Weight.Input2Value.Size", value=[Param.In.Num, Param.Out.Num])
#         elif HasAttrs(param, "Weight.Input2Value.Size"):
#             SetAttrs(param, "Out.Num", value=param.Weight.Input2Value.Size[1])
#         else:
#             raise Exception()

#         if cache.IsInit:
#             data.Input2Query = DLUtils.transform.CreateWeight2D(param.Weight.Input2Query)
#             data.Input2Key   = DLUtils.transform.CreateWeight2D(param.Weight.Input2Key)
#             data.Input2Value = DLUtils.transform.CreateWeight2D(param.Weight.Input2Value)
#         else:
#             data.Input2Query = DLUtils.ToTorchTensor(data.Input2Query)
#             data.Input2Key   = DLUtils.ToTorchTensor(data.Input2Key)
#             data.Input2Value = DLUtils.ToTorchTensor(data.Input2Value)

#         cache.Tensor = DLUtils.EmptyPyObj()
        
#         cache.TensorDict["Input2Query"] = cache.Input

#         cache.TrainParam.append([data, "Input2Query", data.Input2Query])
#         cache.TrainParam.append([data, "Input2Key",   data.Input2Key])
#         cache.TrainParam.append([data, "Input2Value", data.Input2Value])
#         cache.AttentionCoefficient = 1.0 / param.Attention.Feature.Num ** 0.5

#         return self
#     def SplitHead(self, Data, HeadNum):
#         BatchSize = Data.shape[0]
#         TokenNum = Data.shape[1]
#         # FeatureNumTotal = Data.shape[2]
#         Data = Data.view(BatchSize, TokenNum, HeadNum, -1)
#         Data = Data.permute(0, 2, 1, 3)
#         return Data
#     def forward(self, Input, log):
#         # Input: [BatchSize, TokenNum, InNum]
#         data = self.data
#         cache = self.cache
#         Query = torch.mm(Input, data.Input2Query) # [BatchSize, TokenNum, AttentionFeatureNum]
#         Key = torch.mm(Input, data.Input2Key)     # [BatchSize, TokenNum, AttentionFeatureNum]
#         Value = torch.mm(Input, data.Input2Value) # [BatchSize, TokenNum, OutNum]

#         self.LogCache("Attention.Key", Key, "Attention", log=log)
#         self.LogCache("Attention.Query", Query, "Attention", log=log)
#         self.LogCache("Attention.Value", Value, "Attention", log=log)

#         if cache.MultiHead:
#             QueryMultiHead = torch.mm(Query, data.Query2MultiHeadQuery) # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
#             KeyMultiHead   = torch.mm(Key,   data.Query2MultiHeadKey)   # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
#             ValueMultiHead = torch.mm(Value, data.Query2MultiHeadValue) # [BatchSize, TokenNum, OutputFeatureNum]

#             QueryMultiHead = self.SplitHead(QueryMultiHead, cache.HeadNum) # [BatchSize, HeadNum, TokenNum, AttentionPerHeadFeatureNum]
#             KeyMultiHead   = self.SplitHead(KeyMultiHead,   cache.HeadNum)
#             ValueMultiHead = self.SplitHead(ValueMultiHead, cache.HeadNum)

#             AttentionMultiHead = torch.einsum("bhtf,bhtf->bht", QueryMultiHead, KeyMultiHead) # [BatchSize, HeadNum, TokenNum, TokenNum]
#             AttentionMultiHead *= cache.AttentionPerHeadFeatureNum
#             AttentionMultiHead = F.softmax(AttentionMultiHead, axis=3)

#             OutputMultiHead = torch.einsum("bhtu,bhuo->bhto", AttentionMultiHead, ValueMultiHead)
            
#             self.LogCache("AttentionMultiHead",       AttentionMultiHead, "Attention", log=log)
#             self.LogCache("AttentionMultiHead.Query", QueryMultiHead,     "Attention", log=log)
#             self.LogCache("AttentionMultiHead.Key",   KeyMultiHead,       "Attention", log=log)
#             self.LogCache("AttentionMultiHead.Value", ValueMultiHead,     "Attention", log=log)
#             self.LogCache("OutputMultiHead",          OutputMultiHead,    "Activity",  log=log)
#             return OutputMultiHead
#         else:
#             Attention = torch.bmm(Query, Key.permute(0, 2, 1)) ** cache.AttentionFeatureNum # [BatchSize, TokenNum, TokenNum]
#             Attention = F.softmax(Attention, dim=2) # [BatchSize, TokenNum, TokenNum]
#             Output = torch.bmm(Attention, Value) # [BatchSize, TokenNum, OutNum]

#             self.LogCache("Attention", Attention, "Attention", log=log)
#             self.LogCache("Output", Output, "Activity", log=log)
#             return Output

# __MainClass__ = SelfAttention1D
# DLUtils.transform.SetMethodForTransformModule(__MainClass__)