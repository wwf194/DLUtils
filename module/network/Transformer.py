import torch
import math

import DLUtils
class SelfAttention(DLUtils.module.AbstractNetwork):
    def __init__(self, SubModule=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.NN.ResLayer"
        if SubModule is None:
            self.AddSubModule("SubModule", SubModule)
    def SetParam(self, **Dict):
        Param = self.Param
        InputNum = Dict.get("InputNum")
        if InputNum is None:
            Param.Input.Num = InputNum
        QKSize = Dict.get("QKSize")
        if QKSize is not None:
            Param.QK.Size = QKSize

        VSize = Dict.get("VSize")
        if VSize is not None:
            Param.V.Size = VSize

    def SetWeight(self, **Dict):
        Param = self.Param
        Q = Dict.get("Q")
        if Q is not None:
            self.AddTrainParam("Q", Q)
        
        K = Dict.get("K")
        if K is not None:
            self.AddTrainParam("K", K)
        
        V = Dict.get("V")
        if V is not None:
            self.AddTrainParam("V", V)

        return self
    def Receive(self, X:torch.Tensor):
        QKSize = self.QKSize
        HeadNum = self.HeadNum
        QKHeadSize = self.QKHeadSize
        VHeadSize = self.VHeadSize
        # X: [BatchSize, TokenNum, FeatureNum]
        BatchSize = X.size(0)
        TokenNum = X.size(1)
        FeatureNum = X.size(2)
        X = X.reshape(BatchSize* TokenNum, FeatureNum)

        V = torch.bmm(X, self.V) # [BatchSize * TokenNum, TokenNum * VSize]
        K = torch.bmm(X, self.K) # [BatchSize * TokenNum, QKSize]
        Q = torch.bmm(X, self.Q) # [BatchSize * TokenNum, QKSize]

        VAttention = MultiHeadAttention(Q, K, V, BatchSize, TokenNum, HeadNum, VHeadSize, QKHeadSize)

        return VAttention

    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.InputNum = Param.Input.Num
        if not Param.QK.Head.hasattr("Num"):
            Param.Head.Num = 1
            Param.QK.Head.Size = Param.QK.Size // Param.QK.Head.Num
        self.HeadNum = Param.Head.Num
        self.QKHeadSize = Param.QK.Head.Size
        self.QKSize = Param.QK.Size
        self.OutputNum = Param.V.Size
        self.VSize = Param.V.Size
        assert self.QKSize % self.HeadNum == 0
        assert self.VSize % self.HeadNum == 0
        return super().Init(IsSuper, IsRoot)

def Attention(Q, K, V):
    # Q: [BatchSize, TokenNum,  QKSize]
    # K: [BatchSize, TokenNum,  QKSize]
    # V: [BatchSize, TokenNum,  VSize ]
    QSize = Q.size(2)
    QK = torch.bmm(
        Q,
        K.permute(0, 2, 1)
    ) # [BatchSize, TokenNum, TokenNum]
    QK = QK / math.sqrt(QSize)
    QKSoftmax = torch.softmax(QK, dim=-1) # [BatchSize, TokenNum, TokenNum]
    VWeighedByAttention  = torch.bmm(QKSoftmax, V) # [BatchSize, TokenNum, VSize]
    return VWeighedByAttention

SingleHeadAttention = Attention

def MultiHeadAttention(Q, K, V, BatchSize, TokenNum, HeadNum, VHeadSize, QKHeadSize):
    # V: [BatchSize, HeadNum, TokenNum, VHeadSize]
    V = V.reshape(BatchSize, TokenNum, HeadNum, VHeadSize).permute(0, 2, 1, 3)
    # K: [BatchSize, HeadNum, TokenNum, QKHeadSize]
    K = K.reshape(BatchSize, TokenNum, HeadNum, QKHeadSize).permute(0, 2, 1, 3)
    # Q: [BatchSize, HeadNum, TokenNum, QKHeadSize]
    Q = Q.reshape(BatchSize, TokenNum, HeadNum, QKHeadSize).permute(0, 2, 1, 3)

    QK = torch.bmm(
        Q.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize), # [BatchSize * HeadNum, TokenNum, QKHeadSize]
        K.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize).permute(0, 2, 1) # [BatchSize * HeadNum, QKHeadSize, TokenNum]
    ).reshape(BatchSize, HeadNum, TokenNum, TokenNum) # [BatchSize * HeadNum, TokenNum, TokenNum]
    
    QK = QK / math.sqrt(QKHeadSize)
    QKSoftmax = torch.softmax(QK, dim=2) # [BatchSize * HeadNum, TokenNum, TokenNum]
    
    VAttention  = torch.bmm(
        QKSoftmax, # [BatchSize * HeadNum, TokenNum, TokenNum]
        V.reshape(BatchSize * HeadNum, TokenNum, VHeadSize)
    ).reshape(BatchSize, HeadNum, TokenNum, VHeadSize) # [BatchSize, HeadNum, TokenNum, VHeadSize]
    VAttention = VAttention.permute(0, 2, 1, 3) # [BatchSize, TokenNum, HeadNum, VHeadSize]
    #VAttention = VAttention.reshape(BatchSize, TokenNum, HeadNum, VHeadSize) 
    VAttention = VAttention.reshape(BatchSize, TokenNum, HeadNum * VHeadSize)

def LayerNorm(X:torch.Tensor):
    # X: [BatchSize, ]
    pass

from ..AbstractModule import AbstractNetwork
class Transformer(AbstractNetwork):
    def __init__(param):
        return
    def forward(X):
        # X: [BatchSize, TokenNum, EmbeddingNum]
        pass
    def LayerNorm(X):
        return
    def Init():
        return



