import numpy as np
import torch
import math
import DLUtils

import DLUtils
class Transformer(DLUtils.module.AbstractNetwork):
    def __init__(self, SubModule=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.Transformer"
        if SubModule is None:
            self.AddSubModule("SubModule", SubModule)
    def Receive(self, X):
        # X: (BatchSize, TokenNum, FeatureSize)
        
        Y = self.Attention(X)
        Y = X + Y # Residual Adding
        Y = self.LayerNorm1(Y)

        Z = self.MLP(Y)
        Z = Y + Z
        Z = self.LayerNorm2(Y)
        return Y
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not Param.hasattr("LayerNorm1"):
            self.AddSubModule(
                "LayerNorm1", DLUtils.norm.LayerNorm()
            )
        if not Param.hasattr("LayerNomr2"):
            self.AddSubModule(
                "LayerNorm2", DLUtils.norm.LayerNorm()
            )
        return super().Init(IsSuper=True, IsRoot=IsRoot)


class TransformLayer(DLUtils.module.AbstractNetwork):
    # multi-head self attention --> 2-layer mlp
    def Receive(self, X):
        # X: (BatchSize, TokenNum, TokenFeatureNum)
        Y = self.Attention(X)
        Y = X + Y # Residual Adding
        Y = self.LayerNorm1(Y)
        Z = self.MLP(Y)
        Z = Y + Z
        Z = self.LayerNorm2(Y)
        return Y
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param


        if self.IsInit():
            Param.setdefault("NonLinear", "ReLU")
            Param.setdefault("BiasOnLastLayer", False)
            Param.setdefault("NonLinearOnLastLayer", False)
            if not Param.hasattr("LayerNorm"):
                self.AddSubModule(
                    "LayerNorm", DLUtils.norm.LayerNorm(eps=1e-6)
                )
            if not Param.hasattr("MLP"):
                self.AddSubModule(
                    "MLP", DLUtils.network.MLP(
                        UnitNum = (
                            Param.InputNum,
                            Param.HiddenLayerNum,
                            Param.Output.Num
                        ),
                        BiasOnLastLayer=False,
                        NonLinearOnLastLayer=False,
                        NonLinear=Param.NonLinear
                    )
                )
        return super().Init(IsSuper=True, IsRoot=IsRoot)


class MultiHeadAttention(DLUtils.module.AbstractNetwork):
    
    def __init__(self, SubModule=None):
        super().__init__()
        if SubModule is not None:
            self.AddSubModule("SubModule", SubModule)
    def _Receive(self, In, K, V):
        BatchSize = In.size(0)
        TokenNumQ = In.size(1)
        TokenNumKV = K.size(1)
        InNum = In.size(2)
        # (BatchSize, TokenNumQ, InNum(TokenFeatureNum))
        # (BatchSize, TokenNumKV, InNum(TokenFeatureNum))
        # (BatchSize, TokenNumKV, InNum(TokenFeatureNum))
        Q1 = torch.matmul(In, self.Q)
            # (BatchSize, TokenNumQ, QKSizeTotal)
        K1 = torch.matmul(K, self.K)
            # (BatchSize, TokenNumKV, QKSizeTotal)
        V1 = torch.matmul(V, self.V)
            # (BatchSize, TokenNumKV, VSizeTotal)
    
        Q1 = Q1.view(BatchSize, self.TokenNumQ, self.HeadNum, self.QKSizeHead)
        K1 = K1.view(BatchSize, self.TokenNumKV, self.HeadNum, self.QKSizeHead)
        V1 = V1.view(BatchSize, self.TokenNumKV, self.HeadNum, self.QKSizeHead)

        Q1 = Q1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumQ, QKSize)
        K1 = K1.permute(0, 2, 3, 1) # (BatchSize, HeadNum, QKSize, TokenNumKV)
        V1 = V1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumKV, VSize)

        AttentionCoeff = torch.matmul(Q1, K1) / self.QKDotProductCoeff
        # (BatchSize, HeadNum, TokenNumQ, TokenNumKV)
        VAttention = torch.matmul(V1, AttentionCoeff)
        # (BatchSize, HeadNum, TokenNumQ, VSize)
        
        V2 = VAttention.permute(0, 2, 1, 3)
        V2 = V2.view(BatchSize, TokenNumQ, self.VSize)
        Out = torch.matmul(V2, self.Value2Out) # (BatchSize, TokenNumQ, OutNum)
        return Out

    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("SubModule")
        assert Param.Head.hasattr("Num", "Size")
        
        assert Param.hasattr("In.Num", "Out.Num")
        assert Param.Attention.hasattr("QK.Size", "V.Size", "Head.Num")
         
        # QKSize and VSize means total size, rather than size of each head.
        self.QKSize = Param.Attention.QK.Size
        self.VSize = Param.Attention.V.Size
        self.HeadNum = Param.Attention.Head.Num
        self.OutNum = Param.Out.Num
        self.QKDotProductCoeff = 1.0 / self.QKSize ** 0.5
        

        assert self.QKSize % self.HeadNum == 0
        assert self.VSize % self.HeadNum == 0

        self.QKSizeHead = self.QKSize // self.HeadNum
        self.VSizeHead = self.VSize // self.HeadNum

        if self.IsInit():
            if not Param.Data.hasattr("In2Query"):
                Param.Attention.Q.Value = DLUtils.DefaultLinearLayerWeight(            
                    Shape=(self.InNum, self.QKSize)
                )
                self.RegisterTrainParam("In2Query", "Attention.Q.Data")
            if not Param.Data.hasattr("In2Key"):
                Param.Attention.Q.Value = DLUtils.DefaultLinearLayerWeight(            
                    Shape=(self.InNum, self.QKSize)
                )
                self.RegisterTrainParam("In2Key", "Attention.K.Data")
            if not Param.Data.hasattr("In2Value"):
                Param.Attention.K.Value = DLUtils.DefaultLinearLayerWeight(            
                    Shape=(self.InNum, self.OutNum)
                )
                self.RegisterTrainParam("In2Value", "Attention.V.Data")
            if not Param.Data.hasattr("Value2Out"):
                Param.Attention.K.Value = DLUtils.DefaultLinearLayerWeight(            
                    Shape=(self.VSize, self.OutNum)
                )
                self.RegisterTrainParam("Value2Out", "Attention.O.Data")
        else:
            assert Param.Attention.hasattr("Q.Data", "K.Data", "V.Data", "O.Data")
        
        if not IsSuper:
            self.Receive = self._Receive
    
        return super().Init(IsSuper=True, IsRoot=IsRoot)

MA = MHA = MultiHeadAttention

class MultiHeadSelfAttention(MultiHeadAttention):
    def Receive(self, In):
        return self._Receive(In, In, In)
MSA = MHSA = MultiHeadSelfAttention

def Attention(Q, K, V, QKSize=None):
    # Q: [BatchSize, TokenNum,  QKSize]
    # K: [BatchSize, TokenNum,  QKSize]
    # V: [BatchSize, TokenNum,  VSize ]
    QKSize = Q.size(2)
    QK = torch.bmm(
        Q,                  # [BatchSize, TokenNum,  QKSize]
        K.permute(0, 2, 1)  # [BatchSize, QKSize,    TokenNum]
    ) # [BatchSize, TokenNum, TokenNum]
    QK = QK / math.sqrt(QKSize)
    QKSoftmax = torch.softmax(QK, dim=2)  # [BatchSize, TokenNum, TokenNum]
    VAttention  = torch.bmm(QKSoftmax, V) # [BatchSize, TokenNum, VSize]
    return VAttention

def _MultiHeadAttention(Q, K, V, BatchSize, TokenNum, HeadNum, VHeadSize, QKHeadSize):
    # V: (BatchSize, HeadNum, TokenNum, VHeadSize)
    V = V.reshape(BatchSize, TokenNum, HeadNum, VHeadSize).permute(0, 2, 1, 3)
    # K: (BatchSize, HeadNum, TokenNum, QKHeadSize)
    K = K.reshape(BatchSize, TokenNum, HeadNum, QKHeadSize).permute(0, 2, 1, 3)
    # Q: (BatchSize, HeadNum, TokenNum, QKHeadSize)
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
AttentionMultiHead = MultiHeadAttention