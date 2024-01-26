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
import math

import DLUtils.network as network
class TransformerEncoder(DLUtils.module.AbstractNetwork):
    # a stack of multiple multi-head self-attention layer
    ParamMap = DLUtils.IterableKeyToElement({
        ("LayerNum"): "Layer.Num",
        ("TokenSize", "FeatureSize"): ("Token.Size"),
        ("QKSize"): "MSA.Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "MSA.Attention.V.Size", # total size. not size of each head.
        ("HeadNum"): "MSA.Attention.Head.Num",
        ("MLPSize"): "MLP.HiddenLayer.Size",
        ("MLPNonLinear"): "MLP.NonLinear",
        ("DropOut"): "DropOut.Probability",
        ("DropOutInplace", "DropOutInPlace"): "DropOut.InPlace"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def Receive(self, In):
        # In: (BatchSize, TokenNum, TokenSize)
        OutList = []
        for LayerIndex, LayerModule in enumerate(self.LayerList):
            Out = LayerModule(In)
            OutList.append(Out)
            In = Out
        return Out
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Layer.hasattr("Num")
        self.LayerNum = Param.Layer.Num
        self.TokenSize = Param.Token.Size
        if self.IsInit():
            for LayerIndex in range(self.LayerNum):
                SubModuleName = "L%d"%LayerIndex
                if not self.HasSubModule("L%d"%LayerIndex):
                    self.AddSubModule(
                        SubModuleName,
                        MultiheadSelfAttentionLayer(
                            InTokenSize = self.TokenSize,
                            OutTokenSize = self.TokenSize,
                            QKSize = Param.MSA.Attention.QK.Size,
                            VSize = Param.MSA.Attention.V.Size,
                            HeadNum = Param.MSA.Attention.Head.Num,
                            MLPSize=Param.MLP.HiddenLayer.Size,
                            DropOut = Param.DropOut.setdefault("Probability", 0.0)
                        )
                    )
        self.LayerList = []
        for LayerIndex in range(self.LayerNum):
            self.LayerList.append(self.GetSubModule("L%d"%LayerIndex))
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class MultiheadSelfAttentionLayer(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("InSize", "InNum", "InTokenSize"): "In.Token.Size",
        ("OutSize", "OutNum", "OutTokenSize"): "Out.Token.Size",
        ("QKSize"): "MSA.Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "MSA.Attention.V.Size", # total size. not size of each head.
        ("HeadNum"): "MSA.Attention.Head.Num",
        ("MLPSize"): "MLP.HiddenLayer.Size",
        ("NonLinear"): "MLP.NonLinear",
        ("DropOut"): "DropOut.Probability",
        ("DropOutInplace", "DropOutInPlace"): "DropOut.InPlace"
    })
    # multi-head self attention, with layer norm and 2-layer mlp
    def ReceiveNormTransformResidual(self, X):
        # LayerNorm --> MSA / MLP --> DropOut --> Residual
        # X: (BatchSize, TokenNumQ, TokenFeatureNum)
        Y = self.LayerNorm1(X) # layer_norm
        Y = self.MSA(Y) # multi-head attention
        Y = self.DropOut(Y) # dropout
        Y = X + Y # residual
        
        Z = self.LayerNorm2(Y) # layer_norm
        Z = self.MLP(Z) # multi-layer perceptron
        Z = self.DropOut(Z) # dropout
        Z = Y + Z # residual
        return Z
    def ReceiveTransformResidualNorm(self, X):
        # MSA / MLP --> DropOut --> Residual --> LayerNorm
        # X: (BatchSize, TokenNumQ, TokenFeatureNum)
        Y = self.MSA(X)
        Y = X + Y # residual
        Y = self.LayerNorm1(Y)
        
        Z = self.MLP(Y)
        Z = Y + Z # residual
        Z = self.LayerNorm2(Z)
        return Z
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.InSize = Param.In
        self.OutSize = Param.Out.Size
        self.QKSize = Param.MSA.Attention.QK.size
        self.VSize = Param.MSA.Attention.V.size
        self.HeadNum = Param.MSA.Attention.Head.Num
        
        if self.IsInit():
            # multi-head self attention
            if not self.HasSubModule("MSA"): # multi-head self attention
                Param.MSA.Out.setdefault("Token.Size", Param.Out.Token.Size)
                self.AddSubModule(
                    "MSA", MultiHeadSelfAttention(
                        InSize=Param.In.Token.Size,
                        OutSize=Param.MSA.Out.Token.Size,
                        QKSize = Param.MSA.Attention.QK.Size,
                        VSize = Param.MSA.Attention.V.Size,
                        HeadNum = Param.MSA.Attention.Head.Num
                    )
                )

            # 2-layer mlp
            if not self.HasSubModule("MLP"):
                Param.MLP.setdefault("NonLinear", "ReLU")
                MLPUnitNum = [
                    Param.MSA.Out.Token.Size,
                    Param.MLP.HiddenLayer.Size,
                    Param.Out.Token.Size
                ]
                self.AddSubModule(
                    "MLP", network.MLP(
                        UnitNum = MLPUnitNum,
                        BiasOnLastLayer=Param.setdefault("BiasOnLastLayer", True),
                        NonLinearOnLastLayer=Param.setdefault("NonLinearOnLastLayer", False),
                        NonLinear=Param.MLP.setdefault("NonLinear", "ReLU")
                    )
                )

            # layer norm after multi-head attention
            if not self.HasSubModule("LayerNorm1"):
                self.AddSubModule(
                    "LayerNorm1", network.LayerNorm(NormShape=(Param.MSA.Out.Token.Size), Affine=True)
                )
                
            # layer norm after 2-layer mlp
            if not self.HasSubModule("LayerNorm2"):
                self.AddSubModule(
                    "LayerNorm2", network.LayerNorm(NormShape=(Param.Out.Token.Size), Affine=True)
                )

            # operation order setting.
            self.OperationOrder = Param.setdefault("OperationOrder", "NormTransformResidual")
        
            # dropout setting
            self.SetDropOutInit()

        self.SetDropOut()
        
        # operation order setting.
        self.OperationOrder = Param.OperationOrder
        if self.OperationOrder in ["NormTransformResidual"]:
            self.Receive = self.ReceiveNormTransformResidual
        elif self.OperationOrder in ["TransformResidualNorm"]:
            self.Receive = self.ReceiveTransformResidualNorm
        else:
            raise Exception()
        
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class MultiHeadAttention(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("HeadNum"): "Attention.Head.Num",
        ("InSize", "InNum", "OutTokenSize"): "In.Token.Size",
        ("OutSize", "OutNum", "OutTokenSize"): "Out.Token.Size",
        ("QKSize"): "Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "Attention.V.Size" # total size. not size of each head.
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
    def _Receive(self, Q, K, V):
        BatchSize = Q.size(0)
        TokenNumQ = Q.size(1)
        TokenNumKV = K.size(1)
        QKSize = Q.size(2)
        # (BatchSize, TokenNumQ,  QKSize)
        # (BatchSize, TokenNumKV, QKSize)
        # (BatchSize, TokenNumKV, VSize)
        Q1 = torch.matmul(Q, self.WeightQ)
            # (BatchSize, TokenNumQ, QKSizeTotal)
        K1 = torch.matmul(K, self.WeightK)
            # (BatchSize, TokenNumKV, QKSizeTotal)
        V1 = torch.matmul(V, self.WeightV)
            # (BatchSize, TokenNumKV, VSizeTotal)

        Q1 = Q1.view(BatchSize, TokenNumQ, self.HeadNum, self.QKSizeHead)
        K1 = K1.view(BatchSize, TokenNumKV, self.HeadNum, self.QKSizeHead)
        V1 = V1.view(BatchSize, TokenNumKV, self.HeadNum, self.VSizeHead)

        Q1 = Q1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumQ, QKSize)
        K1 = K1.permute(0, 2, 3, 1) # (BatchSize, HeadNum, QKSize, TokenNumKV)
        V1 = V1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumKV, VSize)

        AttentionCoeff = torch.matmul(Q1, K1) / self.QKDotProductCoeff
        # (BatchSize, HeadNum, TokenNumQ, TokenNumKV)
        AttentionCoeff = F.softmax(AttentionCoeff, 3)
        
        VAttention = torch.matmul(AttentionCoeff, V1)
        # (BatchSize, HeadNum, TokenNumQ, VSize)

        V2 = VAttention.permute(0, 2, 1, 3)
        V2 = V2.reshape(BatchSize, TokenNumQ, self.VSize)
        Out = torch.matmul(V2, self.Value2Out) # (BatchSize, TokenNumQ, OutSize)
        return Out

    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("In.Token.Size", "Out.Token.Size")
        assert Param.Attention.hasattr("QK.Size", "V.Size", "Head.Num")
         
        # QKSize and VSize is total size, not size of each head.
        self.QKSize = Param.Attention.QK.Size
        self.VSize = Param.Attention.V.Size
        self.HeadNum = Param.Attention.Head.Num
        self.InSize = Param.In.Token.Size
        self.OutSize = Param.Out.Token.Size
        assert self.QKSize % self.HeadNum == 0
        assert self.VSize % self.HeadNum == 0

        self.QKSizeHead = self.QKSize // self.HeadNum
        self.VSizeHead = self.VSize // self.HeadNum
        self.QKDotProductCoeff = 1.0 / self.QKSizeHead ** 0.5
    
        if self.IsInit():
            if not Param.Attention.Q.hasattr("Data"):
                self.SetTrainParam(
                    Name="WeightQ",
                    Path="Attention.Q.Data",
                    Data=DLUtils.DefaultLinearLayerWeight(            
                            Shape=(self.InSize, self.QKSize)
                        )
                )
            if not Param.Attention.K.hasattr("Data"):
                self.SetTrainParam(
                    Name="WeightK",
                    Path="Attention.K.Data",
                    Data=DLUtils.DefaultLinearLayerWeight(            
                            Shape=(self.InSize, self.QKSize)
                        )
                )
            if not Param.Attention.V.hasattr("Data"):
                self.SetTrainParam(
                    Name="WeightV",
                    Path="Attention.V.Data",
                    Data=DLUtils.DefaultLinearLayerWeight(            
                            Shape=(self.InSize, self.VSize)
                        )
                )
            if not Param.Attention.O.hasattr("Data"):
                self.SetTrainParam(
                    Name="Value2Out",
                    Path="Attention.O.Data",
                    Data=DLUtils.DefaultLinearLayerWeight(            
                            Shape=(self.VSize, self.OutSize)
                        )
                )
        else:
            assert Param.Attention.hasattr("Q.Data", "K.Data", "V.Data", "O.Data")
    
        # this class might be inherited. child class might have their own Receive method.
        if not IsSuper:
            self.Receive = self._Receive
    
        return super().Init(IsSuper=True, IsRoot=IsRoot)

MA = MHA = MultiHeadAttention
AttentionMultiHead = MultiHeadAttention

class MultiHeadSelfAttention(MultiHeadAttention):
    def Receive(self, In):
        return self._Receive(In, In, In)
    def Build(self, IsSuper=False, IsRoot=True):
        return super().Init(IsSuper=True, IsRoot=IsRoot)
MSA = MHSA = MultiHeadSelfAttention

def attention(Q, K, V, WQ, WK, WV, WO):
    # Q: (BatchSize, TokenNumQ,   InSize)
    # K: (BatchSize, TokenNumKV,  InSize)
    # V: (BatchSize, TokenNumKV,  InSize)
    # WQ: (InSize, QKSize)
    # WK: (InSize, QKSize)
    # WV: (InSize, VSize)
    # WO: (VSize, OutSize)
    Q1 = torch.matmul(Q, WQ) # (BatchSize, TokenNumQ, QKSize)
    K1 = torch.matmul(K, WK) # (BatchSize, TokenNumKV, QKSize)
    V1 = torch.matmul(V, WV) # (BatchSize, TokenNumKV, VSize)
    
    QKSize = Q.size(2)
    AttentionCoeff = torch.matmul(
        Q1,                  # (BatchSize, TokenNumQ, QKSize)
        K1.permute(0, 2, 1)  # (BatchSize, QKSize, TokenNumKV)
    ) # (BatchSize, TokenNumQ, TokenNumKV)

    AttentionCoeff = AttentionCoeff / math.sqrt(QKSize)
    AttentionCoeff = F.softmax(AttentionCoeff,21) # (BatchSize, TokenNumQ, TokenNumKV)
    
    V2  = torch.matmul(AttentionCoeff, V1) # (BatchSize, TokenNumQ, VSize)
    V3 = torch.matmul(V2, WO)
    return V3

def attention_multi_head(Q, K, V, WQ, WK, WV, WO, HeadNum):
    # Q: (BatchSize, TokenNumQ,   QKSize)
    # K: (BatchSize, TokenNumKV,  QKSize)
    # V: (BatchSize, TokenNumKV,  VSize )

    BatchSize = Q.size(0)
    TokenNumQ = Q.size(1)
    QKSize = Q.size(2)
    TokenNumKV = K.size(1)
    VSize = WV.size(0)
    QKSizeHead = QKSize // HeadNum
    VSizeHead = VSize // HeadNum

    Q1 = torch.matmul(Q, WQ)
    K1 = torch.matmul(K, WK)
    V1 = torch.matmul(V, WV)
    
    Q1 = Q1.view(BatchSize, TokenNumQ, HeadNum, QKSizeHead)
    K1 = K1.view(BatchSize, TokenNumKV, HeadNum, QKSizeHead)
    V1 = V1.view(BatchSize, TokenNumKV, HeadNum, VSizeHead)
    
    Q1 = Q1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumQ, QKSizeHead)
    K1 = K1.permute(0, 2, 3, 1) # (BatchSize, HeadNum, QKSizeHead, TokenNumKV)
    V1 = V1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumKV, VSizeHead)

    AttentionCoeff = torch.matmul(Q1, K1) / QKSizeHead ** 0.5
    # (BatchSize, HeadNum, TokenNumQ, TokenNumKV)
    AttentionCoeff = F.softmax(AttentionCoeff)
    VAttention = torch.matmul(AttentionCoeff, V1)
    # (BatchSize, HeadNum, TokenNumQ, VSizeHead)
    
    V2 = VAttention.permute(0, 2, 1, 3) # (BatchSize, TokenNumQ, HeadNum, VSizeHead)
    V2 = V2.reshape(BatchSize, TokenNumQ, VSize) # (BatchSize, TokenNumQ, VSize)
    
    V3 = torch.matmul(V2, WO) # (BatchSize, TokenNumQ, OutSize)
    # output token num is same as token num of Q.
    return V2
attention_multihead = multihead_attetnion = multi_head_attention = attention_multi_head