



import torch
import math
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

AttentionSingleHead = Attention

def AttentionWithHead(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
    # Q: [BatchSize, HeadNum, TokenNum,  QKHeadSize]
    # K: [BatchSize, HeadNum, TokenNum,  QKHeadSize]
    # V: [BatchSize, HeadNum, TokenNum,  VHeadSize ]
    BatchSize = Q.size(0)
    TokenNum = Q.size(1)
    HeadNum = Q.size(2)
    QKHeadSize = Q.size(3)
    VHeadSize = V.size(3)
    
    QK = torch.bmm(
        Q.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize),
        K.reshape(BatchSize * HeadNum, TokenNum, QKHeadSize).permute(0, 2, 1)
    ).reshape(BatchSize, HeadNum, TokenNum, TokenNum) # [BatchSize * HeadNum, TokenNum, TokenNum]
    QK = QK / math.sqrt(QKHeadSize)
    QKSoftmax = torch.softmax(QK, dim=-1) # [BatchSize * HeadNum, TokenNum, TokenNum]
    VAttention  = torch.bmm(
        QKSoftmax, 
        V.reshape(BatchSize * HeadNum, TokenNum, VHeadSize)
    ).reshape(BatchSize, HeadNum, TokenNum, VHeadSize) # [BatchSize, HeadNum, TokenNum, VHeadSize]
    VAttention = VAttention \
        .permute(0, 2, 1, 3) \
        .reshape(BatchSize, TokenNum, HeadNum, VHeadSize) 
    return VAttention.reshape(BatchSize, TokenNum, HeadNum * VHeadSize)

def MultiHeadAttention(Q, K, V, WQ: torch.tensor, WK, WV, WO, HeadNum):
    # Q: [BatchSize, TokenNum,  QKSize   ]
    # K: [BatchSize, TokenNum,  QKSize   ]
    # V: [BatchSize, TokenNum,  ValueSize]
    # WQ: [HeadNum, QKSize, QKHeadSize]
    # WK: [HeadNum, QKSize, QKHeadSize]
    # WV: [HeadNum, VSize,  VHeadSize ]

    # WV: [BatchSize, TokenNum, HeadNum, QKSizeHead]
    
    Q = Q[:, None, :, :] # [BatchSize, 1, TokenNum, QKSize]
    K = K[:, None, :, :] # [BatchSize, 1, TokenNum, QKSize]
    V = V[:, None, :, :] # [BatchSize, 1, TokenNum, VSize ]

    WQ = WQ[None, :, :, :] # [1, HeadNum, QKSize, QKHeadSize]
    WK = WK[None, :, :, :] # [1, HeadNum, QKSize, QKHeadSize]
    WV = WK[None, :, :, :] # [1, HeadNum, VSize,  VHeadSize ]

    QHeads = torch.matmul(Q, WQ) # [BatchSize, HeadNum, TokenNum, QKHeadSize]
    KHeads = torch.matmul(K, WK) # [BatchSize, HeadNum, TokenNum, QKHeadSize]
    VHeads = torch.matmul(V, WV) # [BatchSize, HeadNum, TokenNum, VHeadSize ]
    
    VHeadsAttention = AttentionWithHead(QHeads, KHeads, VHeads)

def LayerNorm(X:torch.Tensor):
    # X: [BatchSize, ]

class Transformer():
    def __init__(param):
        return
    def forward(X):
        # X: [BatchSize, TokenNum, EmbeddingNum]
    def LayerNorm(X):
        




