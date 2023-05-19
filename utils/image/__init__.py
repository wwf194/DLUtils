import DLUtils.utils.image.compress as compress
import DLUtils

import numpy as np

def GenerateBlackImage(Height=512, Width=512, ChannelNum=3):
    assert 1<= ChannelNum <= 4
    return np.ones(shape=(Height, Width, ChannelNum), dtype=np.int16)

def GenerateBlackImageInt16(Height=512, Width=512, ChannelNum=3):
    assert 1<= ChannelNum <= 4
    return np.ones(shape=(Height, Width, ChannelNum), dtype=np.int16)
    return 

def GenerateSingleColorImage(Height, Width, Color, ChannelNum, DataType):
    
    
    DataType = DLUtils.ParseDataTypeNp(DataType)