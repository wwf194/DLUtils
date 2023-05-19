import DLUtils.utils.image.compress as compress
import DLUtils
import cv2
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

import DLUtils.utils.image.compress as compress
from .compress import File2Image, Image2File, ImageNp2File, ResizeImage


def TextOnImage(Image):
    cv2.PutText()
    
def File2JpgFile(FilePath):
    Image = File2Image(FilePath)
    Name, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePath)
    Result = Image2File(Image, Name + ".jpg", Format=".jpg")
    return Result


    
