import DLUtils
import cv2
import numpy as np
import DLUtils.utils.image.compress as compress
from .compress import File2Image, Image2File, ImageNp2File, ResizeImage

def GenerateBlackImage(Height=512, Width=512, ChannelNum=3):
    assert 1<= ChannelNum <= 4
    return np.ones(shape=(Height, Width, ChannelNum), dtype=np.int16)

def GenerateBlackImageInt16(Height=512, Width=512, ChannelNum=3):
    assert 1<= ChannelNum <= 4
    return np.ones(shape=(Height, Width, ChannelNum), dtype=np.int16)
    return 

def GenerateSingleColorImage(Height, Width, Color, ChannelNum, DataType):
    DataType = DLUtils.ParseDataTypeNp(DataType)


def TextOnImageCenter(Image, Text="Text", Color=(0, 0, 0)):
    Height, Width = Image.shape[0], Image.shape[1]
    TextWidthMax = round(Width * 0.5)
    TextHeightMax = round(Height * 0.5)
    
    Scale = 1.0
    Size = cv2.getTextSize(
        text=Text,
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=Scale,
        thickness=3
    ) # ((Width, Height), ?)
    
    WidthCurrent = Size[0][0]
    HeightCurrent = Size[0][1]
    
    Scale = Scale * min(TextWidthMax / WidthCurrent, TextHeightMax / HeightCurrent)

    Size = cv2.getTextSize(
        text=Text,
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=Scale,
        thickness=3
    ) # ((Width, Height), ?)

    WidthCurrent = Size[0][0]
    HeightCurrent = Size[0][1]
    
    XLeft = round(Width / 2.0 - WidthCurrent / 2.0)
    YBottom = round(Height / 2.0 + HeightCurrent / 2.0)

    XYLeftBottom = (XLeft, YBottom)

    TextOnImageCv(Image, Text, XYLeftBottom, Color, Scale)

def TextOnImageCv(Image, Text="Text", XYLeftBottom=None, Color=(0, 0, 0), Scale=1.0):
    if XYLeftBottom is None:
        XYLeftBottom = (0, Image.shape[0])

    cv2.putText(
        img=Image,
        text=Text,
        org=XYLeftBottom, # coordinate of left bottom corner
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=Scale,
        color=Color,
        thickness=3
    )
    return Image
TextOnImage = TextOnImageCv

def ImageFile2JpgImageFile(FilePath, Verbose=True):
    Image = File2Image(FilePath)
    if Image is None:
        if Verbose:
            DLUtils.print("error reading image:", FilePath)
        return False
    Name, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePath)
    Result = Image2File(Image, Name + ".jpg", Format=".jpg")
    return Result

File2JpgFile = ImageFile2JpgFile = ImageFile2JpgImageFile

from matplotlib import pyplot as plt
def ImageFile2NpArray(FilePath):
    FilePath = DLUtils.StandardizeFilePath(FilePath)
    assert DLUtils.file.FileExists(FilePath)
    Image = plt.imread(FilePath)
    # data type: float. value range: [0.0, 1.0]
    # Image = cv2.cv.LoadImage(FilePath)
    if Image is not None: # some error occurs:
        return Image
    Image = cv2.imread(FilePath)
    return Image

def ImageUInt2Float(Image):
    return Image / 255.0

from .transform import ImageFile2Jpg, ImageFile2PNG, ImageFile2Webp

JPG2NpArray = Jpeg2NpArray = Jpg2NpArray = ImageFile2NpArray