import DLUtils.utils.image.compress as compress
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

def ImageFile2JpgImageFile(FilePath):
    Image = File2Image(FilePath)
    Name, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePath)
    Result = Image2File(Image, Name + ".jpg", Format=".jpg")
    return Result

def ExampleImage(Shape):
    FilePath = DLUtils.file.CurrentDirPath(__file__) + "test-image-lenna.png"

def ResizeImageAndCropToKeepAspectRatio(Image, Shape):
    # Shape: (height, width)
    Height, Width = Image.shape[0], Image.shape[1]
    Height1, Width1 = Shape[0]

    RatioHeight = Height1 / Height
    RatioWidth = Width1 / Width
    
    if RatioHeight > RatioWidth:
        Width2 = round(RatioHeight * Width) 
    

File2JpgFile = ImageFile2JpgFile = ImageFile2JpgImageFile
