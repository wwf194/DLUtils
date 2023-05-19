
import DLUtils.utils.image.compress as compress
from .compress import File2Image, Image2File, ImageNp2File, ResizeImage

import DLUtils
import cv2
def TextOnImage(Image):
    cv2.PutText()
    
def File2JpgFile(FilePath):
    Image = File2Image(FilePath)
    Name, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePath)
    Result = Image2File(Image, Name + ".jpg", Format=".jpg")
    return Result


    