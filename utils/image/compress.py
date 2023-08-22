import sys
sys.path.append("../")
import DLUtils

import cv2
import PIL

def CompressImageFile(FilePathSource, FilePathDest=None, *List, **Dict):
    if FilePathDest is None:
        Name, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePathSource)
        FilePathDest = Name + "-compressed" + "." + Suffix
    
    Image = File2Image(FilePathSource)
    ImageNew = CompressImage(Image, Ratio=0.5)
    Image2File(ImageNew, FilePathDest)
                
def CompressImage(Image, Ratio=0.5):
    ImageNew = ResizeImage(Image, Ratio=Ratio)
    # ImageData: ImageNp
    return ImageNew

def ResizeImage(Image, Height=None, Width=None, Ratio=None, KeepShape=False, MaxHeight=None, MaxWidth=None):
    Height0 = Image.shape[0]
    Width0 = Image.shape[1]
    if KeepShape:
        if Height is not None:
            assert Width is None
            assert Ratio is not None
            Ratio = Height / Height0
            Width = round(Ratio * Width0)
        if Width is not None:
            assert Height is None
            assert Ratio is None
            Ratio = Width / Width0
            Height = round(Ratio * Height0)
    
    if Ratio is None:
        assert Height is None
        assert Width is None
        Height = round(Image)
    else:
        assert isinstance(Ratio, float)
        Height = round(Ratio * Height0)
        Width = round(Ratio * Width0)
        
    ImageNew = cv2.resize(Image, (Width, Height), interpolation=cv2.INTER_AREA)
    return ImageNew

import numpy as np
def File2Image(FilePath):
    # cv2.imread might have problem when FilePath contains unicode characters.
    Stream = open(FilePath, "rb")
    ImageBytes = bytearray(Stream.read())
    ImageNp = np.asarray(ImageBytes, dtype=np.uint8)
    try:
        ImageCv = cv2.imdecode(ImageNp, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    return ImageCv

def ImageNp2FileCv(Image, FilePath, Format=None):
    # ImageData: np.ndarray
    FilePath = DLUtils.EnsureFileDir(FilePath)
    # Result = cv2.imwrite(FilePath, Image)
    # cv2.imwrite might have problem when FilePath contains unicode characters.
    
    if Format is None:
        Suffix = DLUtils.file.GetFileSuffix(FilePath)
        DLUtils.file.SeparateFileNameSuffix(FilePath)
        if Suffix == "":
            Format = "png"
        else:
            Format = Suffix
    else:
        assert isinstance(Format, str)
        Format = Format.lstrip(".")

    # convert to jpeg and save in variable
    ImageBytes = cv2.imencode("." + Format, Image)[1].tobytes()
        # note cv2 uses bgr not rgb.
    with open(FilePath, 'wb') as f:
        f.write(ImageBytes)
    return True

from PIL import Image as Im
def ImageUnit82FilePIL(Image, FilePath):
    FilePath = DLUtils.EnsureFileDir(FilePath)
    ImagePIL = Im.fromarray(Image)
    ImagePIL.save(FilePath)
Image2File = ImageNp2File = ImageNp2FilePIL = ImageUnit8ToFilePIL = ImageUnit82FilePIL

def ImageFloat012NpFile(Image, FilePath):
    return ImageNp2File((Image * 255.0).astype(np.uint8), FilePath)
ImageFloat01ToNpFile = ImageFloat012NpFile

def Test():
    ImageFilePath = DLUtils.file.DirPathOfFile(__file__) + "test-image-lenna.png"
    # ImageFilePath = "Z:/temp/1.png"
    # CompressImageFile(ImageFilePath, Ratio=0.5)
    # DLUtils.utils.image.File2JpgFile(ImageFilePath)
    Image0 = File2Image(ImageFilePath) # (height, width, channel)
    for Index in range(10):
        Image = Image0.copy()
        Str = "%02d"%Index
        DLUtils.image.TextOnImageCenter(Image, Str, Color=(0, 255, 0))
        import matplotlib.pyplot as plt
        plt.imshow(Image)
        Image2File(Image, DLUtils.file.AppendSuffix2FileName(ImageFilePath, "-text-%s"%Str))

if __name__ == '__main__':
    Test()

def CompressImageAtFolder(DirPath):
    ListFiles = DLUtils.file.ListAllFileNames(DirPath)
    return
