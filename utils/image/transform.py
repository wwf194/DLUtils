import warnings
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PIL import Image as Im
    from PIL import ExifTags
    import pillow_heif
    import pillow_avif
    from pillow_heif import register_heif_opener
else:
    Im = DLUtils.LazyFromImport("PIL", "Image")
    ExifTags = DLUtils.LazyFromImport("PIL", "ExifTags")
    pillow_heif = DLUtils.LazyImport("pillow_heif", FuncAfterImport=lambda module:module.register_heif_opener())
    pillow_avif = DLUtils.LazyImport("pillow_avif")

def ImageFile2Jpg(FilePath, SavePath=None):
    FilePath = DLUtils.CheckFileExists(FilePath)
    Image = Im.open(FilePath).convert("RGB")
        # .avif image requires pillow_avif
        # .heif image requires pillow_heif
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(FilePath, ".jpg")
        SavePath = DLUtils.EnsureFileDir(SavePath)
    SavePath = DLUtils.EnsureFileDir(SavePath)
    Image.save(SavePath, "jpeg",
        # subsampling=0,
        quality=50
    )
    return SavePath

ToJPGFile = ToJpgFile = ToJpg = ToJPG = ImageFile2Jpg

def ImageFile2PNG(FilePath, SavePath=None):
    FilePath = DLUtils.CheckFileExists(FilePath)
    Image = Im.open(FilePath).convert("RGB")
    # supported format: jpg, jfif, avif, webp.
        # .avif image requires pillow_avif
        # .heif image requires pillow_heif
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(FilePath, ".png")
        SavePath = DLUtils.EnsureFileDir(SavePath)
    SavePath = DLUtils.file.RenameFileIfExists(SavePath)
    Image.save(SavePath, "png")
    assert DLUtils.ExistsFile(SavePath)
    return SavePath
ToPNGFile = ToPNG = ImageFile2PNG

def ImageFile2Webp(FilePath, SavePath):
    FilePath = DLUtils.CheckFileExists(FilePath)
    Image = Im.open(FilePath).convert("RGB")
    
    SavePath = DLUtils.EnsureFIleDir(SavePath)
    Image.save(str(format(SavePath, "04"))+".webp", "webp")
ImageFile2webp = ImageFile2Webp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torchvision
    import cairosvg
else:
    torchvision = DLUtils.LazyImport("torchvision")
    cairosvg = DLUtils.LazyImport("cairosvg")

def SVGStr2PNG(Str, SavePath, Scale=2.0):
    SavePath = DLUtils.EnsureFileDir(SavePath)
    cairosvg.svg2png(bytestring=Str,write_to=SavePath, scale=Scale)
    return SavePath
def SVG2PNG(
        FilePath,
        SavePath=None,
        Scale=2.0 # spatial scale
    ):
    if SavePath is None:
        SavePath = DLUtils.ChangeFileNameSuffix(FilePath, ".svg")
    SvgStr = DLUtils.TextFile2Str(FilePath)
    SavePath = SVGStr2PNG(SvgStr, SavePath, Scale=Scale)
    return SavePath
def SVG2NpArray(FilePath, Scale=2.0):
    SvgStr = DLUtils.TextFile2Str(FilePath)        
    TempFilePath = "output.png"
    TempFilePath = DLUtils.RenameFileIfExists(TempFilePath)
    SavePath = SVG2PNG(FilePath, SavePath=TempFilePath, Scale=Scale)
    Image = DLUtils.image.ImageFile2NpArray(SavePath)
    DLUtils.DeleteFile(SavePath)
    return Image
    
