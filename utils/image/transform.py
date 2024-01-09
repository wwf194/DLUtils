import warnings
from PIL import Image as Im
import DLUtils

try: # pip install pillow-avif-plugin
    import pillow_avif # Pillow avif support
except Exception:
    if DLUtils.Verbose:
        warnings.warn("lib pillow_avif not found.")

try: # pip3 install pillow-heif
    import pillow_heif # Pillow heif support
    pillow_heif.register_heif_opener()
except Exception:
    if DLUtils.Verbose:
        warnings.warn("lib pillow_heif not found.")

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

try:
    import torchvision
except Exception:
    pass

try:
    import cairosvg
except Exception:
    warnings.warn("package cairosvg not found.")
else:
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
    
