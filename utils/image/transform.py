import glob
from PIL import Image as Im
import DLUtils

try: # pip install pillow-avif-plugin Pillow
    import pillow_avif # Pillow avif support
except Exception:
    DLUtils.Write("pillow_avif failed to import.")

try: # pip3 install pillow-heif
    import pillow_heif # Pillow avif support
    pillow_heif.register_heif_opener()
except Exception:
    DLUtils.Write("pillow_heif failed to import.")

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

def HEIF2PNG(FilePath, SavePath=None):
    FilePath = DLUtils.CheckFileExists(FilePath)
    heif_file = pillow_heif.read_heif(FilePath)
    Image = Im.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw"
    )
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(FilePath, ".png")
        SavePath = DLUtils.EnsureFileDir(SavePath)
    SavePath = DLUtils.EnsureFileDir(SavePath)
    Image.save(SavePath, "png")
    return SavePath
HEIC2PNG = HEIF2PNG

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

# files = glob.glob(".\*.jpg")
# i = 0
# for file in files:
#     cnvt2png(file, i)
#     i += 1

try:
    import torchvision
except Exception:
    pass
