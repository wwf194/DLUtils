import glob
from PIL import Image as Im

import DLUtils

def ImageFile2Jpg(FilePath, SavePath=None):
    FilePath = DLUtils.CheckFileExists(FilePath)
    Image = Im.open(FilePath).convert("RGB")
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(FilePath, ".jpg")
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
    
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(FilePath, ".png")
    SavePath = DLUtils.EnsureFileDir(SavePath)
    Image.save(SavePath, "png")
 
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