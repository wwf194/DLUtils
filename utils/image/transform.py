import glob
from PIL import Image as Im

import DLUtils

def ImageFile2Jpg(FilePath, SavePath):
    FilePath = DLUtils.CheckFileExists(FilePath)
    im = Im.open(s).convert("RGB")
    im.save(str(format(i, "04"))+".jpg", "jpeg", quality=1)
 
def ImageFile2PNG(FilePath, SavePath=None):
    FilePath = DLUtils.CheckFileExists(FilePath)
    Image = Im.open(FilePath).convert("RGB")
    
    if SavePath is None:
        SavePath = DLUtils.file.ChangeFileNameSuffix(".png")
    SavePath = DLUtils.EnsureFIleDir(SavePath)
    Image.save(str(format(SavePath, "04"))+".png", "png")
 
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