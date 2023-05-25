from ast import Is
import os
import re
import pandas as pd
import shutil # sh_utils
import DLUtils
#from DLUtils.attr import *
import warnings

from DLUtils.utils._param import JsonFile2Param

def LastModifiedTime(Path):
    return os.path.getmtime(Path)

def MostRecentlyModified(PathList, Num=None):
    if Num is None:
        Num = len(PathList)
    if Num == 0:
        return None
    elif Num == 1:
        return PathList[0]
    else:
        Path = PathList[0]
        Time = LastModifiedTime(Path)
        for _Path in PathList[1:]:
            _Time = LastModifiedTime(_Path)
            if _Time > Time:
                Path = _Path
        return Path

def LoadMostRecentlySavedModelInDir(SaveDir):
    ModelFilePath = MostRecentlySavedModelPathInDir(SaveDir)
    Model = DLUtils.FromFile(ModelFilePath)
    return Model

def MostRecentlySavedModelPathInDir(SaveDir):
    SaveDir = StandardizePath(SaveDir)
    FileNameList = ListAllFiles(SaveDir)
    List = []
    for Name in FileNameList:
        if "model" in Name:
            List.append(Name)
    return MostRecentlyModified([SaveDir + FileName for FileName in List])  
MostRecentlySavedModelPath = MostRecentlySavedModelPathInDir

def AfterTrainModelFile(SaveDir):
    SaveDir = StandardizePath(SaveDir)
    FileNameList = ListAllFiles(SaveDir)
    List = []
    for Name in FileNameList:
        if "AfterTrain" in Name:
            List.append(Name)
    return MostRecentlyModified([SaveDir + FileName for FileName in List])

def FileNameFromPath(FilePath):
    FileName = os.path.basename(FilePath)
    return FileName

def ToStandardPathStr(Path):
    if Path is None:
        return None
    # if Path.endswith("/"): # means this is a folder.
    #     assert not ExistsFile(Path.rstrip("/"))
    if Path.startswith("~"):
        Path = AbsPath("~") + Path[1:]
    if ExistsDir(Path):
        if not Path.endswith("/"):
            Path += "/"
    Path = Path.replace("\\", "/") # windows style path
    return Path
StandardizePath = ToStandardPath = ToStandardPathStr
def MoveFile(FilePath, FilePathDest, RaiseIfNonExist=False, Overwrite=True, RaiseIfOverwrite=True):
    if not FileExists(FilePath):
        Msg = f"DLUtils.MoveFile: FilePath {FilePath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            warnings.warn(Msg)
    if DirExists(FilePathDest):
        FileName = FileNameFromPath(FilePath)
        _FilePathDest = FilePathDest
        FilePathDest =FilePathDest + FileName
    EnsureFileDir(FilePathDest)

    if FileExists(FilePathDest):
        if not Overwrite:
            Msg = f"DLUtils.MoveFile: FilePathDest {FilePathDest} already exist."
            if RaiseIfOverwrite:
                raise Exception(Msg)
            else:
                warnings.warn(Msg)
    shutil.move(FilePath, FilePathDest)
    # DeleteFile(FilePath) # shutil.move will delete file upon successful move.
    return True

def MoveFolder(FolderPath, FolderPathNew, RaiseIfNonExist=False, Overwrite=True):
    if not FolderExists(FolderPath):
        Msg = f"DLUtils.MoveFile: FilePath {FolderPath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            warnings.warn(Msg)

    CopyTree(FolderPath, FolderPathNew)
    DeleteTree(FolderPath)
    return True
MoveDir = MoveFolder

def CopyFiles(FileNameList, SourceDir, DestDir):
    for FileName in FileNameList:
        CopyFile(FileName, SourceDir, DestDir)
CopyFiles2DestDir = CopyFilesInSameFolder = CopyFiles

def CopyFile2AllSubDirsUnderDestDir(FileName, SourceDir, DestDir):
    for SubDir in ListAllDirs(DestDir):
        try:
            CopyFile2Folder(FileName, SourceDir, DestDir + SubDir)
        except Exception:
            continue

def CopyFile(FileName, SourceDir, DestDir):
    EnsureFileDir(DestDir + FileName)
    shutil.copy(SourceDir + FileName, DestDir + FileName)
CopyFile2Folder = CopyFile

def IsSameFile(FilePath1, FilePath2):
    return os.path.samefile(FilePath1, FilePath2)

def DeleteFile(FilePath, RaiseIfNonExist=False):
    if not FileExists(FilePath):
        Msg = f"DLUtils.DeleteFile: FilePath {FilePath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            warnings.warn(Msg)
    else:
        os.remove(FilePath)

def DeleteAllFilesAndSubFolders(DirPath):
    for FilePath in ListFilesPath(DirPath):
        DeleteFile(FilePath)
    for DirPath in ListDirsPath(DirPath):
        DeleteTree(DirPath)

def DeleteTree(FolderPath, RaiseIfNonExist=False):
    if not FolderExists(FolderPath):
        Msg = f"DLUtils.DeleteFile: FolderPath {FolderPath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            warnings.warn(Msg)
    shutil.rmtree(FolderPath)
    return
DeleteFolder = DeleteDir = DeleteTree

#from distutils.dir_util import copy_tree
def CopyFolder(SourceDir, DestDir):
    assert IsDir(SourceDir)
    if not DestDir.endswith("/"):
        DestDir += "/"
    CopyTree(SourceDir, DestDir)
    #shutil.copytree(SourceDir, DestDir) # Requires that DestDir not exists.
CopyDir2DestDir = CopyFolder2DestDir = CopyFolder

def FolderPathOfFile(FilePath):
    DirPath = os.path.dirname(os.path.realpath(FilePath)) + "/"
    return DLUtils.file.StandardizePath(DirPath)
DirPathOfFile = ParentFolderPath = FolderPathOfFile
CurrentDirPath = DirPathOfCurrentFile = FolderPathOfFile


def CurrentFilePath(FilePath):
    # FilePath: __file__ variable of caller .py file.
    return DLUtils.StandardizePath(FilePath)

def CurrentFileName(FilePath):
    return FileNameFromPath(FilePath)

from pathlib import Path

def FolderPathOfFolder(FilePath):
    Path = Path(FilePath)
    ParentFolderPath = Path.parent.absolute()

    return ParentFolderPath

def RemoveFiles(FilesPath):
    for FilePath in FilesPath:
        RemoveFile(FilePath)

def RemoveFile(FilePath):
    if not ExistsFile(FilePath):
        DLUtils.AddWarning("No such file: %s"%FilePath)
    else:
        os.remove(FilePath)

def RemoveDir(DirPath):
    assert ExistsDir(DirPath)
    shutil.rmtree(DirPath)
    return

def ClearDir(DirPath):
    if ExistsDir(DirPath):
        RemoveDir(DirPath)
    EnsureDir(DirPath)

def RemoveAllFilesUnderDir(DirPath, verbose=True):
    assert ExistsDir(DirPath)
    for FileName in GetAllFiles(DirPath):
        #FilePath = os.path.join(DirPath, FileName)
        FilePath = DirPath + FileName
        os.remove(FilePath)
        DLUtils.Log("utils_pytorch: removed file: %s"%FilePath)

def RemoveAllFilesAndDirsUnderDir(DirPath, verbose=True):
    assert ExistsDir(DirPath)
    Files, Dirs= GetAllFilesAndDirs(DirPath)
    for FileName in Files:
        FilePath = os.path.join(DirPath, FileName)
        os.remove(FilePath)
        DLUtils.Log("DLUtils: removed file: %s"%FilePath)
    for DirName in Dirs:
        DirPath = os.path.join(DirPath, DirName)
        #os.removedirs(DirPath) # Cannot delete subfolders
        import shutil
        shutil.rmtree(DirPath)
        DLUtils.Log("DLUtils: removed directory: %s"%DirPath)

def IsDir(DirPath):
    return os.path.isdir(DirPath)

def IsFile(FilePath):
    FilePath = ToAbsPath(FilePath)
    return os.path.isfile(FilePath)

def RemoveMatchedFiles(DirPath, Patterns):
    if not os.path.isdir(DirPath):
        raise Exception()
    if not DirPath.endswith("/"):
        DirPath += "/"
    if not isinstance(Patterns, list):
        Patterns = [Patterns]
    for Pattern in Patterns:
        FileNames = ListAllFiles(DirPath)
        for FileName in FileNames:
            MatchResult = re.match(Pattern, FileName)
            if MatchResult is not None:
                FilePath = os.path.join(DirPath, FileName)
                os.remove(FilePath)
                DLUtils.Log("DLUtils: removed file: %s"%FilePath)

def ListAllFilesAndDirs(DirPath):
    if not os.path.exists(DirPath):
        raise Exception()
    if not os.path.isdir(DirPath):
        raise Exception()
    Items = os.listdir(DirPath)
    Files, Dirs = [], []
    for Item in Items:
        if os.path.isfile(os.path.join(DirPath, Item)):
            Files.append(Item)
        elif os.path.isdir(os.path.join(DirPath, Item)):
            Dirs.append(Item + "/")
    return Files, Dirs
GetAllFilesAndDirs = ListAllFilesAndDirs

def IsEmptyDir(DirPath):
    Files, Dirs = ListAllFilesAndDirs(DirPath)
    return len(Files) == 0 and len(Dirs) == 0

def ListFileNames(DirPath):
    assert ExistsDir(DirPath), "Non-existing DirPath: %s"%DirPath
    assert os.path.isdir(DirPath), "Not a Dir: %s"%DirPath
    Items = os.listdir(DirPath)
    Files = []

    for Item in Items:
        if os.path.isfile(os.path.join(DirPath, Item)):
            Files.append(Item)
    return Files

ListAllFiles = ListAllFileNames = GetAllFiles = ListFilesName = ListFileNames

def ListFilePaths(DirPath):
    DirPath = StandardizePath(DirPath)
    FileNameList = ListFilesName(DirPath)
    return [DirPath + FileName for FileName in FileNameList]

ListAllFilesPath = ListAllFilePaths = GetAllFilePaths = GetAllFilesPath = ListFilesPath = ListFilesPaths = ListFilePaths

def ListDirs(DirPath):
    if not os.path.exists(DirPath):
        raise Exception()
    if not os.path.isdir(DirPath):
        raise Exception()
    Names = os.listdir(DirPath)
    Dirs = []
    for Name in Names:
        if os.path.isdir(DirPath + Name):
            Dir = Name + "/"
            Dirs.append(Dir)
    return Dirs
def ListDirsPath(DirPath):
    if not DirPath.endswith("/"):
        DirPath += "/"
    DirNameList = ListDirs(DirPath)
    return [DirPath + DirName for DirName in DirNameList]

ListAllDirs = GetAllDirs = ListDirs
ListAllFolders = ListDirs

def FileExists(FilePath):
    return os.path.isfile(FilePath)
ExistsFile = FileExists

def FolderExists(DirPath):
    IsDir = os.path.isdir(DirPath)
    if IsDir:
        if not DirPath.endswith("/"):
            DirPath += "/"
    return IsDir

ExistsDir = DirExists = FolderExists
ExistsFolder = FolderExists

def CheckFileExists(FilePath):
    if not DLUtils.ExistsFile(FilePath):
        raise Exception("%s does not exist."%FilePath)

def Path2AbsolutePath(Path):
    return os.path.abspath(Path)

def EnsureDirectory(FolderPath):
    #DirPathAbs = Path2AbsolutePath(DirPath)
    FolderPath = AbsPath(FolderPath)
    if os.path.exists(FolderPath):
        if not os.path.isdir(FolderPath):
            raise Exception("%s Already Exists but Is NOT a Directory."%FolderPath)
    else:
        if not FolderPath.endswith("/"):
            FolderPath += "/"
        os.makedirs(FolderPath)
EnsureDir = EnsureDirectory
EnsureFolder = EnsureDirectory

def EnsureFileDirectory(FilePath):
    assert not FilePath.endswith("/"), FilePath
    FilePath = Path2AbsolutePath(FilePath)
    FileDir = os.path.dirname(FilePath)
    EnsureDir(FileDir)
    return
EnsureFileDir = EnsureFileDirectory

def GetFileDir(FilePath):
    assert DLUtils.file.IsFile(FilePath)
    
    if DLUtils.SystemType == "windows":
        return os.path.dirname(FilePath) + "\\"
    else:
        return os.path.dirname(FilePath) + "/"

def EnsurePath(path, isFolder=False): # check if given path exists. if not, create it.
    if isFolder: # caller of this function makes sure that path is a directory/folder.
        if not path.endswith('/'): # folder
            DLUtils.AddWarning('%s is a folder, and should ends with /.'%path)
            path += '/'
        if not os.path.exists(path):
            os.makedirs(path)
    else: # path can either be a directory or a folder. If path exists, then it is what it is (file or folder). If not, depend on whether it ends with '/'.
        if os.path.exists(path): # path exists
            if os.path.isdir(path):
                if not path.endswith('/'): # folder
                    path += '/'     
            elif os.path.isfile(path):
                raise Exception('file already exists: %s'%str(path))
            else:
                raise Exception('special file already exists: %s'%str(path))
        else: # path does not exists
            if path.endswith('/'): # path is a folder
                path_strip = path.rstrip('/')
            else:
                path_strip = path
            if os.path.exists(path_strip): # folder with same name exists
                raise Exception('EnsurePath: homonymous file exists.')
            else:
                if not os.path.exists(path_strip):
                    os.makedirs(path_strip)
                    #os.mkdir(path) # os.mkdir does not support creating multi-level folders.
                #filepath, filename = os.path.split(path)
    return path

def CreateEmptyFile(FilePath):
    Str2File("", FilePath)
EmptyFile = CreateEmptyFile

def JoinPath(Path1, Path2):
    if not Path1.endswith('/'):
        Path1 += '/'
    Path2 = Path2.lstrip('./')
    if Path2.startswith('/'):
        raise Exception('JoinPath: Path2 is an absolute path: %s'%Path2)
    return Path1 + Path2

def CopyFolder(SourceDir, DestDir, exceptions=[], verbose=True):
    '''
    if args.path is not None:
        path = args.path
    else:
        path = '/data4/wangweifan/backup/'
    '''
    #EnsurePath(SourceDir)
    EnsurePath(DestDir)
    
    for i in range(len(exceptions)):
        exceptions[i] = os.path.abspath(exceptions[i])
        if os.path.isdir(exceptions[i]):
            exceptions[i] += '/'

    SourceDir = os.path.abspath(SourceDir)
    DestDir = os.path.abspath(DestDir)

    if not SourceDir.endswith('/'):
        SourceDir += '/'
    if not DestDir.endswith('/'):
        DestDir += '/'

    if verbose:
        print('Copying folder from %s to %s. Exceptions: %s'%(SourceDir, DestDir, exceptions))

    if SourceDir + '/' in exceptions:
        DLUtils.AddWarning('CopyFolder: neglected the entire root path. nothing will be copied')
        if verbose:
            print('neglected')
    else:
        _CopyFolder(SourceDir, DestDir, subpath='', exceptions=exceptions)

def _CopyFolder(SourceDir, DestDir, subpath='', exceptions=[], verbose=True):
    #EnsurePath(SourceDir + subpath)
    # efficient copy. skip file with same md5 to increase speed and decrease disk write.
    EnsurePath(DestDir + subpath)
    items = os.listdir(SourceDir + subpath)
    for item in items:
        #print(DestDir + subpath + item)
        path = SourceDir + subpath + item
        if os.path.isfile(path): # is a file
            if path + '/' in exceptions:
                if verbose:
                    print('neglected file: %s'%path)
            else:
                if os.path.exists(DestDir + subpath + item):
                    Md5Source = File2MD5(SourceDir + subpath + item)
                    Md5Target = File2MD5(DestDir + subpath + item)
                    if Md5Target==Md5Source: # same file
                        #print('same file')
                        continue
                    else:
                        #print('different file')
                        os.system('rm -r "%s"'%(DestDir + subpath + item))
                        os.system('cp -r "%s" "%s"'%(SourceDir + subpath + item, DestDir + subpath + item))     
                else:
                    os.system('cp -r "%s" "%s"'%(SourceDir + subpath + item, DestDir + subpath + item))
        elif os.path.isdir(path): # is a folder.
            if path + '/' in exceptions:
                if verbose:
                    print('neglected folder: %s'%(path + '/'))
            else:
                _CopyFolder(SourceDir, DestDir, subpath + item + '/', verbose=verbose)
        else:
            DLUtils.AddWarning('%s is neither a file nor a path.')

def ExistsPath(Path):
    return os.path.exists(Path)

def GetFileSuffix(FilePath):
    return SeparateFileNameSuffix(FilePath)[1]

def AppendSuffix2FileName(FileName, Suffix):
    Name, _Suffix = SeparateFileNameSuffix(FileName)
    if _Suffix in [""] or Suffix is None:
        return Name + Suffix
    else:
        return Name + Suffix + "." + _Suffix

AppendSuffixOnFileName = AppendSuffix2FileName

def AppendSuffixOnCurrentFileName(__File__, Suffix):
    FilePath = CurrentFilePath(__File__)
    return AppendSuffixOnFileName(FilePath, Suffix)

AppendOnCurrentFileName = AppendSuffixOnCurrentFileName

def SeparateFileNameSuffix(FilePath):
    if FilePath.endswith("/"):
        raise Exception()
    MatchResult = re.match(r"(.*)\.(.*)", FilePath)
    if MatchResult is None:
        return FilePath, ""
    else:
        return MatchResult.group(1), MatchResult.group(2)
    # to be checked
    # for filename with multiple '.', such as a.b.c, (a.b, c) should be returned

def AppendOnCurrentFileNameAndChangeSuffix(__File__, Append, Suffix):
    FilePath = CurrentFilePath(__File__)
    Name, _Suffix = SeparateFileNameSuffix(FilePath)
    Suffix = Suffix.lstrip(".")
    return Name + Append + "." + Suffix
    
ParseFileNameSuffix = SeparateFileNameSuffix

def AddSuffixToFileWithFormat(FilePath, Suffix):
    _FilePath, Format = ParseFileNameSuffix(FilePath)
    return _FilePath + Suffix + "." + Format

def RenameFile(DirPath, FileName, FileNameNew):
    DirPath = CheckDir(DirPath)
    assert ExistsFile(DirPath + FileName)
    os.rename(DirPath + FileName, DirPath + FileNameNew)

def RenameFileIfExists(FilePath):
    if FilePath.endswith("/"):
        raise Exception()
    FileName, Suffix = ParseFileNameSuffix(FilePath)
    Sig = True
    MatchResult = re.match(r"^(.*)-(\d+)$", FileName)
    if MatchResult is None:
        if ExistsPath(FilePath):
            os.rename(FilePath, FileName + "-0" + "." + Suffix)
            FileNameOrigin = FileName
            Index = 1
        elif ExistsPath(FileName + "-0" + "." + Suffix):
            FileNameOrigin = FileName
            Index = 1
        else:
            Sig = False
    else:
        FileNameOrigin = MatchResult.group(1)
        Index = int(MatchResult.group(2)) + 1
    if Sig:
        while True:
            FilePath = FileNameOrigin + "-%d"%Index + "." + Suffix
            if not ExistsPath(FilePath):
                return FilePath
            Index += 1
    else:
        return FilePath
RenameIfFileExists = RenameFileIfExists

def RenameDir(FolderPath, FolderNameNew):
    # providing a new folder path is not allowed
    # since this might result in move, rather than rename.
    FolderNameNew = FolderNameNew.rstrip("/")
    if not ExistsDir(FolderPath):
        DLUtils.AddWarning("RenameDir: Dir %s does not exist."%FolderPath)
        return False
    ParentFolderPath = FolderPathOfFolder(FolderPath)
    FolderPathNew = ParentFolderPath + FolderNameNew
    assert not ExistsFile(FolderPathNew.rstrip("/"))
    os.rename(FolderPath, FolderPathNew)

RennameFolder = RenameDir

def RenameDirIfExists(DirPath):
    DirPath = DirPath.rstrip("/")
    MatchResult = re.match(r"^(.*)-(\d+)$", DirPath)
    Sig = True
    if MatchResult is None:
        if ExistsPath(DirPath):
            #os.rename(DirPath, DirPath + "-0") # os.rename can apply to both folders and files.
            shutil.move(DirPath, DirPath + "-0")
            DirPathOrigin = DirPath
            Index = 1
        elif ExistsPath(DirPath + "-0"):
            DirPathOrigin = DirPath
            Index = 1
        else:
            Sig = False
    else:
        DirPathOrigin = MatchResult.group(1)
        Index = int(MatchResult.group(2)) + 1
    if Sig:
        while True:
            DirPath = DirPathOrigin + "-%d"%Index
            if not ExistsPath(DirPath):
                break
            Index += 1
    DirPath += "/"
    return DirPath

def Str2File(Str, FilePath):
    DLUtils.EnsureFileDir(FilePath)
    with open(FilePath, "w") as File:
        File.write(Str)

def Bytes2File(Bytes, FilePath):
    DLUtils.EnsureFileDir(FilePath)
    with open(FilePath, "wb") as File:
        File.write(Bytes)

def File2Bytes(FilePath):
    DLUtils.EnsureFileDir(FilePath)
    with open(FilePath, "rb") as File:
        Bytes = File.read()
    return Bytes

def Tensor2TextFile2D(Data, SavePath="./test/"):
    Data = DLUtils.ToNpArray(Data)
    DLUtils.NpArray2D2TextFile(Data, SavePath=SavePath)

def cal_path_from_main(path_rel=None, path_start=None, path_main=None):
    # path_rel: file path relevant to path_start
    if path_main is None:
        path_main = sys.path[0]
    if path_start is None:
        path_start = path_main
        warnings.warn('cal_path_from_main: path_start is None. using default: %s'%path_main)
    path_start = os.path.abspath(path_start)
    path_main = os.path.abspath(path_main)
    if os.path.isfile(path_main):
        path_main = os.path.dirname(path_main)
    if not path_main.endswith('/'):
        path_main += '/' # necessary for os.path.relpath to calculate correctly
    if os.path.isfile(path_start):
        path_start = os.path.dirname(path_start)
    #path_start_rel = os.path.relpath(path_start, start=path_main)

    if path_rel.startswith('./'):
        path_rel.lstrip('./')
    elif path_rel.startswith('/'):
        raise Exception('path_rel: %s is a absolute path.'%path_rel)
    
    path_abs = os.path.abspath(os.path.join(path_start, path_rel))
    #file_path_from_path_start = os.path.relpath(path_rel, start=path_start)
    
    path_from_main = os.path.relpath(path_abs, start=path_main)

    #print('path_abs: %s path_main: %s path_from_main: %s'%(path_abs, path_main, path_from_main))
    '''
    print(main_path)
    print(path_start)
    print('path_start_rel: %s'%path_start_rel)
    print(file_name)
    print('file_path: %s'%file_path)
    #print('file_path_from_path_start: %s'%file_path_from_path_start)
    print('file_path_from_main_path: %s'%file_path_from_main_path)
    print(DestDir_module(file_path_from_main_path))
    '''
    #print('path_rel: %s path_start: %s path_main: %s'%(path_rel, path_start, path_main))
    return path_from_main

import pickle
from pathlib import Path

def File2Str(FilePath):
    return Path(FilePath).read_text()

def FileSizeInBytes(FilePath): 
    return os.path.getsize(FilePath)
FileSize = FileSizeInBytes

KB = 1024.0
MB = 1024.0 * 1024.0
GB = 1024.0 * 1024.0 * 1024.0

def Size2Str(SizeB, Base=1024):
    if Base == 1024 or Base == 1024.0:
        pass
    if SizeB > Base:
        SizeKB = SizeB * 1.0 / KB
    else:
        return "%.2f B"%(SizeB)
    if SizeKB > Base:
        SizeMB = SizeB * 1.0 / MB
    else:
        return "%.2f MB"%(SizeKB)
    if SizeMB > Base:
        SizeGB = "%.2f GB"%(SizeB * 1.0 / GB)
    else:
        return "%.2f MB"%(SizeMB)
    return "%.2f MB"%(SizeGB)

def FileSizeStr(FilePath):
    Bytes = FileSizeInBytes(FilePath)
    return Size2Str(Bytes)

def File2ObjPickle(FilePath):
    with open(FilePath, 'rb') as f:
        Obj = pickle.load(f, encoding='bytes')
    return Obj
File2Obj = File2ObjPickle
BinaryFile2Obj = File2ObjPickle
def Obj2FilePickle(Obj, FilePath):
    DLUtils.EnsureFileDir(FilePath)
    with open(FilePath, "wb") as f:
        pickle.dump(Obj, f)
Obj2File = Obj2FilePickle
Obj2BinaryFile = Obj2FilePickle
JsonObj2DataFile = Obj2File

def Str2TextFile(Str, FilePath):
    with open(FilePath, 'w') as f:
        f.write(Str)

def File2MD5(FilePath):
    import hashlib
    Md5Calculator = hashlib.md5()
    assert DLUtils.ExistsFile(FilePath), FilePath
    with open(FilePath, 'rb') as f:
        bytes = f.read()
    Md5Calculator.update(bytes)
    Md5Str = Md5Calculator.hexdigest()
    return Md5Str

def FileList2Md5(FilePathList):
    Md5List = []
    for FilePath in FilePathList:
        Md5 = File2MD5(FilePath)
        Md5List.append(Md5)
    return Md5List

def ListFilesAndCalculateMd5(DirPath, Md5InKeys=False):
    Files = DLUtils.ListAllFiles(DirPath)
    Dict = {}
    if Md5InKeys:
        for FileName in Files:
            Md5 = DLUtils.file.File2MD5(DirPath + FileName)
            Dict[Md5] = FileName      
    else:
        for FileName in Files:
            Md5 = DLUtils.file.File2MD5(DirPath + FileName)
            Dict[FileName] = Md5
    return Dict

ListFilesAndMd5 = ListFilesAndCalculateMd5


import pathlib
def _AbsPath(Path):
    if "~" in Path:
        Path = os.path.expanduser(Path)
    PathAbs = os.path.abspath(Path)
    if IsDir(PathAbs) and not PathAbs.endswith("/"):
        PathAbs += "/"
    return PathAbs
def AbsPath(Path):
    if Path.endswith("/"):
        EndsWithSlash = True
    else:
        EndsWithSlash = False
    if "~" in Path:
        Path = os.path.expanduser(Path)
    PathAbs = pathlib.Path(Path).absolute()
    PathAbs = str(PathAbs)

    if not PathAbs.endswith("/"):
        if EndsWithSlash:
            PathAbs += "/"
    # if ExistsFolder(PathAbs) and not PathAbs.endswith("/"):
    #    PathAbs += "/"
    return PathAbs
ToAbsPath = AbsPath

def GetRelativePath(PathTarget, PathRef):
    PathTarget = ToAbsPath(PathTarget)
    PathRef = ToAbsPath(PathRef)
    PathRef2Target = PathTarget.replace(PathRef, ".")
    # To be implemented: support forms such as '../../a/b/c'
    return PathRef2Target

def CheckDir(DirPath):
    if not DirPath.endswith("/"):
        DirPath += "/"
    assert IsDir(DirPath)
    return DirPath

CheckDirPath = CheckDir

def VisitDirAndApplyMethodOnFiles(DirPath=None, Method=None, Recur=False, **Dict):
    DirPath = CheckDirPath(DirPath)
    
    if Method is None:
        Method = lambda Context:0
        DLUtils.AddWarning('Method is None.')

    
    FileList, DirList = ListAllFilesAndDirs(DirPath)

    for FileName in FileList:
        Method(DLUtils.PyObj({
            "DirPath": DirPath,
            "FileName": FileName
        }))

    if Recur:
        for DirName in DirList:
            VisitDirAndApplyMethodOnFiles(DirPath + DirName + "/", Method, Recur, **Dict)

def VisitDirAndApplyMethodOnDirs(DirPath=None, Method=None, Recur=False, **Dict):
    DirPath = CheckDirPath(DirPath)
    
    if Method is None:
        Method = lambda Context:0
        DLUtils.AddWarning('Method is None.')

    DirList = ListAllDirs(DirPath)

    for DirName in DirList:
        Method(DLUtils.PyObj({
            "ParentDirPath": DirPath,
            "DirName": DirName
        }))

    # In case DirName is changed in Method.
    DirList = ListAllDirs(DirPath)

    if Recur:
        for DirName in DirList:
            VisitDirAndApplyMethodOnFiles(DirPath + DirName + "/", Method, Recur, **Dict)



def EnsureDirFormat(Dir):
    if not Dir.endswith("/"):
        Dir += "/"
    return Dir

def CopyFilesAndDirs2DestDir(Names, SourceDir, DestDir):
    SourceDir = EnsureDirFormat(SourceDir)
    DestDir = EnsureDirFormat(DestDir)
    for Name in Names:
        ItemPath = SourceDir + Name
        if DLUtils.IsDir(ItemPath):
            _SourceDir = EnsureDirFormat(ItemPath)
            _DestDir = EnsureDirFormat(DestDir + Name)
            EnsureDir(_DestDir)
            CopyDir2DestDir(_SourceDir, _DestDir)
        elif DLUtils.IsFile(ItemPath):
            CopyFile2Folder(Name, SourceDir, DestDir)
        else:
            raise Exception()


def SplitPaths(Paths):
    PathsSplit = []
    for Path in Paths:
        PathsSplit.append(SplitPath(Path))
    return PathsSplit
def SplitPath(Path):
    return Path

def CopyTree(SourceDir, DestDir, **kw):
    kw.setdefault("SubPath", "")
    Exceptions = kw.setdefault("Exceptions", []) # to be implemented: allowing excetionpaths                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    Files, Dirs = ListAllFilesAndDirs(SourceDir)
    for File in Files:
        # if File in Exceptions[0]:
        #     continue
        CopyFile2Folder(File, SourceDir, DestDir)
    for Dir in Dirs:
        EnsureDir(DestDir + Dir)
        CopyTree(SourceDir + Dir, DestDir + Dir, **kw)

def Data2TextFile(data, Name=None, FilePath=None):
    if FilePath is None:
        FilePath = DLUtils.GetSavePathFromName(Name, Suffix=".txt")
    DLUtils.Str2File(str(data), FilePath)

from .json import PyObj2DataFile, DataFile2PyObj, PyObj2JsonFile, \
    JsonFile2PyObj, JsonFile2JsonDict, JsonObj2JsonFile, DataFile2JsonObj, JsonFile2Dict

from ._param import JsonDict2Str

def FolderConfig(FolderPath):
    FolderPath = ToAbsPath(FolderPath)

    if not FolderPath.endswith("/"):
        FolderPath += "/"
    Tree = DLUtils.param({})
    NodeList = [
        DLUtils.param({
            "FolderPath": FolderPath,
            "ParamNode": Tree
        })
    ]
    while len(NodeList) > 0:
        Node = NodeList.pop() # represents a folder
        ParamNode = Node.ParamNode
        FolderPath = Node.FolderPath
        ParamNode.setattr("Files", DLUtils.param({}))
        ParamNode.setattr("Folders", DLUtils.param({}))
        Files = ParamNode.Files
        for FileName in ListAllFiles(FolderPath):
            FilePath = FolderPath + FileName
            Files.setattr(
                FileName.replace(".", "(DOT)"),
                DLUtils.param({
                    "MD5": File2MD5(FolderPath + FileName),
                    "Volume": FileSizeStr(FilePath)
                })
            )
        Folders = ParamNode.Folders
        for FolderName in ListAllDirs(FolderPath):
            FolderName = FolderName.replace(".", "(DOT)")
            Folders.setattr(
                FolderName.rstrip("/"), DLUtils.param({})
            )
            NodeList.append(
                DLUtils.param({
                    "FolderPath": FolderPath + FolderName, # already ends with /
                    "ParamNode": Folders.getattr(FolderName.rstrip("/"))
                })
            )
    return Tree

def CheckIntegrity(FolderPath, Config, RaiseException=True):
    FolderPath = ToAbsPath(FolderPath)
    FolderNodeList = []
    ConfigNodeList = []
    FolderNodeList.append(FolderPath)   
    ConfigNodeList.append(Config)

    while len(ConfigNodeList) > 0:
        ConfigNode = ConfigNodeList.pop()
        FolderPath = FolderNodeList.pop()
        FolderPath = FolderPath.replace("(DOT)", ".")
        FilesNode = ConfigNode.Files
        Files = ListAllFiles(FolderPath)
        for FileName, FileInfo in FilesNode.items():
            FileName = FileName.replace("(DOT)", ".")
            SubFilePath = FolderPath + FileName
            if not FileExists(SubFilePath):
                if RaiseException:
                    raise Exception()
                else:
                    return False
            if not FileInfo.MD5 == File2MD5(SubFilePath):
                if RaiseException:
                    raise Exception()
                else:
                    return False
        FoldersNode = ConfigNode.Folders
        for FolderName, FolderInfo in FoldersNode.items():
            FolderName = FolderName.replace("(DOT)", ".")
            SubFolderPath = FolderPath + FolderName
            if not ExistsDir(SubFolderPath):
                if RaiseException:
                    raise Exception()
                else:
                    return False
            ConfigNodeList.append(FolderInfo)
            FolderNodeList.append(SubFolderPath)
    return True

import gzip
import shutil
def ExtractGzFile(FilePath, ExtractFilePath=None):
    # file -> file
    if ExtractFilePath is None:
        ExtractFilePath = DLUtils.RemoveSuffix(FilePath, ".gz")
        if ExtractFilePath is None:
            raise Exception(ExtractFilePath)
    with gzip.open(FilePath, 'rb') as FileIn:
        with open(ExtractFilePath, 'wb') as FileOut:
            shutil.copyfileobj(FileIn, FileOut)
    return ExtractFilePath
def IsGzFile(FilePath):
    with open(FilePath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'
def _ExtractGzFile(FilePath, SavePath=None):
    # file -> file
    if SavePath is None:
        SavePath = DLUtils.RemoveSuffix(FilePath, "tar.gz")
        if SavePath is None:
            raise Exception(SavePath)
    with gzip.open(FilePath, 'rb') as FileIn:
        with open(SavePath, 'wb') as FileOut:
            shutil.copyfileobj(FileIn, FileIn)
import tarfile
# Input: Zip File
# Output: Extracted Folder
def ExtractTarFile(FolderPath, ExtractFolderPath):
    # file -> folder
    # importing the "tarfile" module
    File = tarfile.open(FolderPath)
    # extracting file
    File.extractall(ExtractFolderPath)
    File.close()
    return ExtractFolderPath

UncompressTarGzFile = ExtractTarFile

def IsTarFile(FilePath):
    return tarfile.is_tarfile(FilePath)

import zipfile
# Input: Zip File
# Output: Extracted Folder
def ExtractZipFile(ZipFilePath, ExtractDir):
    # file -> folder
    ExtractDir = EnsureDir(ExtractDir)
    with zipfile.ZipFile(ZipFilePath, 'r') as zip_ref:
        zip_ref.extractall(ExtractDir)
    return ExtractDir

ZipFile2Folder = ExtractZipFile
ExtractZipFile = ExtractZipFile
UnzipFile = ExtractZipFile

def IsZipFile(FilePath):
    return zipfile.is_zipfile(FilePath)

def Folder2ZipFile(ZipPathList, ZipFilePath):
    return

def FilesUnderDir2ZipFile(DirPath, ZipFilePath):
    DirPath = StandardizePath(DirPath)
    assert ExistsDir(DirPath)
    FilePathList = ListFilesPath(DirPath)
    FileList2ZipFile(FilePathList, ZipFilePath)
    return ZipFilePath

def FileList2ZipFile(FilePathList, ZipFilePath):
    # Create a ZipFile Object
    with zipfile.ZipFile(ZipFilePath, 'w') as zipObj:
        for FilePath in FilePathList:
            FilePathAbs = AbsPath(FilePath) # turn to absolute path
            zipObj.write(FilePathAbs, os.path.basename(FilePathAbs))
    return ZipFilePath

def JsonDict2JsonFile(JsonDict, FilePath):
    JsonStr = JsonDict2Str(JsonDict)
    Str2TextFile(JsonStr, FilePath)
JsonDict2File = JsonDict2JsonFile

import cv2
def Jpg2NpArray(Path):
    Path = DLUtils.StandardizePath(Path)
    assert FileExists(Path)
    Image = cv2.imread(Path)
    assert Image is not None
    return Image
JPG2NpArray = Jpeg2NpArray = Jpg2NpArray

def RemoveAllFileWithSuffix(DirPath, Suffix):
    DirPath = StandardizePath(DirPath)
    FileNameList = ListAllFileNames(DirPath)
    FileNamePattern = fr"(.*)\.(Suffix)"%Suffix
    FileNamePatternCompiled = re.compile(FileNamePattern)
    for FileName in FileNameList:
        MatchResult = FileNamePatternCompiled.match(FileName)
        if MatchResult is not None:
            DeleteFile(DirPath + FileName)
    return True

RemoveFileWithSuffix = RemoveAllFileWithSuffix

def RemoveAllPNGFile():
    DLUtils.file.RemoveMatchedFiles("./", r".*\.png")

def ParseSavePath(SaveDir=None, SaveName=None, SaveNameDefault=None):
    if SaveName is None:
        if SaveDir.endswith("/"):
            return SaveDir + SaveNameDefault
        else:
            return SaveDir
    else:
        if SaveDir is None:
            raise Exception()
        else:
            if not SaveDir.endswith("/"):
                SaveDir += "/"
            assert not SaveName.endswith("/")
            return SaveDir + SaveName