import os
import re
import warnings
import sys
import gzip

import traceback
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import shutil
    import pickle
    from send2trash import send2trash
    import hashlib
    import filecmp
    import json
    import pathlib
    from pathlib import Path
else:
    send2trash = DLUtils.LazyFromImport("send2trash", "send2trash")
    shutil = DLUtils.LazyImport("shutil")
    pickle = DLUtils.LazyImport("pickle")
    hashlib = DLUtils.LazyImport("hashlib")
    filecmp = DLUtils.LazyImport("filecmp")
    json = DLUtils.LazyImport("json")
    pathlib = DLUtils.LazyImport("pathlib")
    Path = DLUtils.LazyFromImport("pathlib", "Path")

from ._json import (
    PyObj2DataFile, DataFile2PyObj, PyObj2JsonFile,
    JsonFile2PyObj, JsonFile2JsonDict, JsonObj2JsonFile,
    DataFile2JsonObj, JsonFile2Dict, Obj2JsonFile
)
from ._param import JsonDict2Str, JsonFile2Param

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

def FileNameFromPath(FilePath, StripSuffix=False):
    assert not FilePath.endswith("/")

def FileNameFromFilePath(FilePath, StripSuffix=False):
    FileName = os.path.basename(FilePath)
    if StripSuffix:
        Name, Suffix = SeparateFileNameSuffix(FilePath)
        return Name
    else:
        return FileName
FileNameFromPath = FileNameFromFilePath

def CurrentFileName(__File__, StripSuffix=False):
    """
    __File__: __file__ variable of caller script.
    """
    return FileNameFromPath(__File__, StripSuffix=StripSuffix)

def CurrentFilePath(__File__, StripSuffix=False):
    """
    __File__: __file__ variable of caller script.
    """
    FilePath = StandardizeFilePath(__File__)
    if StripSuffix:
        Name, Suffix = SeparateFileNameSuffix(FilePath)
        return Name
    else:
        return FilePath
def DirNameFromDirPath(DirPath: str):
    if DirPath.endswith("/"):
        DirPath = DirPath.rstrip("/")
    return FileNameFromPath(DirPath)

DirNameFromPath = DirNameFromDirPath

def ToStandardPathStr(PathStr, Type=None):
    if PathStr is None:
        return None
    # if Path.endswith("/"): # means this is a folder.
    #     assert not ExistsFile(Path.rstrip("/"))
    if PathStr.startswith("~"):
        PathStr = AbsPath("~") + PathStr[1:]
    
    if Type in ["Dir", "Folder"]:
        # if ExistsDir(PathStr):
        if not PathStr.endswith("/"):
            PathStr += "/"
    # windows style path
    PathStr = PathStr.replace("\\\\", "/") 
    PathStr = PathStr.replace("\\", "/")
    return PathStr
StandardizePath = ToStandardPath = ToStandardPathStr

def ToStandardFilePath(PathStr):
    return ToStandardPathStr(PathStr, Type="File")
StandardizeFilePath = ToStandardFilePath

def ToStandardDirPath(PathStr):
    return ToStandardPathStr(PathStr, Type="Dir")
StandardizeFolderPath = StandardizeDirPath = ToStandardDirPath

def MoveFile(FilePath, PathDest=None, RaiseIfNonExist=False, Overwrite=True, RaiseIfOverwrite=True):
    if not FileExists(FilePath):
        Msg = f"DLUtils.MoveFile: FilePath {FilePath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            warnings.warn(Msg)
    if DirExists(PathDest) or PathDest.endswith("/") or PathDest.endswith("\\"): # PathDest is a directory
        DirDest = StandardizeDirPath(PathDest)
        FileName = FileNameFromPath(FilePath)
        FilePathDest = DirDest + FileName
    else:
        FilePathDest = PathDest
        EnsureFileDir(FilePathDest)

    if FileExists(FilePathDest):
        if not Overwrite:
            Msg = f"DLUtils.MoveFile: FilePathDest {FilePathDest} already exist."
            if RaiseIfOverwrite:
                raise Exception(Msg)
            else:
                warnings.warn(Msg)
    # DLUtils.print("FilePath: %s\nFilePathDest: %s"%(FilePath, FilePathDest))
    shutil.move(FilePath, FilePathDest)
    # DeleteFile(FilePath) # shutil.move will delete file upon successful move.
    return True

def ListDirNameWithPattern(DirPath, DirNamePattern, End=""):
    DirNameList = []
    DirNamePatternCompiled = re.compile(DirNamePattern)
    for DirName in DLUtils.file.ListDirNames(DirPath, End=""):
        if DirNamePatternCompiled.match(DirName) is not None:
            DirNameList.append(DirName + End)
    return DirNameList

def ListFileNameWithPattern(DirPath, FileNamePattern):
    FileNameList = []
    FileNamePatternCompiled = re.compile(FileNamePattern)
    for FileName in DLUtils.file.ListFileNames(DirPath):
        if FileNamePatternCompiled.match(FileName) is not None:
            FileNameList.append(FileName)
    return FileNameList

def FileNameMatchPattern(FilePath, Pattern):
    FileName = FileNameFromPath(FilePath)
    FileNamePatternCompiled = re.compile(Pattern)
    Result = FileNamePatternCompiled.match(FileName)
    if Result is not None:
        return Result
    else:
        return False
IsFileNameMatchPattern = FileNameMatchPattern

def ListFilePathsWithPattern(DirPath, FileNamePattern):
    if not DirPath.endswith("/"):
        DirPath += "/"
    FilePathList = []
    FileNamePatternCompiled = re.compile(FileNamePattern)
    for FileName in DLUtils.ListFileNames(DirPath):
        if FileNamePatternCompiled.match(FileName) is not None:
            FilePathList.append(DirPath + FileName)
    return FilePathList
AllFilePathsWithNamePattern = AllFilePathsWithFileNamePattern = ListFilePathWithPattern = ListFilePathsWithPattern

def MoveAllFiles(DirSource, DirDest, MoveFileBeingUsed=True):
    DirSource = DLUtils.file.StandardizeDirPath(DirSource)
    DirDest = DLUtils.file.EnsureDir(DirDest)
    assert DLUtils.file.ExistsDir(DirSource)
    for FileName in ListAllFileNames(DirSource):
        FilePathSource = DirSource + FileName
        FilePathDest = DirSource + FileName
        if not MoveFileBeingUsed:
            if DLUtils.IsFileUsedByAnotherProcess(FilePathSource):
                DLUtils.print("File:(%s) is used by another process."%FilePathSource)
                continue                
        MoveFile(FilePathSource, FilePathDest)
MoveAllFile = MoveAllFiles

def DeleteFileWithFileNamePattern(DirSource, FileNamePattern=None):
    DirSource = DLUtils.CheckDirExists(DirSource)
    assert FileNamePattern is not None
    Num = 0
    FileNamePatternCompiled = re.compile(FileNamePattern)
    for FileName in DLUtils.ListFileNames(DirSource):
        if FileNamePatternCompiled.match(FileName) is not None: 
            FilePathSource = DirSource + FileName
            try:
                DeleteFile(FilePathSource, RaiseIfNonExist=True)
            except Exception:
                DLUtils.print(f"Error when deleteing file: ({FilePathSource})")
            else:
                DLUtils.print(f"Deleted file: ({FilePathSource})")
                Num += 1
    return Num
DeleteFileWithFilePattern = DeleteFileWithFileNamePattern

def DeleteFileIfExists(FilePath):
    FilePath = StandardizeFilePath(FilePath)
    if _ExistsFile(FilePath):
        return DeleteFile(FilePath)
    else:
        return False
def MoveFileWithFileNamePattern(
        DirSource,
        DirDest,
        FileNamePattern=None,
        FileSizeMax=None,
        MoveFileBeingUsed=True
    ):
    Num = 0
    DirSource = CheckDirExists(DirSource)
    DirDest = EnsureDir(DirDest)
    
    if FileSizeMax is not None:
        FileSizeMax = ParseFileSize(FileSizeMax)
    else:
        FileSizeMax = -1
    
    if FileNamePattern is None:
        for FileName in DLUtils.ListFileNames(DirSource):
            FilePathSource = DirSource + FileName
            FilePathDest = DirDest + FileName
            if FileSizeMax > 0:
                if FileSizeInBytes(FilePathSource) > FileSizeMax:
                    continue
            if not MoveFileBeingUsed:
                if IsFileUsedByAnotherProcess(FilePathSource):
                    DLUtils.print("File:(%s) is used by another process."%FilePathSource)
                    continue
            Result = DLUtils.file.MoveFile(
                FilePathSource, FilePathDest
            )
            DLUtils.print(f"Moved file: ({FilePathSource})-->({FilePathDest})")
            Num += 1
    else:
        FileNamePatternCompiled = re.compile(FileNamePattern)
        for FileName in DLUtils.ListFileNames(DirSource):
            FilePathSource = DirSource + FileName
            FilePathDest = DirDest + FileName
            if not MoveFileBeingUsed:
                if IsFileUsedByAnotherProcess(FilePathSource):
                    DLUtils.print("File:(%s) is used by another process."%FilePathSource)
                    continue
            if FileNamePatternCompiled.match(FileName) is not None: 
                FilePathSource=FilePathSource
                FilePathDest=FilePathDest
                Result = DLUtils.file.MoveFile(
                    FilePathSource, FilePathDest
                )
                DLUtils.print(f"Moved File. ({FilePathSource})-->({FilePathDest})")
                Num += 1
    return Num
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

def HaveSameContent(FilePath1, FilePath2):
    import filecmp # python standard lib
    FilePath1 = DLUtils.CheckFileExists(FilePath1)
    FilePath2 = DLUtils.CheckFileExists(FilePath2)
    return filecmp.cmp(FilePath1, FilePath2, shallow=False) # byte to byte compare

def _MoveDirIntoDirMethod(DirDest, DirSource, FileName, FilePath, FilePathRel, **Dict):
    FilePathDest = DirDest + FilePathRel
    if DLUtils.ExistsFile(FilePathDest):
        try:
            if HaveSameContent(FilePath, FilePathDest): # byte to byte compare
                try:
                    DLUtils.DeleteFile(FilePath)
                except Exception:
                    DLUtils.PrintUTF8ToStdOut("Failed to delete: (%s)"%FilePath)
            else:
                FilePathDestNew = DLUtils.RenameFileIfExists(FilePathDest)
                DLUtils.MoveFile(FilePath, FilePathDestNew)
                DLUtils.PrintUTF8ToStdOut("Moved file. (%s)-->(%s)"%(FilePath, FilePathDestNew))
        except Exception:
            pass
    else:
        DLUtils.MoveFile(FilePath, FilePathDest)
        assert not DLUtils.ExistsFile(FilePath)
        assert DLUtils.ExistsFile(FilePathDest)
        DLUtils.PrintUTF8ToStdOut("Moved file. (%s)-->(%s)"%(FilePath, FilePathDest))

def MoveDirIntoDir(DirSource, DirDest):
    DirSource = DLUtils.StandardizeDirPath(DirSource)
    DirDest = DLUtils.StandardizeDirPath(DirDest)
    import functools
    DirSource = DLUtils.CheckDirExists(DirSource)
    DirDest = DLUtils.StandardizeDirPath(DirDest)
    Result = DLUtils.DirContains(DirSource, DirDest)
    DLUtils.VisitDirAndApplyMethodOnFiles(
        DirPath=DirSource, Recur=True, Method=functools.partial(_MoveDirIntoDirMethod, DirDest=DirDest, DirSource=DirSource)
    )

def CopyFiles(FileNameList, SourceDir, DestDir):
    for FileName in FileNameList:
        CopyFile2Folder(FileName, SourceDir, DestDir)
CopyFiles2DestDir = CopyFilesInSameFolder = CopyFiles

def CopyFile2AllSubDirsUnderDestDir(FileName, SourceDir, DestDir):
    for SubDir in ListAllDirs(DestDir):
        try:
            CopyFile2Folder(FileName, SourceDir, DestDir + SubDir)
        except Exception:
            continue

def CopyFile2Dir(FilePath=None, FileName=None, DirSource=None, DirDest=None):
    if FilePath is not None:
        DirDest = EnsureDir(DirDest)
        FilePath = CheckFileExists(FilePath)
        FileName = FileNameFromFilePath(FilePath)
        shutil.copy(FilePath, DirDest + FileName)
    else:
        DirSource = CheckDirExists(DirSource)
        EnsureFileDir(DirDest + FileName)
        shutil.copy(DirSource + FileName, DirDest + FileName)
CopyFile2Folder = CopyFileTo = CopyFile2Dir

def CopyFile(FilePath, FilePathDest):
    FilePath = StandardizePath(FilePath)
    FilePathDest = StandardizePath(FilePathDest)
    shutil.copy(FilePath, FilePathDest) # overwrite if exists

def IsSameFile(FilePath1, FilePath2):
    return os.path.samefile(FilePath1, FilePath2)


def DeleteFile(FilePath, RaiseIfNonExist=False, Move2TrashBin=False, DeleteIfMove2TrashBinFail=True, Verbose=True):
    FilePath = StandardizeFilePath(FilePath)
    SigMove2TrashBinFail = False
    if Move2TrashBin:
        if DLUtils.system.IsWindows():
            try:
                assert ExistsFile(FilePath)
                send2trash(FilePath)
            except Exception:
                if Verbose:
                    DLUtils.print("Failed to delete file to trashbin (%s)"%FilePath)
                    DLUtils.print(traceback.format_exc())
                    if DeleteIfMove2TrashBinFail:
                        DLUtils.print("Trying delete only.")
                SigMove2TrashBinFail = True
    if not FileExists(FilePath):
        Msg = f"DLUtils.DeleteFile: FilePath {FilePath} does not exist."
        if RaiseIfNonExist:
            raise Exception(Msg)
        else:
            if Verbose:
                warnings.warn(Msg)
    else:
        if Move2TrashBin and not DeleteIfMove2TrashBinFail: 
            return
        else:
            os.remove(FilePath)

def DeleteFile2TrashBin(FilePath, Verbose=True, DeleteIfMove2TrashBinFail=True): 
    # assert ExistsFile(FilePath)
    # from send2trash import send2trash
    # send2trash(FilePath)
    return DeleteFile(
        FilePath,
        RaiseIfNonExist=False,
        Move2TrashBin=True,
        Verbose=Verbose,
        DeleteIfMove2TrashBinFail=DeleteIfMove2TrashBinFail
    )
DeleteFileToTrashBin = FileToTrashBin = File2TrashBin = FileToTrash = File2Trash = DeleteFile2TrashBin

def DeleteAllFilesAndSubFolders(DirPath):
    for FilePath in ListFilesPath(DirPath):
        DeleteFile(FilePath)
    for DirPath in ListDirsPath(DirPath):
        DeleteTree(DirPath)
ClearDir = EmptyDir = MakeDirEmpty = DeleteAllFilesAndSubFolders

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

# from distutils.dir_util import copy_tree
def CopyFolder(SourceDir, DestDir):
    SourceDir = CheckDirExists(SourceDir)
    CopyTree(SourceDir, DestDir)
    # shutil.copytree(SourceDir, DestDir) # Requires that DestDir not exists.
CopyDir2DestDir = CopyFolder2DestDir = CopyFolder

def FolderPathFromFilePath(FilePath):
    DirPath = os.path.dirname(os.path.realpath(FilePath))
    return DLUtils.file.StandardizeDirPath(DirPath)
# def FolderPathFromFilePath(FilePath):
#     FilePathObj = Path(FilePath)
#     ParentFolderPath = str(FilePathObj.parent.absolute())
#     return ParentFolderPath
FolderPathOfFile = FolderPathFromFilePath
DirPathOfFile = ParentFolderPath = ParentDirPath = FolderPathFromFilePath
CurrentDirPath = DirPathOfCurrentFile = FolderPathFromFilePath

def ParentFolderName(Path):
    return os.path.basename(Path)

def ParentDirPathFromDirPath(DirPath):
    DirPath = StandardizeDirPath(DirPath)
    DirPathObj = Path(DirPath)
    return str(DirPathObj.parent.absolute()) + "/"

def DirNameFromDirPath(DirPath):
    DirPath = StandardizeDirPath(DirPath)
    DirPathObj = Path(DirPath)
    return DirPathObj.name
FolderNameOfDirPath = DirNameFromDirPath

def SeperateFileNameAndDirPath(FilePath):
    FilePath = DLUtils.StandardizeFilePath(FilePath)
    DirPath = DirPathFromFilePath(FilePath)
    FileName = FileNameFromFilePath(FilePath)
    return DirPath, FileName

def DirPathFromFileName(FilePath):
    """
    return path of a directory, with same name parent directory and name(without suffix) as given file path.
    """
    FilePath = StandardizeFilePath(FilePath)
    Name, Suffix = SeparateFileNameSuffix(FilePath)
    assert Suffix is not None and Suffix not in [""]
    return StandardizeDirPath(Name)

def DirPathFromFilePath(FilePath):
    FilePathObj = Path(FilePath)
    ParentDirPath = FilePathObj.parent.absolute()
    ParentDirPath = str(ParentDirPath)
    ParentDirPath = StandardizeDirPath(ParentDirPath)
    return ParentDirPath
FolderPathOfFolder = DirPathFromFilePath

def RemoveFiles(*FilePathList):
    for FilePath in FilePathList:
        RemoveFile(FilePath)

def RemoveFile(FilePath):
    if not ExistsFile(FilePath):
        DLUtils.AddWarning("No such file: %s"%FilePath)
    else:
        os.remove(FilePath)

def RemoveFileIfExists(FilePath):
    if ExistsFile(FilePath):
        os.remove(FilePath)

def RemoveDir(DirPath):
    DirPath = StandardizeDirPath(DirPath)
    assert _ExistsDir(DirPath), DirPath
    shutil.rmtree(DirPath)
    return
DeleteDir = RemoveDir

def RemoveDirIfExists(DirPath):
    DirPath = StandardizeDirPath(DirPath)
    if _ExistsDir(DirPath):
        shutil.rmtree(DirPath)
DeleteDirIfExists = RemoveDirIfExists

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
    DirPath = StandardizeDirPath(DirPath)
    FileNameList = ListFilesName(DirPath)
    return [DirPath + FileName for FileName in FileNameList]

ListAllFilesPath = ListAllFilePaths = GetAllFilePaths = GetAllFilesPath = ListFilesPath = ListFilesPaths = ListFilePaths

def ListDirNames(DirPath, End=""):
    DirPath = CheckDirExists(DirPath)
    Names = os.listdir(DirPath)
    DirNameList = []
    for Name in Names:
        if os.path.isdir(DirPath + Name):
            DirName = Name + End
            DirNameList.append(DirName)
    return DirNameList
ListAllDirs = GetAllDirs = ListDirs = ListDirsName = ListDirNames
ListAllDirNames = ListAllDirsName = ListAllFolders = ListDirNames
ListAllFoldersName = ListAllFolderNames = ListDirNames
AllDirNames = AllDirsName = ListDirNames

def ListDirsPath(DirPath):
    if not DirPath.endswith("/"):
        DirPath += "/"
    DirNameList = ListDirsName(DirPath)
    return [DirPath + DirName for DirName in DirNameList]
ListAllDirPaths = ListAllDirsPath = ListDirsPath

def FileExists(FilePath, *List):
    if len(List) == 0: # one file
        FilePath = DLUtils.StandardizeFilePath(FilePath)
        return os.path.isfile(FilePath)
    else: # multiple file
        for _FilePath in [FilePath, *List]:
            if not FileExists(FilePath):
                return False
            else:
                return True

def AllFilesExist(*List, **Dict):
    for FilePath in list(List) + list(Dict.values()):
        if not FileExists(FilePath):
            return False
    return True

Exists = _ExistsFile = ExistsFile = FileExists

def _FolderExists(DirPath):
    # no path string style checking
    return os.path.isdir(DirPath)
_ExistsDir = _DirExists = _FolderExists

def FolderExists(DirPath):
    IsDir = os.path.isdir(DirPath)
    if IsDir:
        if not DirPath.endswith("/"):
            DirPath += "/"
    return IsDir

ExistsDir = DirExists = FolderExists
ExistsFolder = FolderExists

def CheckFileExists(FilePath):
    FilePath = ToStandardFilePath(FilePath)
    if not _ExistsFile(FilePath):
        raise Exception("%s does not exist."%FilePath)
    return FilePath

def CheckAllFilesExist(*List, **Dict):
    for FilePath in list(List) + list(Dict.values()):
        CheckFileExists(FilePath)

def CheckFolderExists(DirPath):
    DirPath = ToStandardDirPath(DirPath)
    assert _ExistsDir(DirPath), DirPath
    return DirPath

CheckDirExists = CheckFolderExists

def Path2AbsolutePath(Path):
    return os.path.abspath(Path)

def _EnsureDirectory(DirPath):
    if os.path.exists(DirPath):
        if not os.path.isdir(DirPath):
            raise Exception("%s already exists but is NOT a directory."%DirPath)
    else:
        if not DirPath.endswith("/"):
            DirPath += "/"
        os.makedirs(DirPath)
    return DirPath
_EnsureDir = _EnsureFolder = _EnsureDirectory

def EnsureDirectory(DirPath):
    #DirPathAbs = Path2AbsolutePath(DirPath)
    DirPath = StandardizeDirPath(DirPath)
    return _EnsureDirectory(DirPath)
EnsureDirPath = EnsureDirExists = EnsureDir = EnsureFolder = EnsureDirectory

def EnsureFileDirectory(FilePath):
    assert not FilePath.endswith("/"), FilePath
    FilePath = DLUtils.StandardizeFilePath(FilePath)
    DirPath = DLUtils.DirPathOfFile(FilePath)
    EnsureDir(DirPath)
    return FilePath
EnsureFileDir = EnsureFileDirectory

def _EnsureFileDirectory(FilePath):
    DirPath = DirPathOfFile(FilePath)
    _EnsureDir(DirPath)
    return FilePath
_EnsureFileDir = _EnsureFileDirectory

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

def RemoveFileNameSuffix(FilePath, RaiseIfNoSuffix=False):
    Name, Suffix = SeparateFileNameSuffix(FilePath)
    if RaiseIfNoSuffix:
        if Suffix is None or Suffix in [""]:
            raise Exception()
    return Name
RemoveFileSuffix = RemoveFilePathSuffix = RemoveFileNameSuffix

def AppendSuffix2FileName(FileName, Suffix):
    Name, _Suffix = SeparateFileNameSuffix(FileName)
    if _Suffix in [""] or Suffix is None:
        return Name + Suffix
    else:
        return Name + Suffix + "." + _Suffix
AppendSuffixOnFileName = AppendSuffix2FileName

def AppendSuffixOnCurrentFileName(__File__, Suffix):
    FilePath = StandardizeFilePath(__File__)
    return AppendSuffixOnFileName(FilePath, Suffix)
AppendOnCurrentFileName = AppendSuffixOnCurrentFileName

def AppendOnFileNameAndChangeSuffix(FilePath, Append, Suffix):
    FilePath = StandardizeFilePath(FilePath)
    Name, _Suffix = SeparateFileNameSuffix(FilePath)
    Suffix = Suffix.lstrip(".")
    return Name + Append + "." + Suffix
AppendOnCurrentFileNameAndChangeSuffix = AppendOnFileNameAndChangeSuffix

def SeparateDirPathAndFileName(FilePath):
    FileName = os.path.basename(FilePath)
    DirPath = DirPathFromFilePath(FilePath)
    return FileName, DirPath

def GetFilePathNoSuffix(FilePath):
    Name, Suffix = SeparateFileNameSuffix(FilePath)
    return Name

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
ParseFileNameSuffix = SeparateFileNameSuffix

def ChangeFilePathSuffix(FileName, Suffix):
    Name, _Suffix = SeparateFileNameSuffix(FileName)
    Suffix = Suffix.lstrip(".")
    return Name + "." + Suffix
ChangeFileNameSuffix = ChangeFilePathSuffix

def ChangeFileDirPath(FilePath, DirPath):
    DirPath = StandardizeDirPath(DirPath)
    return DirPath + FileNameFromPath(FilePath)
ChangeCurrentFileNameSuffix = ChangeNameSuffix = ChangeFileNameSuffix

def AddSuffixToFileWithFormat(FilePath, Suffix):
    _FilePath, Format = ParseFileNameSuffix(FilePath)
    return _FilePath + Suffix + "." + Format

def RenameFileInDir(DirPath, FileName, FileNameNew):
    DirPath = CheckDir(DirPath)
    assert ExistsFile(DirPath + FileName)
    os.rename(DirPath + FileName, DirPath + FileNameNew)

def RenameFile(FilePath, FilePathNew):
    FilePath = CheckFileExists(FilePath)
    FilePathNew = StandardizeFilePath(FilePathNew)
    os.rename(FilePath, FilePathNew)
ChangeFileName = RenameFile

def RenameFileIfExists(FilePath, RenameExistingFile=False):
    if FilePath.endswith("/"):
        raise Exception()
    FileName, Suffix = ParseFileNameSuffix(FilePath)
    Sig = True
    MatchResult = re.match(r"^(.*)-(\d+)$", FileName)
    if MatchResult is None:
        if ExistsPath(FilePath):
            if RenameExistingFile:
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

def IsFileUsedByAnotherProcess(FilePath):
    # windows only
    FilePath = DLUtils.CheckFileExists(FilePath)
    try:
        os.rename(FilePath, FilePath)
        return False
    except OSError:    # file is in use
        return True

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
    DirPath = StandardizeDirPath(DirPath)
    _DirPath = DirPath.rstrip("/")
    if _ExistsDir(_DirPath + "-0/"):
        Index = 1
        while True:
            DirPathTry = _DirPath + "-%d/"%Index
            if _ExistsDir(DirPathTry):
                Index += 1
                continue
            else:
                return DirPathTry
    else:
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
    DLUtils.NpArray2DToTextFile(Data, SavePath=SavePath)

def GetRelativePath(path_rel=None, path_start=None, path_main=None):
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

def File2Str(FilePath):
    return Path(FilePath).read_text()

def FileSizeInBytes(FilePath): 
    return os.path.getsize(FilePath)
FileSize = FileSizeByte = FileSizeInBytes

KB = 1024.0
KBInt = 1024
MB = 1024.0 * 1024.0
MBInt = 1024 * 1024
GB = 1024.0 * 1024.0 * 1024.0
GBInt = 1024 * 1024 * 1024

def ParseFileSize(FileSize):
    if isinstance(FileSize, int):
        return FileSize
    elif isinstance(FileSize, str):
        Result = re.match(r"(.*)GB|(.*)G|(.*)gb|(.*)g", FileSize)
        if Result is not None:
            return round(float(Result.group(1))) * GBInt
        Result = re.match(r"(.*)MB|(.*)M|(.*)MiB|(.*)mb", FileSize)
        if Result is not None:
            return round(float(Result.group(1))) * MBInt
        raise Exception()
    else:
        raise Exception()

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
        return "%.2f KB"%(SizeKB)
    if SizeMB > Base:
        SizeGB = "%.2f GB"%(SizeB * 1.0 / GB)
    else:
        return "%.2f MB"%(SizeMB)
    return SizeGB

def FileSizeStr(FilePath):
    Bytes = FileSizeInBytes(FilePath)
    return Size2Str(Bytes)

def File2ObjPickle(FilePath):
    FilePath = DLUtils.CheckFileExists(FilePath)
    with open(FilePath, 'rb') as f:
        Obj = pickle.load(f, encoding='bytes')
    return Obj
File2Obj = File2ObjPickle
BinaryFile2Obj = File2ObjPickle
def Obj2FilePickle(Obj, FilePath):
    FilePath = DLUtils.EnsureFileDir(FilePath)
    with open(FilePath, "wb") as f:
        pickle.dump(Obj, f)
Obj2File = Obj2FilePickle
Obj2BinaryFile = Obj2FilePickle
JsonObj2DataFile = Obj2File

def Append2TextFile(Str, FilePath):
    with open(FilePath, 'a') as f:
        f.write(Str)

def Str2TextFile(Str, FilePath):
    FilePath = StandardizeFilePath(FilePath)
    with open(FilePath, 'w') as f:
        f.write(Str)

def TextFile2Str(FilePath):
    FilePath = StandardizeFilePath(FilePath)
    with open(FilePath, "r") as f:
        Str = f.read()
    return Str

def Str2MD5(Str):
    Bytes = Str.encode('utf-8')
    return hashlib.md5(Bytes).hexdigest()

def ToMD5(Obj):
    return Str2MD5(str(Obj))


def HaveSameContent(FilePath1, FilePath2):
    return filecmp.cmp(FilePath1, FilePath2, shallow=False)

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

def VisitDirAndApplyMethodOnFiles(DirPath=None, Method=None, Recur=True, DirPathRel="", **Dict):
    DirPath = CheckDirPath(DirPath)
    if Method is None:
        Method = DLUtils.EmptyFunction
        DLUtils.warn('Method is None.')
    FileNameList = ListAllFileNames(DirPath)
    for FileName in FileNameList:
        Method(FilePath=DirPath + FileName, FileName=FileName, DirPath=DirPath, FilePathRel=DirPathRel + FileName, **Dict)
    if Recur:
        DirNameList = DLUtils.ListAllDirNames(DirPath)
        for DirName in DirNameList:
            VisitDirAndApplyMethodOnFiles(DirPath + DirName + "/", Method, Recur, DirPathRel = DirPathRel + DirName + "/", **Dict)

def DirContains(DirSource, DirDest, Recur=True, DirPathRel="", **Dict):
    FileNameList = ListAllFileNames(DirSource)
    for FileName in FileNameList:
        FilePath = DirSource + FileName
        FilePathDest = DirDest + FileName
        if not HaveSameContent(FilePath, FilePathDest):
            return False
    if Recur:
        DirNameList = DLUtils.ListAllDirNames(DirSource)
        for DirName in DirNameList:
            Result = DirContains(DirSource + DirName + "/", DirDest + DirName + "/", Recur, DirPathRel = DirPathRel + DirName + "/", **Dict)
            if Result is False:
                return False
    return True

def FileNumInDir(DirPath, Recur=True):
    FileNameList = ListAllFileNames(DirPath)
    Num = len(FileNameList)
    if Recur:
        DirNameList = DLUtils.ListAllDirNames(DirPath)
        for DirName in DirNameList:
            _Num = DirContains(DirPath + DirName + "/", Recur=Recur)
            Num += _Num
    return Num

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

def GetTreeStruct(DirPath):
    DirPath = CheckDirExists(DirPath)
    Dict = {
        "DirList": [],
        "FileList": []
    }
    for DirName in ListAllDirs(DirPath):
        Dict["DirList"][DirName.rstrip("/")] = GetTreeStruct(
            DirPath + DirName
        )
    for FileName in ListAllFileNames(DirPath):
        Dict["FileList"].append(FileName)
    return Dict
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

def ExtractGzFile(FilePath, DestFilePath=None):
    """
        turn a compressed .gz file to uncompressed file
    """
    FilePath = DLUtils.CheckFileExists(FilePath)
    if DestFilePath is None:
        Name, Suffix = DLUtils.SeparateFileNameSuffix(FilePath)
        if Suffix is None:
            DestFilePath = Name + "-from-gz"
        else:
            DestFilePath = Name
    else:
        DestFilePath = DLUtils.EnsureFileDir(DestFilePath)

    with gzip.open(FilePath, 'rb') as FileIn:
        with open(DestFilePath, 'wb') as FileOut:
            shutil.copyfileobj(FileIn, FileOut)
    return DestFilePath

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

def ExtractTarFile(FolderPath, ExtractFolderPath):
    # Input: Zip File
    # Output: Extracted Folder

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

def ExtractZipFile(ZipFilePath, DestDirPath):
    """
        Extract .zip file to a folder.
    """
    ZipFilePath = DLUtils.CheckFileExists(ZipFilePath)
    DestDirPath = DLUtils.EnsureDir(DestDirPath)
    with zipfile.ZipFile(ZipFilePath, 'r') as zip_ref:
        zip_ref.extractall(DestDirPath)
    return DestDirPath

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

def JsonDict2JsonFile(JsonDict, FilePath, Mode=None):
    if Mode in ["Simple"]:
        JsonStr = json.dumps(JsonDict, indent=4)
    else:
        JsonStr = JsonDict2Str(JsonDict)
    Str2TextFile(JsonStr, FilePath)
JsonDict2File = JsonDict2JsonFile


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
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import psutil # pip install psutil
else:
    psutil = DLUtils.GetLazyPsUtil()

def IsFileUsedByOtherProcess(FilePath):
    # assert IsPsUtilImported
    for proc in psutil.process_iter():
        try:
            # this returns the list of opened files by the current process
            FileList = proc.open_files()
            if FileList:
                print(proc.pid,proc.name)
                for FileName in FileList:
                    print("\t",FileName.path)
        except psutil.NoSuchProcess as err:
            # This catches a race condition where a process ends before we can examine its files
            pass
        except Exception:
            traceback.print_exc()
            pass

def FileCreateTime(FilePath):
    FilePath = DLUtils.file.CheckFileExists(FilePath)
    if DLUtils.system.IsWin():
        return os.path.getctime(FilePath)
    else:
        raise NotImplementedError()

def FolderLastModifiedTimeStamp(DirPath):
    # following operation will update a folder's last modified time:
        # delete child file or folder.
        # create(include by paste) child file or folder.
    DirPath = DLUtils.CheckDirExists(DirPath)
    LastModifiedTimeStamp = os.path.getmtime(DirPath)
    return LastModifiedTimeStamp
DirLastModifiedTimeStamp = FolderLastModifiedTimeStamp

def FolderLastModifiedTime(DirPath, ReturnType="LocalTimeStr"):
    # following operation will update a folder's last modified time:
        # delete child file or folder.
        # create(include by paste) child file or folder.
    DirPath = DLUtils.CheckDirExists(DirPath)
    LastModifiedTimeStamp = os.path.getmtime(DirPath)
    return DLUtils.time.TimeStampToType(LastModifiedTimeStamp, ReturnType)
DirLastModifiedTime = FolderLastModifiedTime

def FileLastModifiedTimeStamp(FilePath):
    # following operation will update a folder's last modified time:
        # delete child file or folder.
        # create(include by paste) child file or folder.
    FilePath = DLUtils.CheckFileExists(FilePath)
    TimeStamp = os.path.getmtime(FilePath)
    return TimeStamp

def FileLastModifiedTime(FilePath, ReturnType="LocalTimeStr"):
    LastModifiedTimeStamp = FileLastModifiedTime(FilePath)
    return DLUtils.time.TimeStampToType(LastModifiedTimeStamp, ReturnType)

def FileCreatedTime(FilePath):
    FilePath = DLUtils.CheckFileExists(FilePath)

def LastModifiedTimeStamp(Path):
    if DLUtils.file._ExistsFile(Path):
        # Path = DLUtils.file.StandardizeFilePath(Path)
        return FileLastModifiedTimeStamp(Path)
    elif DLUtils.file._ExistsDir(Path):
        # Path = DLUtils.file.StandardizeDirPath(Path)
        return FolderLastModifiedTimeStamp(Path)
    else:
        raise Exception("non-existent path: %s"%Path)

def LastModifiedTime(Path, ReturnType="TimeStamp"):
    if DLUtils.file._ExistsFile(Path):
        # Path = DLUtils.file.StandardizeFilePath(Path)
        return FileLastModifiedTime(Path, ReturnType)
    elif DLUtils.file._ExistsDir(Path):
        # Path = DLUtils.file.StandardizeDirPath(Path)
        return FolderLastModifiedTime(Path, ReturnType)
    else:
        raise Exception("non-existent path: %s"%Path)
