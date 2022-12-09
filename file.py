from ast import Is
import os
import re
import pandas as pd
import shutil # sh_utils
import DLUtils
#from DLUtils.attr import *
import warnings

from DLUtils.utils._param import JsonFile2Param

def ParentFolderPath(FilePath):
    return os.path.dirname(os.path.realpath(FilePath)) + "/"
FolderPathOfFile = ParentFolderPath

def RemoveFiles(FilesPath):
    for FilePath in FilesPath:
        RemoveFile(FilePath)

def RemoveFile(FilePath):
    if not ExistsFile(FilePath):
        DLUtils.AddWarning("No such file: %s"%FilePath)
    else:
        os.remove(FilePath)

def RemoveAllFilesUnderDir(DirPath, verbose=True):
    assert ExistsDir(DirPath)
    for FileName in GetAllFiles(DirPath):
        #FilePath = os.path.join(DirPath, FileName)
        FilePath = DirPath + FileName
        os.remove(FilePath)
        DLUtils.AddLog("utils_pytorch: removed file: %s"%FilePath)

def RemoveAllFilesAndDirsUnderDir(DirPath, verbose=True):
    assert ExistsDir(DirPath)
    Files, Dirs= GetAllFilesAndDirs(DirPath)
    for FileName in Files:
        FilePath = os.path.join(DirPath, FileName)
        os.remove(FilePath)
        DLUtils.AddLog("DLUtils: removed file: %s"%FilePath)
    for DirName in Dirs:
        DirPath = os.path.join(DirPath, DirName)
        #os.removedirs(DirPath) # Cannot delete subfolders
        import shutil
        shutil.rmtree(DirPath)
        DLUtils.AddLog("DLUtils: removed directory: %s"%DirPath)

def IsDir(DirPath):
    return os.path.isdir(DirPath)

def IsFile(FilePath):
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
                DLUtils.AddLog("DLUtils: removed file: %s"%FilePath)

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

def ListFiles(DirPath):
    assert os.path.exists(DirPath), "Non-existing DirPath: %s"%DirPath
    assert os.path.isdir(DirPath), "Not a Dir: %s"%DirPath
    Items = os.listdir(DirPath)
    Files = []
    for Item in Items:
        if os.path.isfile(os.path.join(DirPath, Item)):
            Files.append(Item)
    return Files

ListAllFiles = GetAllFiles = ListFiles

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
ListAllDirs = GetAllDirs = ListDirs
ListAllFolders = ListDirs

def FileExists(FilePath):
    return os.path.isfile(FilePath)
ExistsFile = FileExists

def DirExists(DirPath):
    return os.path.isdir(DirPath)
ExistsDir = DirExists

def CheckFileExists(FilePath):
    if not DLUtils.ExistsFile(FilePath):
        raise Exception("%s does not exist."%FilePath)

def Path2AbsolutePath(Path):
    return os.path.abspath(Path)

def EnsureDirectory(DirPath):
    #DirPathAbs = Path2AbsolutePath(DirPath)
    if os.path.exists(DirPath):
        if not os.path.isdir(DirPath):
            raise Exception("%s Already Exists but Is NOT a Directory."%DirPath)
    else:
        if not DirPath.endswith("/"):
            DirPath += "/"
        os.makedirs(DirPath)

EnsureDir = EnsureDirectory
EnsureFolder = EnsureDirectory

def EnsureFileDirectory(FilePath):
    assert not FilePath.endswith("/"), FilePath
    FilePath = Path2AbsolutePath(FilePath)
    FileDir = os.path.dirname(FilePath)
    EnsureDir(FileDir)

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
                    Md5_source = GetMd5(SourceDir + subpath + item)
                    Md5_target = GetMd5(DestDir + subpath + item)
                    if Md5_target==Md5_source: # same file
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

def ParseFileNameSuffix(FilePath):
    if FilePath.endswith("/"):
        raise Exception()
    MatchResult = re.match(r"(.*)\.(.*)", FilePath)
    if MatchResult is None:
        return FilePath, ""
    else:
        return MatchResult.group(1), MatchResult.group(2)

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

def RenameDir(DirOld, DirNew):
    if not ExistsDir(DirOld):
        DLUtils.AddWarning("RenameDir: Dir %s does not exist."%DirOld)
        return
    assert not ExistsFile(DirNew.rstrip("/"))
    os.rename(DirOld, DirNew)

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
    with open(FilePath, "w") as file:
        file.write(Str)

def Table2TextFileDict(Dict, SavePath):
    DLUtils.Str2File(pd.DataFrame(Dict).to_string(), SavePath)   
Table2TextFile = Table2TextFileDict

def Table2TextFileColumns(*Columns, **kw):
    # ColNum = len(Columns)
    # Str = " ".join(kw["Names"])
    # Str += "\n"
    # for RowIndex in range(len(Columns[0])):
    #     for ColIndex in range(ColNum):
    #         Str += str(Columns[ColIndex][RowIndex])
    #         Str += " "
    #     Str += "\n"
    # DLUtils.Str2File(Str, kw["SavePath"])
    Names = kw["Names"]
    Dict = {}
    for Index, Column in enumerate(Columns):
        Dict[Names[Index]] = Column
    Table2TextFileDict(Dict, kw["SavePath"])

def LoadParamFromFile(Args, **kw):
    if isinstance(Args, dict):
        _LoadParamFromFile(DLUtils.json.JsonObj2PyObj(Args), **kw)
    elif isinstance(Args, list) or DLUtils.IsListLikePyObj(Args):
        for Arg in Args:
            _LoadParamFromFile(Arg, **kw)
    elif DLUtils.IsDictLikePyObj(Args):
        _LoadParamFromFile(Args, **kw)
    else:
        raise Exception()

def _LoadParamFromFile(Args, **kw):
    FilePathList = DLUtils.ToList(Args.FilePath)
    MountPathList = DLUtils.ToList(Args.MountPath)
    for MountPath, FilePath in zip(MountPathList, FilePathList):
        Obj = DLUtils.json.JsonFile2PyObj(FilePath)
        if not isinstance(Obj, list):
            EnsureAttrs(Args, "SetResolveBase", default=True)
            if Args.SetResolveBase:
                setattr(Obj, "__IsResolveBase__", True)
        DLUtils.MountObj(MountPath, Obj, **kw)
    return

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

# def select_file(name, candidate_files, default_file=None, match_prefix='', match_suffix='.py', file_type='', raise_no_match_error=True):
#     use_default_file = False
#     perfect_match = False
#     if name is None:
#         use_default_file = True
#     else:
#         matched_count = 0
#         matched_files = []
#         perfect_match_name = None
#         if match_prefix + name + match_suffix in candidate_files: # perfect match. return this file directly
#             perfect_match_name = match_prefix + name + match_suffix
#             perfect_match = True
#             matched_files.append(perfect_match_name)
#             matched_count += 1
#         for file_name in candidate_files:
#             if name in file_name:
#                 if file_name!=perfect_match_name:
#                     matched_files.append(file_name)
#                     matched_count += 1
#         #print(matched_files)
#         if matched_count==1: # only one matched file
#             return matched_files[0]
#         elif matched_count>1: # multiple files matched
#             warning = 'multiple %s files matched: '%file_type
#             for file_name in matched_files:
#                 warning += file_name
#                 warning += ' '
#             warning += '\n'
#             if perfect_match:
#                 warning += 'Using perfectly matched file: %s'%matched_files[0]
#             else:
#                 warning += 'Using first matched file: %s'%matched_files[0]
#             warnings.warn(warning)
#             return matched_files[0]
#         else:
#             warnings.warn('No file matched name: %s. Trying using default %s file.'%(str(name), file_type))
#             use_default_file = True
#     if use_default_file:
#         if default_file is not None:
#             if default_file in candidate_files:
#                 print('Using default %s file: %s'%(str(file_type), default_file))
#                 return default_file
#             else:
#                 sig = True
#                 for candidate_file in candidate_files:
#                     if default_file in candidate_file:
#                         print('Using default %s file: %s'%(str(file_type), candidate_file))
#                         sig = False
#                         return candidate_file
#                 if not sig:
#                     if raise_no_match_error:
#                         raise Exception('Did not find default %s file: %s'%(file_type, str(default_file)))
#                     else:
#                         return None
#         else:
#             if raise_no_match_error:
#                 raise Exception('Plan to use default %s file. But default %s file is not given.'%(file_type, file_type))
#             else:
#                 return None
#     else:
#         return None

def ToAbsPath(Path):
    if "~" in Path:
        Path = os.path.expanduser(Path)
    AbsPath = os.path.abspath(Path)
    if IsDir(AbsPath) and not AbsPath.endswith("/"):
        AbsPath += "/"
    return AbsPath

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

def VisitDirAndApplyMethodOnFiles(DirPath=None, Method=None, Recur=False, **kw):
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
            VisitDirAndApplyMethodOnFiles(DirPath + DirName + "/", Method, Recur, **kw)

def VisitDirAndApplyMethodOnDirs(DirPath=None, Method=None, Recur=False, **kw):
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
            VisitDirAndApplyMethodOnFiles(DirPath + DirName + "/", Method, Recur, **kw)

def CopyFiles2DestDir(FileNameList, SourceDir, DestDir):
    for FileName in FileNameList:
        CopyFile2DestDir(FileName, SourceDir, DestDir)

def CopyFile2AllSubDirsUnderDestDir(FileName, SourceDir, DestDir):
    for SubDir in ListAllDirs(DestDir):
        try:
            CopyFile2DestDir(FileName, SourceDir, DestDir + SubDir)
        except Exception:
            continue

def CopyFile2DestDir(FileName, SourceDir, DestDir):
    EnsureFileDir(DestDir + FileName)
    shutil.copy(SourceDir + FileName, DestDir + FileName)

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
            CopyFile2DestDir(Name, SourceDir, DestDir)
        else:
            raise Exception()
#from distutils.dir_util import copy_tree
def CopyFolder2DestDir(SourceDir, DestDir):
    assert IsDir(SourceDir)
    if not DestDir.endswith("/"):
        DestDir += "/"
    _CopyTree(SourceDir, DestDir)
    #shutil.copytree(SourceDir, DestDir) # Requires that DestDir not exists.
CopyDir2DestDir = CopyFolder2DestDir

def SplitPaths(Paths):
    PathsSplit = []
    for Path in Paths:
        PathsSplit.append(SplitPath(Path))
    return PathsSplit
def SplitPath(Path):
    return Path

def _CopyTree(SourceDir, DestDir, **kw):
    kw.setdefault("SubPath", "")
    Exceptions = kw.setdefault("Exceptions", []) # to be implemented: allowing excetionpaths                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    Files, Dirs = ListAllFilesAndDirs(SourceDir)
    for File in Files:
        # if File in Exceptions[0]:
        #     continue
        CopyFile2DestDir(File, SourceDir, DestDir)
    for Dir in Dirs:
        EnsureDir(DestDir + Dir)
        _CopyTree(SourceDir + Dir, DestDir + Dir, **kw)


def Data2TextFile(data, Name=None, FilePath=None):
    if FilePath is None:
        FilePath = DLUtils.GetSavePathFromName(Name, Suffix=".txt")
    DLUtils.Str2File(str(data), FilePath)

from .utils.json import PyObj2DataFile, DataFile2PyObj, PyObj2JsonFile, \
    JsonFile2PyObj, JsonFile2JsonDict, JsonObj2JsonFile, DataFile2JsonObj

from .utils._param import JsonDict2Str

def FolderDesciprtion(FolderPath):
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
                FileName,
                DLUtils.param({
                    "MD5": File2MD5(FolderPath + FileName),
                    "Volume": FileSizeStr(FilePath)
                })
            )
        Folders = ParamNode.Folders
        for FolderName in ListAllDirs(FolderPath):
            
            Folders.setattr(
                FolderName, DLUtils.param({})
            )
            NodeList.append(
                DLUtils.param({
                    "FolderPath": FolderPath + FolderName, # already ends with /
                    "ParamNode": Folders.getattr(FolderName)
                })
            )
    return Tree

def CheckIntegrity(FolderPath, Config):
    FolderPath = ToAbsPath(FolderPath)
    FolderNodeList = []
    ConfigNodeList = []
    FolderNodeList.append(FolderPath)   
    ConfigNodeList.append(Config)

    while len(ConfigNodeList) > 0:
        ConfigNode = ConfigNodeList.pop()
        FolderPath = FolderNodeList.pop()
        FilesNode = ConfigNode.Files
        Files = ListAllFiles(FolderPath)
        for FileName, FileInfo in FilesNode.items():
            SubFilePath = FolderPath + FileName
            if not FileExists(SubFilePath):
                return False
            if not FileInfo.MD5 == File2MD5(SubFilePath):
                return False
        FoldersNode = ConfigNode.Folders
        for FolderName, FolderInfo in FoldersNode.items():
            SubFolderPath = FolderPath + FolderName
            if not ExistsDir(SubFolderPath):
                return False
            ConfigNodeList.append(FolderInfo)
            FolderNodeList.append(SubFolderPath)
    return True

def JsonDict2JsonFile(JsonDict, FilePath):
    JsonStr = JsonDict2Str(JsonDict)
    Str2TextFile(JsonStr, FilePath)
JsonDict2File = JsonDict2JsonFile