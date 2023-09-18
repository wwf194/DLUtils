import os
import sys
import traceback

# sys.path.append(r"S:/0 Science/软件 项目")
# sys.path.append(r"S:\0 Science\Project-Code")
# sys.path.append(r"D:\Project-Code")
# import DLUtils
sys.path.append(os.path.dirname(__file__))
from import_lib_path import DLUtilsPath
EnvDict = {
    "Path": {}
}
def WriteUTF8ToStdOut(Str, Indent=None):
    if Indent is not None:
        assert isinstance(Indent, int)
        StrList = Str.split("\n")
        for Index, Str in enumerate(StrList):
            WriteUTF8ToStdOut("    " * Indent + Str + "\n")
            # if(Index < len(StrList) - 1):
            #     WriteUTF8ToStdOut("    " * Indent + Str + "\n")
            # else:
            #     WriteUTF8ToStdOut("    " * Indent + Str)
        return
    Bytes = Str.encode("utf-8")
    try:
        sys.__stdout__.buffer.write(Bytes)
    except Exception:
        pass # broken pipe

ImportPath = "NULL"
def Import():
    global ImportPath
    global DLUtils, HasImported
    WriteUTF8ToStdOut("Trying import DLUtils. ")
    Verbose = True
    Sig = False
    for Path in DLUtilsPath:
        if os.path.exists(Path):
            try:
                sys.path.append(Path)
                import DLUtils
                EnvDict["Log"] = "Imported DLUtils from: %s\n"%Path
                ImportPath = Path
                Sig = True
                break
            except Exception:
                # traceback.print_exc()
                # Log = traceback.format_exc()
                t, v, tb = sys.exc_info()
                if isinstance(v, ModuleNotFoundError) \
                    and v.__str__() == "No module named 'DLUtils'.":
                    EnvDict["Log"] = "Cannot find module.\n%s"%traceback.format_exc()
                else:
                    EnvDict["Path"][Path] = traceback.format_exc()
                sys.path.pop()
                continue

    if not Sig:
        WriteUTF8ToStdOut("ERROR: Cannot import DLUtils.\n")
        if Verbose:
            for Path, Error in EnvDict["Path"].items():
                AbsPath = os.path.abspath(Path)
                if Path != AbsPath:
                    WriteUTF8ToStdOut("Path:(%s). AbsPath:(%s) Error:\n"%(Path, AbsPath))
                else:
                    WriteUTF8ToStdOut("Path:(%s). Error:\n"%Path)
                if "No module named 'DLUtils'" in Error:
                    WriteUTF8ToStdOut("ModuleNotFoundError", Indent=1)
                    # WriteUTF8ToStdOut(Error, Indent=1)
                    WriteUTF8ToStdOut("\n")
                else:
                    WriteUTF8ToStdOut(Error, Indent=1)
                    if not Error.endswith("\n"):
                        WriteUTF8ToStdOut("\n")
        sys.exit(0)
    else:
        WriteUTF8ToStdOut("Imported DLUtils from %s\n"%ImportPath)
        HasImported = True
    
    return HasImported
HasImported = False
Import()