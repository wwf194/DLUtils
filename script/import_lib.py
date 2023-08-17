import os
import sys
import traceback
DLUtilsPath = [
    "S:/0 Science/软件 项目",
    "A:/0 Science/软件 项目",
    "C:/Users/Tim Wang/script",
    "..",
    "../.."
]
EnvDict = {}

def WriteUTF8(Str):
    Bytes = Str.encode("utf-8")
    try:
        sys.__stdout__.buffer.write(Bytes)
    except Exception:
        pass # broken pipe
    
def Import():
    global DLUtils, HasImported
    WriteUTF8("trying import DLUtils. ")
    Verbose = True
    Sig = False
    for Path in DLUtilsPath:
        if os.path.exists(Path):
            try:
                sys.path.append(Path)
                import DLUtils
                EnvDict["Log"] = "imported DLUtils from: %s"%Path
                DLUtils.print(EnvDict["Log"])
                Sig = True
                break
            except Exception:
                # traceback.print_exc()
                # Log = traceback.format_exc()
                t, v, tb = sys.exc_info()
                if isinstance(v, ModuleNotFoundError) \
                    and v.__str__() == "No module named 'DLUtils'":
                    EnvDict["Log"] = "Cannot find module."
                else:
                    EnvDict["Log"] = traceback.format_exc()
                # Log = traceback.format_exception(t, v, tb)
                # Log = sys.exc_info()
                sys.path.pop()
                continue

    if not Sig:
        WriteUTF8("ERROR: Cannot import DLUtils.")
        if Verbose:
            WriteUTF8(EnvDict["Log"])
        sys.exit(0)
    else:
        HasImported = True
    
    return HasImported
HasImported = False
Import()