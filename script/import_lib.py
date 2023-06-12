import os
import sys
import traceback
DLUtilsPath = [
    "S:/0 Science/项目 软件",
    "A:/0 Science/项目 软件",
    "C:/Users/Tim Wang/script",
    "..",
    "../..",
    "~/Project",
    "/home/wwf/Project",
    "root/"
]
EnvDict = {}
def WriteUTF8(Str):
    sys.stdout.buffer.write(Str.encode("utf-8"))
WriteUTF8("trying import DLUtils. ")
Verbose = True
Sig = False
for Path in DLUtilsPath:
    Path = os.path.abspath(Path)
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
import DLUtils