import os
import sys
import traceback
DLUtilsPath = [
    "S:/0 Science/项目 软件",
    "A:/0 Science/项目 软件",
    "C:/Users/Tim Wang/script",
    "..",
    "../.."
]
EnvDict = {}
print("trying import DLUtils.")
Verbose = True
Sig = False
for Path in DLUtilsPath:
    if os.path.exists(Path):
        try:
            sys.path.append(Path)
            import DLUtils
            EnvDict["Log"] = "imported DLUtils from: %s"%Path
            print(EnvDict["Log"])
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
    print("ERROR: Cannot import DLUtils.")
    if Verbose:
        print(EnvDict["Log"])
    sys.exit(-1)

import DLUtils