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
Verbose = True
Sig = False
print("trying import DLUtils.")
ErrorInfo = ""
for Path in DLUtilsPath:
    if os.path.exists(Path):
        try:
            # print("Trying Path: %s"%Path)
            sys.path.append(Path)
            import DLUtils
            print("imported DLUtils from: %s."%DLUtils.StandardizeDirPath(Path))
            Sig = True
            break
        except Exception:
            ErrorInfo = traceback.format_exc()
            if Verbose:
                print(ErrorInfo)
            continue
if not Sig:
    print("ERROR: Cannot import DLUtils.")
    print(ErrorInfo)
    sys.exit(-1)

import DLUtils