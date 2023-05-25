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

Sig = False
for Path in DLUtilsPath:
    if os.path.exists(Path):
        try:
            print("Trying Path: %s"%Path)
            sys.path.append(Path)
            import DLUtils
            print("Successfully imported DLUtils.")
            Sig = True
            break
        except Exception:
            traceback.print_exc()
            continue
if not Sig:
    print("ERROR: Cannot import DLUtils.")
    sys.exit(-1)

import DLUtils