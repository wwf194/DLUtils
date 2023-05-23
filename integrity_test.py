import os
import sys

DLUtilsPath = [
    "../"
]

for Path in DLUtilsPath:
    if os.path.exists(Path):
        try:
            # print("Trying Path: %s"%Path)
            sys.path.append(Path)
            import DLUtils
            print("Successfully imported DLUtils.")
            break
        except Exception:
            continue

    print("library import: failed.")
    sys.exit(-1)

print("library import: succeeded.")
sys.exit(0)