
import DLUtils
import importlib

def IsLegalPyName(name):
    if name=="":
        return False
    if name[0].isalpha() or name[0] == '_':
        for i in name[1:]:
            if not (i.isalnum() or i == '_'):
                return False
        else:
            return True
    else:
        return False

def CheckIsLegalPyName(name):
    if not IsLegalPyName(name):
        raise Exception("%s is not a legal python name."%name)

def ParseClass(ClassPath):
    try:
        Module = DLUtils.ImportModule(ClassPath)
        if hasattr(Module, "__MainClass__"):
            return Module.__MainClass__
        else:
            return Module
    except Exception:
        Class = eval(ClassPath)
        return Class

def ImportModule(ModulePath):
    try:
        return importlib.import_module(ModulePath)
    except Exception:
        return eval(ModulePath)
