def TargetDir_module(path):
    path = path.lstrip('./')
    path = path.lstrip('/')
    if not path.endswith('/'):
        path += '/'
    path =  path.replace('/','.')
    return path

def GetAllMethodsOfModule(ModulePath):
    from inspect import getmembers, isfunction
    Module = ImportModule(ModulePath)
    return getmembers(Module, isfunction)

ListAllMethodsOfModule = GetAllMethodsOfModule