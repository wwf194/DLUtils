import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()

class ModuleParallel(DLUtils.module.AbstractModuleGroup):
    def Receive(self, In):
        OutList = []
        for SubModule in self.ModuleList:
            Out = SubModule(In)
            OutList.append(Out)
        return OutList
    def Build(self, IsSuper=False, IsRoot=True):
        self.ModuleList = list(self._SubModules.values())
        return super().Init(IsSuper, IsRoot)