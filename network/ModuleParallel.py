import torch
import numpy as np

import DLUtils
class ModuleParallel(DLUtils.module.AbstractModuleGroup):
    def Receive(self, In):
        OutList = []
        for SubModule in self.ModuleList:
            Out = SubModule(In)
            OutList.append(Out)
        return OutList
    def Init(self, IsSuper=False, IsRoot=True):
        self.ModuleList = list(self.SubModules.values())
        Param = self.Param
        return super().Init(IsSuper, IsRoot)