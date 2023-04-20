
class ModuleSeries(DLUtils.module.AbstractModuleGroup):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("OutName", "OutputName"): "Out.Name"
    })
    def LoadParam(self, Param):
        super().LoadParam(Param)
        Param = self.Param
        self.ModuleList = []
        for Name, SubModuleParam in Param.SubModules.items():
            self.ModuleList.append(self.SubModules[Name])
        Param.Module.Num = self.ModuleNum = len(self.ModuleList)
        return self
    def AddSubModule(self, Name=None, SubModule=None, **Dict):
        if Name is not None:
            assert len(Dict) == 0
            assert SubModule is not None
            return self.AppendSubModule(Name=Name, SubModule=SubModule)
        else:
            for _Name, _SubModule in Dict.items():
                self.AppendSubModule(Name=_Name, SubModule=_SubModule)
        return self
    def AppendSubModule(self, Name=None, SubModule=None):
        Param = self.Param
        ModuleList = self.ModuleList
        Index = len(ModuleList)
        self.ModuleList.append(SubModule)
        if Name is None:
            Name = f"L{Index}"
        super().AddSubModule(Name, SubModule)
        self.ModuleNum = Param.Module.Num = len(ModuleList)
        return self
    def _forward_out(self, In):
        OutList = []
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            OutList.append(Out)
            In = Out
        return Out
    def _forward_all_tuple(self, In):
        OutList = []
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            OutList.append(Out)
            In = Out
        return tuple(OutList)
    def _forward_all_dict(self, In):
        OutDict = {"In": In}
        for LayerIndex in range(self.ModuleNum):
            Out = self.ModuleList[LayerIndex](In)
            OutDict[self.OutNameList[LayerIndex]] = Out
            In = Out
        return OutDict

    def Init(self, IsSuper=False, IsRoot=True):
        if self.IsLoad():
            self.ModuleList = list(self.SubModules.values())
        
        Param = self.Param
        OutType = Param.Out.setdefault("Type", "Out")
        import functools

        ReceiveMethodMap = DLUtils.IterableKeyToElement({
            ("Out", "OutOnly"): self._forward_out,
            ("All", "AllInTuple"): self._forward_all_tuple,
            ("AllInDict"): self._forward_all_dict
        })

        #self.Receive = functools.partial(self.ReceiveMethodMap[OutType], self=self)
        self.Receive = ReceiveMethodMap[OutType]
        self.ModuleNum = Param.Module.Num = len(self.ModuleList)
        assert self.ModuleNum > 0
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.LayerNum = self.ModuleNum
        if OutType in ["AllInDict"]:
            if Param.Out.hasattr("Name"):
                self.OutNameList = Param.Out.Name
            else:
                self.OutNameList = ["L%d"%LayerIndex for LayerIndex in range(self.LayerNum)]

        return self
    

ModuleList = ModuleSeries