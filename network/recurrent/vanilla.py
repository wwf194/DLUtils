
import DLUtils
import torch
class VanillaRNN(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("LayerNum"): "Layer.Num",
        ("HiddenNum", "FeatureNum"): ("Hidden.Num"),
        ("QKSize"): "MSA.Attention.QK.Size", # total size. not size of each head.
        ("VSize"): "MSA.Attention.V.Size", # total size. not size of each head.
        ("NonLinear", "HiddenNonLinear", "RecurrentNonLinear"): "Hidden.NonLinear",
        ("DropOut"): "DropOut.Probability",
        ("DropOutInplace", "DropOutInPlace"): "DropOut.InPlace"
    })
    def __init__(self, SubModule=None):
        super().__init__()
        if SubModule is not None:
            self.AddSubModule("SubModule", SubModule)
    def Receive(self, InList, StepNum=None):
        # InList: (StepNum, BatchSize, InNum)
        OutList = []
        Hidden = self.GetInitState()
        if StepNum is None:
            StepNum = self.StepNum
        for StepIndex in range(StepNum):
            In = InList[StepNum, :, :]
            In = self.In(In)
            Hidden = self.Hidden(Hidden)
            Hidden = In + Hidden
            Hidden = self.NonLinear(self.RecurrentWeight, Hidden)
            Out = self.Out(Hidden)
            OutList.append(Out)
        
        OutFinal = self.GetOut(OutList)
        return {
            "Out": OutFinal
        }

    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if self.IsInit():
            if not self.HasSubModule("In"):
                self.AddSubModule(
                    Name="In",
                    SubModule=DLUtils.network.NonLinearLayer(
                        InNum=Param.In.Num,
                        OutNum=Param.Hidden.Num,
                        NonLinear=Param.In.setdefault("NonLinear", "None")
                    )
                )
            
            # recurrent module setting
            if not self.HasSubModule("Hidden"):
                self.AddSubModule(
                    Name="Hidden",
                    SubModule=DLUtils.network.NonLinearLayer(
                        InNum=Param.Hidden.Num,
                        OutNum=Param.Hidden.Num,
                        NonLinear=Param.Hidden.setdefault("NonLinear", "ReLU")
                    ).SetWeight(
                        DLUtils.DefaultVanillaRNNHiddenWeight(
                            (Param.Hidden.Num, Param.Hidden.Num)
                        )
                    )
                )
            

            # output setting
            if not self.HasSubModule("Out"):
                self.AddSubModule(
                    Name="Out",
                    SubModule=DLUtils.network.NonLinearLayer(
                        InNum=Param.Hidden.Num,
                        OutNum=Param.Out.Num,
                        NonLinear=Param.In.setdefault("NonLinear", "None")
                    )
                )
            
            Param.Out.setdefault("Mode", "All")

            # initial state setting
            Param.InitState.setdefault("Method", "Fixed")
            if Param.InitState.Method in ["Fixed"]:
                if not Param.InitState.hasattr("Data"):
                    self.SetTensor(
                        Name="InitState",
                        Path="InitState.Data",
                        Value=DLUtils.SampleFrom01NormalDistribution(
                            (Param.Hidden.Num)
                        )
                    )
                    return
                Param.InitState.setdefault("Trainable", True)

                if Param.InitState.Trainable:
                    self.Register
            
        # init state setting
        # init state is the first hidden state
        InitStateMethod = self.InitStateMethod = Param.InitState.Method
        if InitStateMethod in ["Zero", "None"]:
            self.GetInitState = self.GetInitStateZero
        elif InitStateMethod in ["Fixed"]:
            self.GetInitState = self.GetInitStateFixed
        else:
            raise Exception()

        # output method setting
        OutMode = self.OutMode = Param.Out.Mode
        if OutMode in ["Last"]:
            self.GetOut = self.GetOutLast
        elif OutMode in ["All"]:
            self.GetOut = self.GetOutAll
        else:
            raise Exception()

        self.HiddenNum = Param.Hidden.Num
        self.InNum = Param.In.Num
        self.OutNum = Param.Out.Num
        
        if Param.Iter.hasattr("Num"):
            self.StepNum = Param.Iter.Num
        
        
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def GetInitStateZero(self, BatchSize):
        return torch.zeros((BatchSize, self.HiddenNum), device=self.Device)
    def GetInitStateFixed(self, BatchSize):
        return self.InitState
    def GetOutLast(self, OutList):
        # OutList: (StepNum, BatchSize, OutNum)
        return OutList[-1]
    def GetOutAll(self, OutList):
        return OutList