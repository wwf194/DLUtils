import DLUtils
from .nonlinear import NonLinearLayer
from ..module import _ModuleList

class MLP(_ModuleList):
    ParamMap = DLUtils.IterableKeyToElement({
        ("NonLinear"): "NonLinear.DefaultType",
        ("NonLinearOnLastLayer"): "NonLinear.ApplyOnLastLayer",
        ("BiasOnLastLayer"): "Bias.ApplyOnLastLayer",
        ("UnitNum", "NeuronNum"): "Layer.Unit.Num",
        ("Bias"): "Bias.Enable"
    })
    def __init__(self, UnitNum=None, *List, **Dict):        
        if len(List) > 0:
            assert isinstance(UnitNum, int)
            for _UnitNum in List:
                assert isinstance(_UnitNum, int)
            Dict["UnitNum"] = [UnitNum, *List]
        elif UnitNum is not None:
            UnitNum = DLUtils.ToList(UnitNum)
            Dict["UnitNum"] = UnitNum
            assert len(UnitNum) > 1
        super().__init__(**Dict)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if self.IsInit():
            assert Param.Layer.Unit.hasattr("Num")
            LayerNum = Param.Layer.Num = len(Param.Layer.Unit.Num) - 1
            
            # nonlinear setting
            Param.NonLinear.setdefault("Enable", True)
            if Param.NonLinear.Enable:
                if not Param.NonLinear.hasattr("Type"):
                    Param.NonLinear.setdefault("Type", "ReLU")
                Param.NonLinear.setdefault("ApplyOnLastLayer", True)
            
            # bias setting
            Param.Bias.setdefault("Enable", True)
            if Param.Bias.Enable:
                Param.Bias.setdefault("ApplyOnLastLayer", True)

            for Index in range(Param.Layer.Num):
                NonLinearStr = self.GetNonLinear(Index, LayerNum)
                Bias = True
                if Index == Param.Layer.Num - 1:
                    if not Param.Bias.ApplyOnLastLayer:
                        Bias = False
                self.AppendSubModule(
                    Name="L%d"%Index,
                    SubModule=NonLinearLayer(
                        InSize=Param.Layer.Unit.Num[Index],
                        OutSize=Param.Layer.Unit.Num[Index + 1],
                        NonLinear=NonLinearStr,
                        Bias=Bias
                    )
                )
        self.LayerNum = Param.Layer.Num
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def GetNonLinear(self, LayerIndex, LayerNum):
        Param = self.Param
        NonLinearType = Param.NonLinear.Type
        if Param.NonLinear.Enable:
            Param.NonLinear
            if LayerIndex == LayerNum - 1: # Last Layer:
                if Param.NonLinear.get("ApplyOnLastLayer"):
                    return NonLinearType
                else:
                    return "None"
            else:
                return NonLinearType
        else:
            return "None"