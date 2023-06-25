import numpy as np
import scipy
import cv2 as cv
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_style("white")

default_res=60

import DLUtils

def SetMatplotlibParamToDefault():
    mpl.rcParams.update(mpl.rcParamsDefault)
# matplotlib default param might be overwritten by other lib using it, such as seaborn.
SetMatplotlibParamToDefault() # ticks may disappear after imshow.

NamedColor = DLUtils.Param().from_dict({
    "White": (1.0, 1.0, 1.0),
    "Black": (0.0, 0.0, 0.0),
    "Red":   (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue":  (0.0, 0.0, 1.0),
    "Gray":  (0.5, 0.5, 0.5),
    "Grey":  (0.5, 0.5, 0.5),
    "LightGray": (211.0 / 256.0, 211.0 / 256.0, 211.0 / 256.0) # D3D3D3
})

# def get_cmap(n, name='gist_rainbow'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)

def GenerateColors(Num=10, ColorMap="Auto"):
    # ColorMap: hsv: some colors look too similar.
    assert isinstance(Num, int) and Num > 0
    if Num <= 10:
        ColorMap = "tab10"
    elif Num <= 20:
        ColorMap = "tab20"
    else:
        raise Exception()
    cmap = plt.cm.get_cmap(ColorMap)  # type: matplotlib.colors.ListedColormap
    Colors = cmap.colors  # type: list
    ColorList = []
    for Index in range(Num):
        ColorList.append(Colors[Index])
    
    return ColorList

def PlotLinesPlt(ax, XYsStart, XYsEnd=None, Width=1.0, Color=NamedColor.Black):
    if XYsEnd is None:
        Edges = DLUtils.ToNpArray(XYsStart)
        XYsStart = Edges[:, 0]
        XYsEnd = Edges[:, 1]
    else:
        XYsStart = DLUtils.ToNpArray(XYsStart)
        XYsEnd = DLUtils.ToNpArray(XYsEnd)
    LineNum = XYsStart.shape[0]
    for Index in range(LineNum):
        X = [XYsStart[Index, 0], XYsEnd[Index, 0]]
        Y = [XYsStart[Index, 1], XYsEnd[Index, 1]]
        ax.add_line(Line2D(X, Y, linewidth=Width, color=Color))
PlotLines = PlotLinesPlt

def SetHeightWidthRatio(ax, ratio):
    ax.set_aspect(ratio)

def PlotLineAndMarkVerticesXY(ax, PointStart, PointEnd, Width=1.0, Color=NamedColor.Black):
    PlotLinePlt(ax, PointStart, PointEnd, Width, Color)
    PlotPointAndMarkXY(ax, PointStart)
    PlotPointAndMarkXY(ax, PointEnd)

def PlotArrowAndMarkVerticesXY(ax, PointStart, PointEnd, Width=0.001, Color=NamedColor.Black):
    PlotArrowFromVertexPairsPlt(ax, PointStart, PointEnd, Width=Width, Color=Color)
    PlotPointAndMarkXY(ax, PointStart)
    PlotPointAndMarkXY(ax, PointEnd)

def PlotLinePlt(ax, PointStart, PointEnd, Width=1.0, Color=NamedColor.Black, Style="-"):
    # Width: Line Width in points(?pixels)
    X = [PointStart[0], PointEnd[0]]
    Y = [PointStart[1], PointEnd[1]]
    line = ax.add_line(Line2D(X, Y))
    line.set_linewidth(Width)
    line.set_color(Color)
    line.set_linestyle(Style)
PlotLine = PlotLinePlt

def PlotDashedLinePlt(ax, PointStart, PointEnd, Width=1.0, Color=NamedColor.Black):
    PlotLinePlt(ax, PointStart, PointEnd, Width, Color, Style=(0, (5, 10)))
PlotDashedLine = PlotDashedLinePlt

def PlotPointAndAddText(ax, XY, PointColor=NamedColor.Blue, Text="TextOnPoint", TextColor=None):
    if TextColor is None:
        TextColor = PointColor
    PlotPoint(ax, XY, PointColor)
    PlotText(ax, XY, Text, TextColor)

def PlotText(ax, XY, Text, Color=NamedColor.Blue):
    ax.text(XY[0], XY[1], Text, color=Color)

def print_notes(notes, y_line, y_interv):
    if(notes!=""):
        if(isinstance(notes, str)):
            plt.annotate(notes, xy=(0.02, y_line), xycoords='axes fraction')
            y_line-=y_interv
        elif(isinstance(notes, list)):
            for note in notes:
                plt.annotate(notes, xy=(0.02, y_line), xycoords='axes fraction')
                y_line-=y_interv
        else:
            print("invalid notes type")

def ParsePointTypePlt(Type):
    if isinstance(Type, str):
        if Type in ["Circ", "Circle", "EmptyCircle"]:
            return "o"
        elif Type in ["Triangle"]:
            return "^"
        else:
            #raise Exception(Type)
            return Type
    else:
        #raise Exception(Type)
        return Type

def PlotPoint(ax, XY, Color=NamedColor.Blue, Type="Circle", Size=None):
    Color = ParseColor(Color)
    Type = ParsePointTypePlt(Type)
    
    if Size is not None:
        if isinstance(Size, float):
            Size = mpl.rcParams['lines.markersize'] ** 2 * Size ** 2
    ax.scatter(
        [XY[0]], [XY[1]], 
        s = Size,
        color=Color, marker=Type
    )

def ParseMarkerSize(Size):
    if Size is None:
        pass
    elif isinstance(Size, float):
        Size = mpl.rcParams['lines.markersize'] ** 2 * Size ** 2
    else:
        raise Exception()
    return Size

def PlotPointsPltNp(
        ax, Points, Color="Blue", Type="Circle", Size=None,
        XLabel=None, YLabel=None, Title=None, XRange=None, YRange=None
    ):
    Points = DLUtils.ToNpArray(Points)
    Xs = Points[:, 0]
    Ys = Points[:, 1]
    Color=ParseColor(Color)
    Size = ParseMarkerSize(Size)
    Type = ParsePointTypePlt(Type)
    ax.scatter(Xs, Ys, color=Color, s=Size, marker=Type, facecolors="none")
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SetTicksAndRangeForAx(ax, Xs, Ys, XRange, YRange)
    return
PlotPoints = PlotPointsPltNp

def PlotMultiPoints(
        ax, Xs, Ys, Color="Blue", Type="Circle", Size=None, Labels=None,
        XLabel=None, YLabel=None, Title=None, XRange=None, YRange=None
    ):
    GroupNum = len(Xs)
    Colors = GenerateColors(GroupNum)
    if Labels is None:
        Labels = [None for _ in range(GroupNum)]
    for Index, (_Xs, _Ys) in enumerate(zip(Xs, Ys)):
        Color=ParseColor(Colors[Index])
        Size = ParseMarkerSize(Size)
        Type = ParsePointTypePlt(Type)
        ax.scatter(_Xs, _Ys, color=Color, s=Size, marker=Type, facecolors="none", label=Labels[Index])
    ax.legend(loc="upper right")
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    # SetTicksAndRangeForAx(ax, Xs, Ys, XRange, YRange)
    return

def SetXTicksAndRange(*List, **Dict):
    XTicks = Dict.setdefault("XTicks", "float")
    if XTicks in ["float", "Float"]:
        SetXTicksAndRangeFloat(*List, **Dict)
    elif XTicks in ["int", "Int"]:
        SetXTicksAndRangeInt(*List, **Dict)
    else:
        raise Exception(XTicks)

def SetYTicksAndRange(*List, **Dict):
    YTicks = Dict.setdefault("YTicks", "float")
    if YTicks in ["float", "Float"]:
        SetYTicksAndRangeFloat(*List, **Dict)
    elif YTicks in ["int", "Int"]:
        SetYTicksAndRangeInt(*List, **Dict)
    else:
        raise Exception(YTicks)

def SetXTicksInt(ax, Min, Max, Method="Auto", **Dict):
    if Min > Max:
        Min, Max = Max, Min
    # Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max, **Dict)
    ax.set_xticks(Ticks, labels=TicksStr)
    #ax.set_xticklabels(TicksStr)
    return Ticks, TicksStr

def GetXTicksInt(Min, Max, Method="Auto", **Dict):
    if Min > Max:
        Min, Max = Max, Min
    #Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max, **Dict)
    return Ticks, TicksStr

def Min(Data, DimNum=1):
    if DimNum == 1:
        return np.nanmin(Data)
    elif DimNum == 2:
        return MinList2D(Data)
    else:
        raise Exception()

def Max(Data, DimNum=1):
    if DimNum == 1:
        return np.nanmax(Data)
    elif DimNum == 2:
        return MaxList2D(Data)
    else:
        raise Exception()

def MinList2D(List):
    if len(List) == 0:
        return None
    SubList = List[0]
    Min = min(SubList)
    for SubList in List[1:]:
        _Min = min(SubList)
        if _Min < Min:
            Min = _Min
    return Min

def MaxList2D(List):
    if len(List) == 0:
        return None
    SubList = List[0]
    Max = max(SubList)
    for SubList in List[1:]:
        _Max = max(SubList)
        if _Max > Max:
            Max = _Max
    return Max

def SetXTicksAndRangeInt(ax, Xs, Range=None, **Dict):
    XMin = Dict.setdefault("XMin", None)
    XMax = Dict.setdefault("XMax", None)
    DimNum = Dict.setdefault("DimNum", 1)
    if Range is None:
        if XMin is None:
            XMin = Min(Xs, DimNum)
            XMin = math.floor(XMin)
        if XMax is None:
            XMax = Max(Xs, DimNum)
            XMax = math.ceil(XMax)
    else:
        XMin, XMax = Range[0], Range[1]
    assert isinstance(XMin, int) and isinstance(XMax, int)
    SetXTicksInt(ax, XMin, XMax)
    SetXRangeMinMaxInt(ax, XMin, XMax)

def SetYTicksAndRangeInt(ax, Ys, Range=None, **Dict):
    YMin = Dict.setdefault("YMin", None)
    YMax = Dict.setdefault("YMax", None)
    DimNum = Dict.setdefault("DimNum", 1)
    if Range is None:
        if YMin is None:
            YMin = Min(Ys, DimNum)
            YMin = math.floor(YMin)
        if YMax is None:
            YMax = Max(Ys, DimNum)
            YMax = math.ceil(YMax)
    else:
        YMin, YMax = Range[0], Range[1]
    assert isinstance(YMin, int) and isinstance(YMax, int)
    SetYTicksInt(ax, YMin, YMax)
    SetYRangeMinMaxInt(ax, YMin, YMax)

def SetYTicksInt(ax, Min, Max, Method="Auto", **Dict):
    if Min > Max:
        Min, Max = Max, Min
    #Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max, **Dict)
    ax.set_yticks(Ticks, labels=TicksStr)
    #ax.set_yticklabels(TicksStr)
    return Ticks, TicksStr

def GetYTicksInt(Min, Max, Method="Auto", **Dict):
    if Min > Max:
        Min, Max = Max, Min
    #Min, Max = round(Min) + 1, round(Max)
    Ticks, TicksStr = CalculateTicksInt(Method, Min, Max, **Dict)
    return Ticks, TicksStr

def SetTicksAndRangeForAx(ax, Xs, Ys, XRange, YRange):
    SetXTicksAndRangeFloat(ax, Xs=Xs, Range=XRange)
    SetYTicksAndRangeFloat(ax, Ys=Ys, Range=YRange)

def SetXTicksAndRangeFloat(ax, Xs, Range=None, **Dict):
    XMin = Dict.setdefault("XMin", None)
    XMax = Dict.setdefault("XMax", None)
    DimNum = Dict.setdefault("DimNum", 1)
    if Range is None:
        if XMin is None:
            XMin = Min(Xs, DimNum)
        if XMax is None:
            XMax = Max(Xs, DimNum)
    else:
        XMin, XMax = Range[0], Range[1]
    SetXTicksFloat(ax, XMin, XMax)
    SetXRangeMinMaxFloat(ax, XMin, XMax)

def SetYTicksAndRangeFloat(ax, Ys, Range=None, **Dict):
    YMin = Dict.setdefault("YMin", None)
    YMax = Dict.setdefault("YMax", None)
    DimNum = Dict.setdefault("DimNum", 1)
    if Range is None:
        if YMin is None:
            YMin = Min(Ys, DimNum)
        if YMax is None:
            YMax = Max(Ys, DimNum)
        #YMin, YMax = Range[0], Range[1]
    SetYTicksFloat(ax, YMin, YMax)
    SetYRangeMinMaxFloat(ax, YMin, YMax)

def PlotPointsPltNp(
        ax, Points, Color="Blue", Type="Circle", Size=None,
        XLabel=None, YLabel=None, Title=None, XRange=None, YRange=None
    ):
    Points = DLUtils.ToNpArray(Points)
    Xs = Points[:, 0]
    Ys = Points[:, 1]
    Color=ParseColor(Color)
    Size = ParseMarkerSize(Size)
    Type = ParsePointTypePlt(Type)
    ax.scatter(Xs, Ys, color=Color, s=Size, marker=Type, facecolors="none")
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SetTicksAndRangeForAx(ax, Xs, Ys, XRange, YRange)
    return

def PlotDirectionsOnEdges(ax, Edges, Directions, **kw):
    Edges = DLUtils.ToNpArray(Edges)
    Directions = DLUtils.ToNpArray(Directions)
    EdgesLength = DLUtils.geometry2D.Vectors2Lengths(DLUtils.geometry2D.VertexPairs2Vectors(Edges))
    ArrowsLength = 0.2 * np.median(EdgesLength, keepdims=True)
    Directions = DLUtils.geometry2D.Vectors2GivenLengths(Directions, ArrowsLength)
    MidPoints = DLUtils.geometry2D.Edges2MidPointsNp(Edges)
    PlotArrows(ax, MidPoints - 0.5 * Directions, Directions, **kw)

def PlotDirectionOnEdge(ax, Edge, Direction, **kw):
    PlotDirectionsOnEdges(ax, [Edge], [Direction], **kw)

def Str2RGB(ColorStr):
    if ColorStr in NamedCol:
        return NamedCol.getattr(ColorStr)
    else:
        raise Exception()

def Map2Color(
        data, ColorMap="jet", Method="MinMax", Alpha=False,
        RefData=None, **Dict
    ):
    data = DLUtils.ToNpArray(data)
    ColorForEqualValues = Dict.setdefault("ColorForEqualValues", ParseColor("LightGray"))

    if Method in ["MinMax", "GivenRange", "GivenMinMax"]:
        if Method in ["MinMax"]:
            if RefData is not None: # If RefData is provided, color mapping will be on RefData, rather than data.
                if RefData.size > 0:
                    dataMin, dataMax = np.nanmin(RefData), np.nanmax(RefData)
                else:
                    dataMin, dataMax = np.NaN, np.NaN
            else:
                dataMin, dataMax = np.nanmin(data), np.nanmax(data)
        elif Method in ["GivenRange", "GivenMinMax"]:
            dataMin, dataMax = Dict["Min"], Dict["Max"]

        # corner case: all values equal to each other.
        if dataMin == dataMax or not np.isfinite(dataMin) or not np.isfinite(dataMax):
            if len(ColorForEqualValues) == 3:
                Color = [*ColorForEqualValues, 0.0]
            elif len(ColorForEqualValues) == 4:
                Color = ColorForEqualValues
            else:
                raise Exception()
            dataInColor = np.full((*data.shape, 4), Color)
        else:
            dataNormed = (data - dataMin) / (dataMax - dataMin) # normalize to [0, 1]
            dataInColor = ParseColorMapPlt(ColorMap)(dataNormed) # [*data.Shape, (r,g,b,a)]
    else:
        raise Exception(Method)

    if not Alpha: # Alpha == 1.0 means total transparency / totally unseeable.
        dataInColor = eval("dataInColor[%s 0:3]"%("".join(":," for _ in range(len(data.shape)))))
        #dataColored = dataColored.take([0, 1, 2], axis=-1)
    return DLUtils.Param({
        "dataInColor": dataInColor,
        # "dataNormed": dataNormed,
        "Min": dataMin,
        "Max": dataMax
    })
Map2Colors = Map2Color

def MapColors2Colors(dataColored, ColorMap="jet", Range="MinMax"):
    return # to be implemented

def norm(data):
    # to be implemented
    return

def ParseResolutionXY(Resolution, Width, Height):
    if Width >= Height:
        ResolutionX = Resolution
        ResolutionY = int( Resolution * Height / Width )
    else:
        ResolutionY = Resolution
        ResolutionX = int(Resolution * Width / Height )
    return ResolutionX, ResolutionY

def PlotXYs(ax, XYs, XYsMark=None):
    if XYsMark is None:
        XYsMark = XYs
    for Index, XY in enumerate(XYs):
        PlotPointAndAddText(ax, XY, Text="(%.2f, %.2f)"%(XYsMark[Index][0], XYsMark[Index][1]))

def PlotPointAndMarkXY(ax, XY):
    PlotPointAndAddText(ax, XY, Text="(%.2f, %.2f)"%(XY[0], XY[1]))

def ParseColor(Color):
    if isinstance(Color, tuple):
        if len(Color)==3:
            return Color
        elif len(Color)==4:
            return Color[0:3]
        else:
            raise Exception()
    elif isinstance(Color, str):
        if hasattr(NamedColor, Color):
            return getattr(NamedColor, Color)
        else:
            return Color
    else:
        raise Exception()

def ParseColorMapPlt(ColorMap):
    if isinstance(ColorMap, str):
        return plt.get_cmap(ColorMap)
    else:
        raise Exception()

ParseColorMap = ParseColorMapPlt

def PlotArrows(
        ax, XYsStart, dXYs, Color=NamedColor.Red,
        HeadWidth=0.05, HeadLength=0.1, SizeScale=1.0, 
        XLabel=None, YLabel=None, Title=None, XRange=None, YRange=None
    ):
    PlotNum = len(XYsStart)
    for Index in range(PlotNum):
        PlotArrowPlt(ax, XYsStart[Index], dXYs[Index], Color=Color, HeadWidth=HeadWidth, SizeScale=SizeScale)
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SetTicksAndRangeForAx(ax, XYsStart[:, 0], XYsStart[:, 1], XRange, YRange)

def PlotArrowFromVertexPairsPlt(ax, XYStart, XYEnd, Width=0.001, Color=NamedColor.Red, SizeScale=1.0):
    XYStart = DLUtils.ToNpArray(XYStart)
    XYEnd = DLUtils.ToNpArray(XYEnd)
    PlotArrowPlt(ax, XYStart, XYEnd - XYStart, Width=Width, Color=Color, SizeScale=SizeScale)

def PlotArrowPlt(ax, XYStart, dXY, Width=0.001, HeadWidth=0.05, HeadLength=0.1, Color=NamedColor.Red, SizeScale=None):
    Color = ParseColor(Color)
    XYStart = DLUtils.ToList(XYStart)
    dXY = DLUtils.ToList(dXY)
    if SizeScale is not None:
        Width = Width * SizeScale
        HeadWidth = HeadWidth * SizeScale
        HeadLength = HeadLength * SizeScale
    else:
        SizeScale = 1.0
    ax.arrow(*XYStart, *dXY, 
        width=Width * SizeScale,
        head_width=HeadWidth * SizeScale,
        head_length=HeadLength * SizeScale,
        facecolor=Color,
        edgecolor=Color
    )

def PlotPolyLineFromVerticesPlt(ax, Points, Color=NamedColor.Black, Width=2.0, Closed=False):
    # Points: np.ndarray with shape [PointNum, (x,y)]
    Points = DLUtils.ToList(Points)
    PointNum = len(Points)
    if Closed:
        LineNum = PointNum + 1
    else:
        LineNum = PointNum
        #Points = np.concatenate((Points, Points[-1, :][np.newaxis, :]), axis=0)

    if isinstance(Color, np.ndarray):
        pass
    else:
        for Index in range(LineNum):
            # DLUtils.Log("(%.3f %.3f) (%.3f %.3f)"%(Points[Index][0], Points[Index][1], Points[(Index + 1)%PointNum][0], Points[(Index + 1)%PointNum][1]))
            PlotLinePlt(ax, Points[Index], Points[(Index + 1)%PointNum], Color=Color, Width=Width)

PlotPolyLine = PlotPolyLineFromVerticesPlt
    # if isinstance(color, dict):
    #     method = search_dict(color, ['method', 'mode'])
    #     if method in ['start-end']:
    #         color_start = np.array(color['start'], dtype=np.float)
    #         color_end = np.array(color['end'], dtype=np.float)
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ratio = i / ( plot_num - 1 )
    #             color_now = tuple( ratio * color_end + (1.0 - ratio) * color_start )
    #             ax.add_line(Line2D(xs, ys, linewidth=Width, color=color_now ))
    #     elif method in ['given']:
    #         color = search_dict(color, ['content', 'data'])
    #         for i in range(plot_num):
    #             xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #             ax.add_line(Line2D(xs, ys, linewidth=Width, color=color[i] ))           
    #     else:
    #         raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))
        
    # elif isinstance(color, tuple):
    #     for i in range(plot_num):
    #         xs, ys = [ points[i][0], points[(i+1)%PointNum][0]], [ points[i][1], points[(i+1)%PointNum][1]]
    #         ax.add_line(Line2D(xs, ys, linewidth=Width, color=color ))
    # else:
    #     raise Exception('PlotPolyLinePlt: invalid color mode:'+str(method))

def ParseResolution(Width, Height, Resolution):
    if Width < Height:
        ResolutionX = Resolution
        ResolutionY = round(ResolutionX * Height / Width)
    else:
        ResolutionY = Resolution
        ResolutionX = round(ResolutionY * Width / Height)
    return ResolutionX, ResolutionY

def PlotPolyLineFromVerticesCV(img, points, closed=False, Color=(0,0,0), Width=2, Type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]): #points:[PointNum, (x,y)]
    if isinstance(points, list):
        points = np.array(points)

    PointNum = points.shape[0]
    if closed:
        line_num = points.shape[0] - 1
    else:
        line_num = points.shape[0]

    ResolutionX, ResolutionY = img.shape[0], img.shape[1]

    for i in range(line_num):
        point_0 = DLUtils.geometry2D.XY2PixelIndex(points[i%PointNum][0], points[i%PointNum][1], BoundaryBox, ResolutionX, ResolutionY)
        point_1 = DLUtils.geometry2D.XY2PixelIndex(points[(i+1)%PointNum][0], points[(i+1)%PointNum][1], BoundaryBox, ResolutionX, ResolutionY)
        cv.line(img, point_0, point_1, Color, Width, type)

def SetAxRangeFromBoundaryBox(ax, BoundaryBox, SetTicks=True):
    ax.set_xlim(BoundaryBox.XMin, BoundaryBox.XMax)
    ax.set_ylim(BoundaryBox.YMin, BoundaryBox.YMax)
    if SetTicks:
        SetXTicksFloat(ax, BoundaryBox.XMin, BoundaryBox.XMax)
        SetYTicksFloat(ax, BoundaryBox.YMin, BoundaryBox.YMax)

def GetDefaultBoundaryBox():
    return DLUtils.PyObj({
        "XMin": 0.0,
        "XMax": 1.0,
        "YMin": 0.0,
        "YMax": 1.0,
    })

def XYs2BoundaryBox(XYs):
    XYs = XYs.reshape(-1, 2) # Flatten to [:, (x, y)]
    Xs = XYs[:, 0]
    Ys = XYs[:, 1]
    BoundaryBox = [np.min(Xs), np.min(Ys), np.max(Xs), np.max(Ys),]
    return DLUtils.PyObj({
        "XMin": BoundaryBox[0],
        "XMax": BoundaryBox[2],
        "YMin": BoundaryBox[1],
        "YMax": BoundaryBox[3],
    })

def UpdateBoundaryBox(BoundaryBox, _BoundaryBox):
    BoundaryBox.XMin = min(BoundaryBox.XMin, _BoundaryBox.XMin)
    BoundaryBox.YMin = min(BoundaryBox.YMin, _BoundaryBox.YMin)
    BoundaryBox.XMax = max(BoundaryBox.XMax, _BoundaryBox.XMax)
    BoundaryBox.YMax = max(BoundaryBox.YMax, _BoundaryBox.YMax)
    BoundaryBox.__value__ = [
        BoundaryBox.XMin,
        BoundaryBox.YMin,
        BoundaryBox.XMax,
        BoundaryBox.YMax,
    ]
    BoundaryBox.Width = BoundaryBox.XMax - BoundaryBox.XMin
    BoundaryBox.Height = BoundaryBox.YMax - BoundaryBox.YMin
    return BoundaryBox

def CopyBoundaryBox(_BoundaryBox):
    BoundaryBox = GetDefaultBoundaryBox()
    BoundaryBox.XMin = _BoundaryBox.XMin
    BoundaryBox.YMin = _BoundaryBox.YMin
    BoundaryBox.XMax = _BoundaryBox.XMax
    BoundaryBox.YMax = _BoundaryBox.YMax
    BoundaryBox.__value__ = [
        BoundaryBox.XMin,
        BoundaryBox.YMin,
        BoundaryBox.XMax,
        BoundaryBox.YMax,
    ]
    BoundaryBox.Width = BoundaryBox.XMax - BoundaryBox.XMin
    BoundaryBox.Height = BoundaryBox.YMax - BoundaryBox.YMin
    return BoundaryBox

def SetAxRangeAndTicksFromBoundaryBox(ax, BoundaryBox):
    SetAxRangeFromBoundaryBox(ax, BoundaryBox)
    SetXTicksFloat(ax, BoundaryBox.XMin, BoundaryBox.XMax)
    SetYTicksFloat(ax, BoundaryBox.YMin, BoundaryBox.YMax)

def ParseIsDataColored2D(data):
    DimensionNum = DLUtils.GetDimensionNum(data)
    if DimensionNum == 2: # [XNum, YNum]
        IsDataColored = False
    elif DimensionNum == 3: # [XNum, YNum, (r, g, b) or (r, g, b, a)], already mapped to colors.
        IsDataColored = True
    else:
        raise Exception(DimensionNum)
    return IsDataColored

def PlotMatrixWithColorBar(
        ax, data=None, dataInColor=None, ColorMap="jet", ColorMethod="MinMax", 
        XYRange=None, Coordinate="Math", dataMask=None,
        Ticks=None, XLabel=None, YLabel=None, Title=None,
        ColorBarOrientation="Auto", ColorBarLocation=None,
        PixelHeightWidthRatio="Equal", DataHeightWidthRatioMax = 5.0, DataHeightWidthRatioMin = 0.2,
        ColorBarTitle=None, Save=False, SavePath=None, **Dict
    ):
    DataHeightWidthRatioMax = Dict.setdefault("DataHeightWidthRatioMax", 5.0)
    DataHeightWidthRatioMin = Dict.setdefault("DataHeightWidthRatioMin", 0.2)
    assert DataHeightWidthRatioMin <= DataHeightWidthRatioMax

    ColorForEqualValues = Dict.setdefault("ColorForEqualValues", ParseColor("LightGray"))
    if dataInColor is None:
        RefDataForColorMap = Dict.setdefault("RefDataForColorMap", None)
        if RefDataForColorMap is not None:
            MapParam = Map2Color(data, ColorMap, DataRef=RefDataForColorMap, ColorForEqualValues=ColorForEqualValues)
        else:
            MapParam = Map2Color(data, ColorMap, ColorForEqualValues=ColorForEqualValues)
        dataInColor = MapParam.dataInColor
        Min = MapParam.Min
        Max = MapParam.Max
        Dict.update({
            "Min": Min,
            "Max": Max,
        })
    STATE0 = PlotMatrix(
        ax, data=data, dataInColor=dataInColor, ColorMap=ColorMap, XYRange=XYRange, Coordinate=Coordinate, 
        PixelHeightWidthRatio=PixelHeightWidthRatio, Ticks=Ticks,
        dataMask=dataMask, Save=False,
        XLabel=XLabel, YLabel=YLabel, Title=Title, **Dict
    )
    
    aspect = GetAspectOfAx(ax)
    axColorBar = GetSubAx(ax, 1.0 + 0.1 / aspect, 0.0, 0.1 / aspect, 1.0)
    ColorBarOrientation = "vertical"

    STATE1 = PlotColorBarInAx(
        axColorBar, ColorMap=ColorMap, Method=ColorMethod, Orientation=ColorBarOrientation, 
        Location=ColorBarLocation, Title=ColorBarTitle,
        **Dict
    )

    #SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)
    return STATE0 | STATE1


def PlotMatrix(
        ax, data=None, dataInColor=None, 
        ColorMap="jet", XYRange=None, Coordinate="X1Y0", dataMask=None, PixelHeightWidthRatio="Auto",
        XLabel=None, YLabel=None, Title=None, Ticks=None, Origin=None,
        Save=False, SavePath=None, Format="svg", **Dict
    ):
    if dataInColor is None:
        # dataMask: [XNum, YNum]. Value: 0 if elemented is to be masked else 1.
        MaskInfOrNaN = Dict.setdefault("MaskInfOrNaN", True)
        if MaskInfOrNaN:
            MaskInfOrNaN = InfOrNaNMask(data)

        if dataMask is None:
            dataMask = MaskInfOrNaN
        else:
            dataMask *= MaskInfOrNaN
        MapParam = Map2Color(data, ColorMap)
        dataInColor = MapParam.dataInColor
    else:
        # masking out inf or nan will not be proceeded
        pass

    if dataMask is not None:
        MaskIn = dataMask.astype(np.float32)
        MaskOut = (~dataMask).astype(np.float32)
        MaskColor = Dict.setdefault("maskColor", "Black")
        MaskColor = ParseColor(MaskColor)
        dataInColor = MaskIn[:, :, np.newaxis] * dataInColor + MaskOut[:, :, np.newaxis] * MaskColor

    XAxisLocation, YAxisLocation = None, None
    if Coordinate in ["X1Y0", "YFirst", "Y0X1"]:        
        # Place the [0, 0] index of the data in the lower left
        #origin = 'lower'
        if Origin is None:
            Origin = "UpperLeft"
    elif Coordinate in ["X0Y1", "Y1X0", "XFirst", "XLeftYTop"]:
        # Place the [0, 0] index of the data in the upper left
        if Origin is None:
            Origin = "LowerLeft"
    else:
        raise Exception()
    if Origin in ["UpperLeft"]:
        XAxisLocation = Dict.setdefault("XAXisLocation", "top")
        YAxisLocation = Dict.setdefault("YAXisLocation", "left")
    else:
        XAxisLocation = Dict.setdefault("XAXisLocation", "bottom")
        YAxisLocation = Dict.setdefault("YAXisLocation", "left")

    if XYRange is not None:
        # The bounding box the image will fill.
            # Also the area that will be rendered.
        extent = [XYRange.XMin, XYRange.XMax, XYRange.YMin, XYRange.YMax] 
    else:
        if Coordinate in ["X1Y0", "YFirst", "Y0X1"]:
            Left, Right, Bottom, Top = 0.0, data.shape[1], 0.0, data.shape[0]
        else:
            Left, Right, Bottom, Top = 0.0, data.shape[0], 0.0, data.shape[1]
        if Origin in ["UpperLeft"]:
            extent = [Left, Right, Top, Bottom] # [Left, Right, Bottom, Top]
        else:
            extent = [Left, Right, Bottom, Top] # [Left, Right, Bottom, Top]
            # extent = [-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5]
        PixelNumX = Right - Left
        PixelNumY = Top - Bottom

    
    PlotHeightWidthRatio = None
    if PixelHeightWidthRatio in ["FillAx"]:
        PixelHeightWidthRatio = "auto"
        axPlot = ax
    elif PixelHeightWidthRatio in ["Auto", "auto"]:
        DataHeightWidthRatioMax = Dict.setdefault("DataHeightWidthRatioMax", 5.0)
        DataHeightWidthRatioMin = Dict.setdefault("DataHeightWidthRatioMin", 0.2)
        DataHeightWidthRatio = 1.0 * PixelNumY / PixelNumX
        if DataHeightWidthRatio > DataHeightWidthRatioMax:
            PlotHeightWidthRatio = DataHeightWidthRatioMax
            # PixelNumY * PixelHeight = PixelNumX * PixelWidth * 5.0
            PixelHeightWidthRatio = 5.0 * PixelNumX / PixelNumY
        elif DataHeightWidthRatio < DataHeightWidthRatioMin:
            PlotHeightWidthRatio = DataHeightWidthRatioMin
            # PixelNumY * PixelHeight = PixelNumX * PixelWidth * 0.2
            PixelHeightWidthRatio = 0.2 * PixelNumX / PixelNumY
        else:
            PlotHeightWidthRatio = DataHeightWidthRatio
            if data.shape[1] > data.shape[0]:
                HeightWidthRatio = 1.0 * data.shape[1] / data.shape[0]
                #axPlot = GetSubAx(ax, 0.5 - WidthHeightRatio / 2.0, 0.0, WidthHeightRatio, 1.0)
            else:
                HeightWidthRatio = 1.0 * data.shape[1] / data.shape[0]
                #axPlot = GetSubAx(ax, 0.0, 0.5 - HeightWidthRatio / 2.0, 1.0, HeightWidthRatio)
            PixelHeightWidthRatio = 1.0 # all pixels fit into the ax range.
        axPlot = ax
    else:
        axPlot = ax

    if Ticks is not None:
        if isinstance(Ticks, dict):
            axPlot.set_xticks(Ticks["XTicks"])
            axPlot.set_xticklabels(Ticks["XTicksStr"])
            axPlot.set_yticks(Ticks["YTicks"])
            axPlot.set_yticklabels(Ticks["YTicksStr"])
        elif Ticks in ["int", "Int"]:
            XTicks, XTicksStr = GetXTicksInt(round(Left), round(Right - 1.0))
            IsXTicksPrune = False
            if PlotHeightWidthRatio is not None:
                if PlotHeightWidthRatio >= DataHeightWidthRatioMax:
                    XTicks, XTicksStr = PruneTicks(XTicks, XTicksStr, MaxNum=3)
                    IsXTicksPrune = True
            axPlot.set_xticks(_OffsetTicks(XTicks, 0.5), labels=XTicksStr)
            if IsXTicksPrune:
                # RotateXTicks(axPlot, 45)
                pass
            
            YTicks, YTicksStr = GetYTicksInt(round(Bottom), round(Top - 1.0))
            IsYTicksPrune = False
            if PlotHeightWidthRatio is not None:
                if PlotHeightWidthRatio <= DataHeightWidthRatioMin:
                    YTicks, YTicksStr = PruneTicks(YTicks, YTicksStr, MaxNum=3)
                    IsYTicksPrune = True
            axPlot.set_yticks(_OffsetTicks(YTicks, 0.5), labels=YTicksStr)
            if IsYTicksPrune:
                # RotateXTicks(axPlot, 45)
                pass

        elif Ticks in ["float", "Float",]:
            SetXTicksFloat(axPlot, round(Left), round(Right))
            SetYTicksFloat(axPlot, round(Bottom), round(Top))
        elif Ticks in ["AccumulateOnX"]:
            ColNum = data.shape[1]
            XTicks, XTicksStr = GetXTicksInt(round(Left), round(Right - 1.0))
            for Index, XTick in enumerate(XTicks):
                XTicksStr[Index] = f"+{XTick}"
            axPlot.set_xticks(_OffsetTicks(XTicks, 0.5), labels=XTicksStr)
            YTicks, YTicksStr = GetYTicksInt(round(Bottom), round(Top - 1.0))
            for Index, YTick in enumerate(YTicks):
                YTicksStr[Index] = str(YTick * ColNum)
            axPlot.set_yticks(_OffsetTicks(YTicks, 0.5), labels=YTicksStr)
        else:
            raise Exception(Ticks)
        pass
    else:
        # SetTicks = Dict.Get("SetTicks", True)
        # if SetTicks is True:
        pass
    Interpolation = Dict.setdefault("Interpolation", 'none')
    if Format in ["svg"]:
        Interpolation = 'none'
    if Origin in ["UpperLeft"]:
        # if imshow receives origin='lower', this flip will be required
            # regardless of whether bottom > top or bottom < top in extent.
        if Coordinate in ["X1Y0", "Y0X1", "Fig"]:
            data = data[::-1, :] 
            dataInColor = dataInColor[::-1, :, :]
        elif Coordinate in ["X0Y1", "Y1X0"]:
            data = data[:, ::-1] 
            dataInColor = dataInColor[:, ::-1, :]
        else:
            raise Exception()
        
    # important note
    # imshow interprets dimension 0 as Y dimension, and dimension 1 as X dimension.
    if Coordinate in ["X1Y0", "Y0X1", "Fig"]:
        _DataInColor = dataInColor
    elif Coordinate in ["X0Y1", "Y1X0", "XLeftYTop"]:
        _DataInColor = dataInColor.transpose(1, 0, 2)
    else:
        raise Exception()
    axPlot.imshow(_DataInColor, extent=extent, origin='lower', aspect=PixelHeightWidthRatio, interpolation=Interpolation)    
    # SetMatplotlibParamToDefault() # ticks may disappear after imshow.
    SetAxisLocationForAx(axPlot, XAxisLocation, YAxisLocation)
    SetTitleAndLabelForAx(axPlot, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)
    return STATE.NORMAL

def RotateXTicks(ax, Angle):
    # Angle: positive: clockwise. negative: counter-clockwise.
    ax.tick_params(axis='x', labelrotation=Angle)

def RotateYTicks(ax, Angle):
    # Angle: positive: clockwise. negative: counter-clockwise.
    ax.tick_params(axis='y', labelrotation=Angle)

def PruneTicks(Ticks, TicksStr, MaxNum = 3):
    TickNum = len(Ticks)
    if TickNum <= MaxNum:
        return Ticks, TicksStr
    else:
        IndexList = np.asarray(range(MaxNum), dtype=np.float32) / (MaxNum - 1)
        IndexList = np.round(IndexList * (TickNum - 1)).astype(np.int32)
    return [Ticks[Index] for Index in IndexList], [TicksStr[Index] for Index in IndexList],

def _OffsetTicks(Ticks, Offset):
    for Index, Tick in enumerate(Ticks):
        Ticks[Index] += Offset
    return Ticks

def PlotActivityAndDistributionAlongTime(
        axes=None, Data=None, activityPlot=None, 
        Title=None,
        Save=True, SavePath=None,
    ):
    
    # activity: (BatchSize, StepNum, NeuronNum)
    BatchSize = Data.shape[0]
    TimeNum = Data.shape[1]
    ActivityNum = Data.shape[2]

    Data = DLUtils.ToNpArray(Data)
    activityPlot = DLUtils.ToNpArray(activityPlot).transpose(1, 0)
    if axes is None:
        #fig, axes = plt.subplots(nrows=1, ncols=2)
        fig, axes = CreateFigurePlt(2)
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = GetAx(axes, 0)
        ax2 = GetAx(axes, 1)

    dataInColor, dataMask = MaskOutInfOrNaN(activityPlot)
    PlotMatrixWithColorBar(
        ax1, activityPlot, dataInColor=dataInColor, 
        dataMask=dataMask, maskColor="Gray",
        XAxisLocation="top", PixelHeightWidthRatio="Auto",
        XLabel="Time Index", YLabel="Activity", Title="Visualization",
        Coordinate="Fig", Ticks="Int"
    )

    PlotMeanAndStdCurve(
        ax2, Xs=range(TimeNum), 
        #Mean=DLUtils.math.ReplaceNaNOrInfWithZero(np.nanmean(activity, axis=(0, 2))),
        #Std=DLUtils.math.ReplaceNaNOrInfWithZero(np.nanstd(activity, axis=(0, 2))),
        Mean = np.nanmean(Data, axis=(0, 2)),
        Std = np.nanstd(Data, axis=(0,2)),
        XLabel="Time Index", YLabel="Mean And Std", Title="Mean and Std Along Time",
        Save=False
    )
    if Title is not None:
        plt.suptitle(Title)
    
    SaveFigForPlt(Save, SavePath)
    return

def PlotData1D(Name, Data, SavePath, **Dict):
    XLabel = Dict.setdefault("XLabel", "Dimension 0")
    YLabel = Dict.setdefault("YLabel", "Dimension 0")
    fig, ax = DLUtils.plot.CreateFigurePlt(1)
    if isinstance(Data, float) or isinstance(Data, int):
        Data = np.asarray([Data])
    # turn 1D weight to 2D shape
    DimentionNum = DLUtils.GetDimensionNum(Data)
    if DimentionNum == 1:
        dataReshape, MaskReshape = DLUtils._1DTo2D(Data)
    else:
        MaskReshape = None
    PlotMatrix(
        ax=ax, data=dataReshape, dataMask=MaskReshape, XLabel=XLabel, YLabel=YLabel, 
        Coordinate="X1Y0", Ticks="AccumulateOnX", Origin="UpperLeft"
    )
    SaveFigForPlt(True, SavePath)

def PlotData2D(Name, Data, SavePath, **Dict):
    XLabel = Dict.setdefault("XLabel", "Dimension 0")
    YLabel = Dict.setdefault("YLabel", "Dimension 1")
    if isinstance(Data, float):
        Data = DLUtils.ToNpArray([[Data]])
    Data = DLUtils.ToNpArray(Data)
    assert len(Data.shape) == 2
    fig, ax = CreateFigurePlt(1)
    PlotMatrix(
        ax=ax, data=Data, XLabel=XLabel, YLabel=YLabel, Ticks="int", Coordinate="Y0X1", Origin="UpperLeft"
    )
    SaveFigForPlt(True, SavePath)
def PlotWeightChange(axes=None, weights=None):
    if axes is None:
        fig, axes = DLUtils.plot.CreateFigurePlt(3)
    return

def PlotDataAndDistribution1D(Data=None, axes=None, Name=None, Save=True, SavePath=None, **Dict):
    XLabel = Dict.setdefault("XLabel", "Dimension 0")
    YLabel = Dict.setdefault("YLabel", "Dimension 0")
    
    if isinstance(Data, float) or isinstance(Data, int):
        Data = np.asarray([Data])
    Data = DLUtils.ToNpArray(Data)
    DataReshape, MaskReshape = DLUtils._1DTo2D(Data)
    PlotDataAndDistribution2D(axes=axes, Data=DataReshape, Name=Name, Save=Save, SavePath=SavePath, 
        dataMask=MaskReshape, Ticks="AccumulateOnX", **Dict
    )


from enum import Flag, auto
class STATE(Flag):
    NORMAL = 0
    EQUAL_MIN_MAX = auto()
    BLUE = auto()

def PlotBatchDataAndDistribution1D(Data=None, axes=None, Name=None, Save=True, SavePath=None, **Dict):
    # Data: [BatchIndex, FeatureIndex]
    Coordinate = Dict.setdefault("Coordinate", "X1Y0")
    XLabel = Dict.setdefault("Feature Index")
    YLabel = Dict.setdefault("Batch Index")
    PlotDataAndDistribution2D(Data=Data, axes=axes, Name=Name, Save=Save, SavePath=SavePath, **Dict)

def PlotDataAndDistribution2D(Data=None, axes=None, Name=None, Save=True, SavePath=None, **Dict):

    dataMask = Dict.setdefault("dataMask", None)
    Ticks = Dict.setdefault("Ticks", "Int")
    ColorForEqualValues = Dict.setdefault("ColorForEqualValues", ParseColor("LightGray"))
    Coordinate = Dict.setdefault("Coordinate", "X1Y0")
    if Coordinate in ["X1Y0", "Y1X0"]:
        XLabel = Dict.setdefault("XLabel", "Dimension 1")
        YLabel = Dict.setdefault("YLabel", "Dimension 0")
    else:
        XLabel = Dict.setdefault("XLabel", "Dimension 0")
        YLabel = Dict.setdefault("YLabel", "Dimension 1") 
    
    if axes is None:
        fig, axes = DLUtils.plot.CreateFigurePlt(2)
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = GetAx(axes, 0)
        ax2 = GetAx(axes, 1)
    plt.suptitle(Name)

    if isinstance(Data, float):
        Data = np.array([[Data]])
    else:
        Data = DLUtils.ToNpArray(Data)

    _Dict = dict(Dict)
    _Dict["Title"] = Dict.setdefault("TitlePlot", "Visualization")
    State = DLUtils.plot.PlotMatrixWithColorBar(
        ax1, data=Data,
        XAxisLocation="top", PixelHeightWidthRatio="Auto",
        **_Dict
    )
    if STATE.EQUAL_MIN_MAX in State:
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        ax2.text(
            0.5, 0.5, f"All values equal to each other",
            ha='center', va='center' #, transform = ax.transAxes
        )
        ax2.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, color=ColorForEqualValues))
        ax2.axis('on')
    else:
        _Dict = dict(Dict)
        _Dict["Title"] = Dict.setdefault("TitleDistribution", "Visualization")
        DataFlat = DLUtils.FlattenNpArray(Data)
        BinStat = DLUtils.math.CalculateBinnedMeanStd(DataFlat)

        _Dict["XLabel"] = Dict.setdefault("XLabelDistribution", "Data Value")
        _Dict["YLabel"] = Dict.setdefault("YLabelDistribution", "Data Num")
        _Dict["YMin"] = 0
        PlotLineChart(ax2, BinStat.Xs, BinStat.YNum, **_Dict)
    SaveFigForPlt(Save, SavePath)

def Merge2Mask(mask1, mask2):
    if mask1 is None:
        if mask2 is None:
            raise Exception()

    if mask1 is not None and mask2 is not None:
        return mask1 * mask2
    elif mask1 is None and mask2 is not None:
        return mask2
    elif mask1 is not None and mask2 is None:
        return mask1
    else:
        return None

def InfOrNaNMask(data):
    mask = np.isfinite(data)
    # if round(np.sum(mask)) == mask.size:
    #     mask = None
    # else:
    #     return mask
    return mask

def MaskOutInfOrNaN(data):
    mask = InfOrNaNMask(data)
    if mask is None:
        return data, None
    else:
        return data[mask], mask

def PlotLineCv(img, points, line_color=(0,0,0), line_Width=2, line_type=4, BoundaryBox=[[0.0,0.0],[1.0,1.0]]):
    ResolutionX, ResolutionY = img.shape[0], img.shape[1]
    point_0 = GetIntCoords(points[0][0], points[0][1], BoundaryBox, ResolutionX, ResolutionY)
    point_1 = Getint_coords(points[1][0], points[1][1], BoundaryBox, ResolutionX, ResolutionY)
    cv.line(img, point_0, point_1, line_color, line_Width, line_type)

def GetRandomColors(Num=5):
    interval = 256 / Num
    pos_now = 0.0
    colors = []
    for i in range(Num):
        pos_now += interval
        colors.append(ColorWheel(int(pos_now)))
    return colors

GetColors = GetTypicalColors = GetRandomColors

def ColorWheel(Index): #生成横跨0-255个位置的彩虹颜色.  
    Index = Index % 255
    if Index < 85:
        return (Index * 3, 255 - Index * 3, 0)
    elif Index < 170:
        Index -= 85
        return (255 - Index * 3, 0, Index * 3)
    else:
        Index -= 170
        return (0, Index * 3, 255 - Index * 3)

def PlotImages(ImageList, ColNum):
    ImageNum = len(ImageList)
    RowNum, ColNum = ParseRowColNum(ImageNum)
    fig, axes = plt.subplots(RowNum, ColNum)

def cat_imgs_h(imgs, ColNum=10, space_Width=4):
    ''' Concat image horizontally with spacer '''
    space_col = np.ones([imgs.shape[1], space_Width, imgs.shape[3]], dtype=np.uint8) * 255
    imgs_cols = []

    img_num = img.shape[0]
    if img_num < ColNum:
        imgs = np.concatenate( [imgs, np.ones([imgs.shape[1], imgs.shape[2], imgs.shape[3]], dtype=np.uint8)*255] , axis=0)
    
    imgs_cols.append(space_col)
    for i in range(ColNum):
        imgs_cols.append(imgs[i])
        imgs_cols.append(space_col)
    return np.concatenate(imgs_cols, axis=0)

def cat_imgs(imgs, ColNum=10, space_Width=4): # images: [num, Width, Height, channel_num], np.uint8
    img_num = imgs.shape[0]
    RowNum = img_num // ColNum
    if img_num%ColNum>0:
        RowNum += 1

    space_row = np.zeros([space_Width, image.shape[0]*ColNum + space_Width*(ColNum+1), imgs.shape[3]], dtype=np.uint8)
    space_row[:,:,:] = 255

    imgs_rows = []

    imgs_rows.append(space_row)
    for row_index in range(RowNum):
        if (row_index+1)*ColNum>img_num:
            imgs_row = imgs[ row_index*ColNum : -1]
        else:
            imgs_row = imgs[ row_index*ColNum : (row_index+1)*ColNum ]
        imgs_row = cat_imgs_h(imgs_row, ColNum, spacer_size)        
        imgs_rows.append(imgs_row)
        #if row_index != RowNum-1:
        imgs_rows.append(space_row)

    return np.concatenate(imgs_rows, axis=1)

def ParseRowColNum(PlotNum, RowNum=None, ColNum=None):
    # ColNum: int. Column Number.
    if RowNum in ["Auto", "auto"]:
        RowNum = None
    if ColNum in ["Auto", "auto"]:
        ColNum = None
    if RowNum is None and ColNum is not None:
        RowNum = PlotNum // ColNum
        if PlotNum % ColNum > 0:
            RowNum += 1
        return RowNum, ColNum
    elif RowNum is not None and ColNum is None:
        ColNum = PlotNum // RowNum
        if PlotNum % RowNum > 0:
            ColNum += 1
        return RowNum, ColNum
    elif RowNum is None and ColNum is None:
        if PlotNum <= 3:
            return 1, PlotNum
        ColNum = round(PlotNum ** 0.5)
        if ColNum == 0:
            ColNum = 1
        RowNum = PlotNum // ColNum
        if PlotNum % ColNum > 0:
            RowNum += 1
        return RowNum, ColNum
    else:
        if PlotNum != RowNum * ColNum:
            raise Exception('PlotNum: %d != RowNum %d x ColumnNum %d'%(PlotNum, RowNum, ColNum))
        else:
            return RowNum, ColNum

def CreateFigurePlt(PlotNum=1, RowNum=None, ColNum=None, Width=None, Height=None, Size="Small"):
    RowNum, ColNum = ParseRowColNum(PlotNum, RowNum, ColNum)
    if Width is None and Height is None:
        if Size in ["Small", "S"]:
            AxSize = 5.0
        elif Size in ["Medium", "M"]:
            AxSize = 7.5
        elif Size in ["Large", "L"]:
            AxSize = 10.0
        else:
            raise Exception(Size)
        Width = ColNum * AxSize # inches
        Height = RowNum * AxSize # inches
    elif Width is not None and Height is not None:
        pass
    else:
        raise Exception()
    plt.close()
    fig, axes = plt.subplots(nrows=RowNum, ncols=ColNum, figsize=(Width, Height))
    return fig, axes
CreateFigure = CreateCanvasPlt = CreateFigurePlt

def SplitAxRightLeft(ax):
    axLeft = GetSubAx(ax, -0.55, 0.0, 1.0, 1.0)
    axRight = GetSubAx(ax, 0.55, 0.0, 1.0, 1.0)
    return axLeft, axRight

def GetAxRowColNum(axes):
    if isinstance(axes, np.ndarray):
        Shape = axes.shape
        if len(Shape)==1:
            return Shape[0], 1
        elif len(Shape)==2:
            return Shape[0], Shape[1]
        else:
            raise Exception()
    # elif isinstance(axes, mpl.axes._subplots.AxesSubplot): # Shape is [1, 1]:
    #     return 1, 1
    else:
        return 1, 1
        # raise Exception()
def GetAx(axes, Index=None, RowIndex=None, ColIndex=None):
    RowNum, ColNum = GetAxRowColNum(axes)

    if Index is not None:
        RowIndex = Index // ColNum
        ColIndex = Index % ColNum
    
    if RowNum > 1 or ColNum > 1:
        if RowNum==1 or ColNum==1:
            if RowIndex==0 or RowIndex is None:
                if isinstance(ColIndex, int):
                    return axes[ColIndex]
                else:
                    raise Exception()
            elif ColIndex==0 or ColIndex is None:
                if isinstance(RowIndex, int):
                    return axes[RowIndex]
                else:
                    raise Exception()
            else:
                raise Exception()
        else:
            if RowIndex is None:
                raise Exception()
            if ColIndex is None:
                raise Exception()
            return axes[RowIndex, ColIndex]
    elif RowNum == 1 and ColNum == 1:
        if not (RowIndex is None or RowIndex==0):
            raise Exception()
        if not (RowIndex is None or RowIndex==0):
            raise Exception()
        return axes
    else:
        raise Exception()

def PlotLineChart(ax=None, Xs=None, Ys=None,
        Title="Undefined", Label=None,
        Color="Black", LineWidth=2.0, 
        Save=False, SavePath=None, **Dict
    ):
    XLabel = Dict.setdefault("XLabel", "X")
    YLabel = Dict.setdefault("YLabel", "Y")
    XTicks = Dict.setdefault("XTicks", "Float")
    YTicks = Dict.setdefault("YTicks", "Float")

    if ax is None:
        fig, ax = CreateFigurePlt()
    Color = (Color)
    ax.plot(Xs, Ys, color=Color, linewidth=LineWidth, label=Label)

    SetTicks = Dict.get("SetTicks", True)
    if SetTicks:
        SetXTicksAndRange(ax, Xs=Xs, **Dict)
        SetYTicksAndRange(ax, Ys=Ys, **Dict)

    SetXYLabelForAx(ax, XLabel, YLabel)
    SetTitleForAx(ax, Title)
    SaveFigForPlt(Save, SavePath)
    return ax

def PlotMultiLineChart(
        ax=None, XsList=None, YsList=None,
        Title="Untitled", Labels=None,
        ColorList=None, LineWidth=2.0,
        Save=False, SavePath=None, **Dict
    ):
    if ax is None:
        fig, ax = CreateFigurePlt()
    XLabel = Dict.setdefault("XLabel", "X")
    YLabel = Dict.setdefault("YLabel", "Y")

    Index = 0
    LineNum = len(XsList)
    if ColorList is None:
        ColorList = GenerateColors(LineNum)
    elif isinstance(ColorList, str):
        ColorList = [ColorList for _ in range(LineNum)]
    else:
        assert len(ColorList) == LineNum
    for Xs, Ys in zip(XsList, YsList):
        if Labels is None:
            Label = None
        else:
            Label = Labels[Index]
        PlotLineChart(
            ax, Xs=Xs, Ys=Ys,
            Label=Label,
            Color=ColorList[Index],
            LineWidth=LineWidth,
            Save=False,
            SetTicks=False
        )
        Index += 1

    SetTicks = Dict.setdefault("SetTicks", True)
    if SetTicks:
        XTicks = Dict.setdefault("XTicks", "float")
        YTicks = Dict.setdefault("YTicks", "float")
        SetXTicksAndRange(ax, Xs=XsList, DimNum=2, **Dict)
        SetYTicksAndRange(ax, Ys=YsList, DimNum=2, **Dict)
    SetXYLabelForAx(ax, XLabel, YLabel)
    SetTitleForAx(ax, Title)

    if Labels is not None:
        ax.legend(loc="upper right")
    SaveFigForPlt(Save, SavePath)
    return ax

def PlotMultiLineChartWithSameXs(ax=None, Xs=None, YsDict=None,
        Title="Undefined", Color="Auto", LineWidth=1.0,
        Save=False, SavePath=None, **Dict
    ):
    XTicks = Dict.setdefault("XTicks", "float")
    YTicks = Dict.setdefault("YTicks", "float")
    XLabel = Dict.setdefault("XLabel", "X")
    YLabel = Dict.setdefault("YLabel", "Y")

    if ax is None:
        fig, ax = CreateFigurePlt()
    PlotNum = len(YsDict)
    if Color in ["Auto", "auto"]:
        Colors = GenerateColors(PlotNum)

    for Index, (Name, Ys) in enumerate(YsDict.items()):
        ax.plot(
            Xs, Ys, color=Colors[Index],
            linewidth=LineWidth,
            label=Name,
        )
    ax.legend(loc="upper right")
    if XTicks in ["Float"]:
        Min, Max = np.nanmin(Xs), np.nanmax(Xs)
        SetXTicksFloat(ax, Min, Max)
        SetXRangeMinMaxFloat(ax, Min, Max, Pad=0.0)

    if YTicks in ["Float"]:
        Min = []
        Max = []
        for Ys in YsDict.values():
            Min.append(np.nanmin(Ys))
            Max.append(np.nanmax(Ys))
        Min, Max = np.nanmin(Min), np.nanmax(Max)
        SetYTicksFloat(ax, Min, Max)
        SetYRangeMinMaxFloat(ax, Min, Max, Pad=0.0)

    SetXYLabelForAx(ax, XLabel, YLabel)
    SetTitleForAx(ax, Title)
    SaveFigForPlt(Save, SavePath)
    return ax

def PlotBoxPlot(
        ax=None, data=None,
        Norm2Sum1=False, Color="Black", 
        XLabel=None, YLabel=None, Title=None,
        Save=False, SavePath=None,
        **kw
    ):
    data = DLUtils.EnsureFlat(data)
    ax.boxplot(data)
    if Title is not None:
        ax.set_title(Title)
    
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)

def PlotHistogram(
        ax=None, data=None,
        Norm2Sum1=False, Color="Black", 
        XLabel=None, YLabel=None, Title=None,
        Save=False, SavePath=None,
        **kw
    ):

    if data is None or data.size == 0:
        ax.text(
            0.5, 0.5, "No Plottable Data", 
            ha='center', va='center',
        )
    else:
        data = DLUtils.EnsureFlat(data)
        data = DLUtils.math.RemoveNaNOrInf(data)
        ax.hist(data, density=Norm2Sum1)
        if Title is not None:
            ax.set_title(Title)
    
    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)

def SetXYLabelForAx(ax, XLabel, YLabel):
    if XLabel is not None:
        ax.set_xlabel(XLabel)
    if YLabel is not None:
        ax.set_ylabel(YLabel)

def SetTitleForAx(ax, Title):
    if Title is not None:
        ax.set_title(Title)

def GetSubAx(ax, Left, Bottom, Width, Height):
    return ax.inset_axes([Left, Bottom, Width, Height]) # left, bottom, width, height. all are ratios to sub-canvas of ax.

def ParseColorBarOrientation(Orientation):
    if Orientation is None:
        Orientation = "Vertical"
    if Orientation in ["Horizontal", "horizontal"]:
        Orientation = "horizontal"
    elif Orientation in ["Vertical", "vertical"]:
        Orientation = "vertical"
    else:
        raise Exception(Orientation)  
    return Orientation

def PlotColorBarInSubAx(ax, ColorMap="jet", Method="MinMax", Orientation="Vertical", Location=None, **kw):
    # Location: [Left, Bottom, Width, Height]
    Orientation = ParseColorBarOrientation(Orientation)
    Location = ParseColorBarLocation(Location, Orientation)
    axSub = GetSubAx(ax, *Location)
    PlotColorBarInAx(axSub, ColorMap=ColorMap, Method=Method, Orientation=Orientation, **kw)

def ParseColorBarLocation(Location, Orientation):
    if Location is None:
        if Orientation in ["vertical"]:
            Location = [1.05, 0.0, 0.05, 1.0]
        elif Orientation in ["horizontal"]:
            Location = [0.0, -0.10, 1.0, 0.05]
        else:
            raise Exception()
    elif isinstance(Location, str):
        raise Exception(Location)
    else:
        pass
    return Location

def PlotColorBarInAx(ax, ColorMap="jet", Method="MinMax", Orientation="Vertical", **Dict):
    Orientation = DLUtils.ToLowerStr(Orientation)

    ColorForEqualValues = Dict.setdefault("ColorForEqualValues", ParseColor("LightGray"))
    if Orientation not in ["horizontal", "vertical"]:
        raise Exception(Orientation)

    if Method in ["MinMax", "GivenMinMax"]:
        Min, Max = Dict["Min"], Dict["Max"]
        if Min == Max or not np.isfinite(Min) or not np.isfinite(Max):
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.text(
                0.5, 0.5, f"All values equal to {DLUtils.Float2StrDisplay(Min)}", 
                rotation=-90.0 if Orientation == "vertical" else 0.0,
                ha='center', va='center' #, transform = ax.transAxes
            )
            ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, color=ColorForEqualValues))
            ax.set_xticks([])
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels([DLUtils.Float2StrDisplay(Min) for _ in range(6)])
            SetAxisLocationForAx(ax, "bottom", "right")
            ax.axis('on')
            return STATE.EQUAL_MIN_MAX

        # if Min == Max:
        #     a = 1
        ColorMap = ParseColorMapPlt(ColorMap)
        Norm = mpl.colors.Normalize(vmin=Min, vmax=Max)

        ColorBar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=Norm, cmap=ColorMap),
            cax=ax,
            #ticks=Ticks,
            orientation=Orientation
        )

        Title = Dict.get("Title")
        if Title is not None:
            ColorBar.set_label(Title, loc="center")

        Ticks = Dict.setdefault("Ticks", "auto")
        SetColorBarTicks(ColorBar, Ticks, Orientation, Min=Min, Max=Max)

        if Orientation in ["horizontal"]: # Avoid Label overlapping
            plt.setp(
                ColorBar.ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor"
            )
        return STATE.NORMAL
    else:
        raise Exception()

def CalculateTickIntervalFloat(Min, Max):
    Range = Max - Min
    if Range == 0.0:
        return 0.0, None, None


    Log = round(math.log(Range, 10))
    Base = 1.0
    Interval = Base * 10 ** Log
    TickNum = Range / Interval

    while not 2.5 <= TickNum <= 6.5:
        if TickNum > 6.0:
            Base, Log = NextIntervalUp(Base, Log)
        elif TickNum < 3.0:
            Base, Log = NextIntervalDown(Base, Log)
        else:
            break
        Interval = Base * 10 ** Log
        TickNum = Range / Interval
    return Interval, Base, Log

def CalculateTickIntervalInt(Min, Max):
    Range = Max - Min
    if Range <= 0:
        return 1
    elif Range <= 5:
        return 2

    Log = round(math.log(Range, 10))
    Base = 1.0
    Interval = Base * 10 ** Log
    TickNum = Range / Interval

    while not 2.5 <= TickNum <= 6.5:
        if TickNum > 6.0:
            Base, Log = NextIntervalUp(Base, Log)
        elif TickNum < 3.0:
            Base, Log = NextIntervalDown(Base, Log)
        else:
            break
        Interval = Base * 10 ** Log
        TickNum = Range / Interval
    return round(Interval)

def NextIntervalUp(Base, Log):
    if Base == 1.0:
        Base = 2.0
    elif Base == 2.0:
        Base = 5.0
    elif Base == 5.0:
        Base = 1.0
        Log += 1
    else:
        raise Exception()
    return Base, Log

def NextIntervalDown(Base, Log):
    if Base == 1.0:
        Base = 5.0
        Log -= 1
    elif Base == 2.0:
        Base = 1.0
    elif Base == 5.0:
        Base = 2.0
    else:
        raise Exception()
    return Base, Log

def CalculateTicksFloat(Method="Auto", Min=None, Max=None, **kw):
    if Method in ["Auto", "auto"]:
        Range = Max - Min
        Interval, Base, Log = CalculateTickIntervalFloat(Min, Max)
        Ticks = []
        Ticks.append(Min)
        Tick = math.ceil(Min / Interval) * Interval
        if Tick - Min < 0.1 * Interval:
            pass
        else:
            Ticks.append(Tick)
        while Tick < Max:
            Tick += Interval
            if Max - Tick < 0.1 * Interval:
                break
            else:
                Ticks.append(Tick)
        Ticks.append(Max)
    elif Method in ["Linear"]:
        Num = kw["Num"]
        Ticks = np.linspace(Min, Max, num=Num)
    else:
        raise Exception()

    TicksStr = []
    if 1 <= Log <= 2:
        TicksStr = list(map(lambda tick:str(int(tick)), Ticks))
    elif Log == 0:
        TicksStr = list(map(lambda tick:'%.1f'%tick, Ticks))
    elif Log == -1:
        TicksStr = list(map(lambda tick:'%.2f'%tick, Ticks))
    elif Log == -2:
        TicksStr = list(map(lambda tick:'%.3f'%tick, Ticks))
    else:
        TicksStr = list(map(lambda tick:'%.2e'%tick, Ticks))
    return Ticks, TicksStr

def CalculateTicksInt(Method="Auto", Min=None, Max=None, **Dict):
    assert isinstance(Min, int) and isinstance(Max, int)
    if Method in ["Auto", "auto"]:
        Range = Max - Min
        Interval = CalculateTickIntervalInt(Min, Max)
        Ticks = []
        Ticks.append(Min)
        Tick = round(math.ceil(1.0 * Min / Interval) * Interval)
        if Tick - Min < 0.1 * Interval:
            pass
        else:
            Ticks.append(Tick)
        while Tick < Max:
            Tick += Interval
            if Max - Tick < 0.1 * Interval:
                break
            else:
                Ticks.append(Tick)
        Ticks.append(Max)
    elif Method in ["Linear"]:
        Num = Dict["Num"]
        Ticks = np.rint(np.linspace(Min, Max, num=Num))
    else:
        raise Exception()

    TicksStr = list(map(lambda tick:str(int(tick)), Ticks))
    Offset = Dict.setdefault("Offset", 0.0)
    if Offset != 0.0:
        Ticks = list(map(lambda tick:tick + Offset, Ticks))
    return Ticks, TicksStr

def SetColorBarTicks(ColorBar, Ticks, Orientation, Min=None, Max=None, **kw):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    # if Min == Max:
    #     if Min == 0.0:
    #         Min, Max = -1.0, 1.0
    #     elif Min > 0.0:
    #         Min, Max = 0.5 * Min, 1.5 * Max
    #     else:  
    #         Min, Max = - 0.5 * Min, - 1.5 * Max
    
    if Ticks in ["Auto", "auto"]:
        #Ticks = np.linspace(Min, Max, num=5)
        Ticks, TicksStr = CalculateTicksFloat("Auto", Min, Max)
        ColorBar.set_ticks(Ticks)
        if Orientation in ["vertical"]:
            ColorBar.ax.set_yticklabels(TicksStr)
        elif Orientation in ["horizontal"]:
            ColorBar.ax.set_xticklabels(TicksStr)
        else:
            raise Exception()
    elif Ticks is None: # No Ticks
        pass
    else:
        raise Exception(Ticks)

def GetColorBarTicks(Ticks, Orientation, Min=None, Max=None, **kw):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min == Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        elif Min > 0.0:
            Min, Max = 0.5 * Min, 1.5 * Max
        else:  
            Min, Max = - 0.5 * Min, - 1.5 * Max
    
    if Ticks in ["Auto", "auto"]:
        Ticks, TicksStr = CalculateTicksFloat("Auto", Min, Max)
        return Ticks, TicksStr
    elif Ticks is None: # No Ticks
        pass
    else:
        raise Exception(Ticks)

def SetXTicksFloatFromData(ax, data, **kw):
    if isinstance(data, np.ndarray):
        Min = np.nanmin(data)
        Max = np.nanmax(data)
    elif isinstance(data, list):
        Min = min(
            list(map(
                lambda x:np.nanmin(x), data
            ))
        )
        Max = max(
            list(map(
                lambda x:np.nanmax(x), data
            ))
        )
    else:
        raise Exception(type(data))
    SetXTicksFloat(ax, Min, Max, **kw)

def SetYTicksFloatFromData(ax, data, **kw):
    if isinstance(data, np.ndarray):
        Min = np.nanmin(data)
        Max = np.nanmax(data)
    elif isinstance(data, list):
        Min = min(
            list(map(
                lambda x:np.nanmin(x), data
            ))
        )
        Max = max(
            list(map(
                lambda x:np.nanmax(x), data
            ))
        )
    else:
        raise Exception(type(data))
    SetYTicksFloat(ax, Min, Max, **kw)

def SetXTicksFloat(ax, Min, Max, Method="Auto", Rotate45=True):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min == Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        elif Min > 0.0:
            Min, Max = 0.5 * Min, 1.5 * Max
        else:  
            Min, Max = - 0.5 * Min, - 1.5 * Max

    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    ax.set_xticks(Ticks)
    ax.set_xticklabels(TicksStr)

    if Rotate45: # Avoid Label overlapping
        plt.setp(
            ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor"
        )
    return Ticks, TicksStr

def SetYTicksFloat(ax, Min, Max, Method="Auto"):
    if not np.isfinite(Min) or not np.isfinite(Max):
        Min, Max = -5.0, 5.0
    if Min==Max:
        if Min == 0.0:
            Min, Max = -1.0, 1.0
        else:
            Min, Max = 0.5 * Min, 1.5 * Max
    Ticks, TicksStr = CalculateTicksFloat(Method, Min, Max)
    ax.set_yticks(Ticks)
    ax.set_yticklabels(TicksStr)
    return Ticks, TicksStr


import imageio
def ImageFiles2GIFFile(ImageFiles, TimePerFrame=0.5, SavePath=None):
    Frames = []
    for ImageFile in ImageFiles:
        Frames.append(imageio.imread(ImageFile))
    imageio.mimsave(SavePath, Frames, 'GIF', duration=TimePerFrame)
    return

ImageFiles2GIF = ImageFiles2GIFFile

def ImagesNp2GIFFile():
    return
ImagesNp2GIF = ImagesNp2GIFFile

import imageio
def create_gif(image_list, gif_name, duration = 1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间, 单位s
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

'''
from PIL import Image 
from images2gif import writeGif
def create_gif_2():
    outfilename = "my.gif" # 转化的GIF图片名称          
    filenames = []         # 存储所需要读取的图片名称
    for i in range(100):   # 读取100张图片
        filename = path    # path是图片所在文件，最后filename的名字必须是存在的图片 
        filenames.append(filename)              # 将使用的读取图片汇总
    frames = []
    for image_name in filenames:                # 索引各自目录
        im = Image.open(image_name)             # 将图片打开，本文图片读取的结果是RGBA格式，如果直接读取的RGB则不需要下面那一步
        im = im.convert("RGB")                  # 通过convert将RGBA格式转化为RGB格式，以便后续处理 
        im = np.array(im)                       # im还不是数组格式，通过此方法将im转化为数组
        frames.append(im)                       # 批量化
    writeGif(outfilename, frames, duration=0.1, subRectangles=False) # 生成GIF，其中durantion是延迟，这里是1ms
'''

def PlotDistribution1D(data, ):
    data, shape = DLUtils.FlattenData(data)

from scipy.stats import gaussian_kde
def PlotGaussianDensityCurve(
        ax=None, data=None, KernelStd="auto", 
        Save=False, SavePath=None,
        **kw
    ): # For 1D data
    if ax is None:
        fig, ax = plt.subplots()
    data = DLUtils.EnsureFlat(data)
    statistics = DLUtils.math.NpArrayStatistics(data)

    if isinstance(KernelStd, float):
        pass
    elif isinstance(KernelStd, str):
        if KernelStd in ["Auto", "auto"]:
            KernelStd = "Ratio2Std"
        if KernelStd in ["Ratio2Range"]:
            Ratio = kw.setdefault("Ratio", 0.02)
            KernelStd = Ratio * (statistics.Max - statistics.Min)
        elif KernelStd in ["Ratio2Std"]:
            Ratio = kw.setdefault("Ratio", 0.02)
            KernelStd = Ratio * statistics.Std
        else:
            raise Exception()
    else:
        raise Exception(type(KernelStd))
    
    if KernelStd == 0.0:
        KernelStd = 1.0
    try:
        DensityCurve = gaussian_kde(data, bw_method=KernelStd)
        ax.plot(data, DensityCurve(data))
    except Exception:
        # Error when calculating bandwidth for repeating values.
        sns.kdeplot(data)
    # DensityCurve.covariance_factor = lambda : .25
    # DensityCurve._compute_covariance()
    SaveFigForPlt(Save, SavePath)
    return ax

PlotDensityCurve = PlotGaussianDensityCurve

def GetAxRangeMinMax(Min, Max, Pad=0.0):
    Range = Max - Min
    Left = Min - Pad * Range
    Right = Max + Pad * Range
    return Left, Right

def GetRangeMinMaxInt(Min, Max, Pad=0.05):
    Left, Right = GetRangeMinMaxFloat(Min, Max, Pad)
    Left = math.floor(Left)
    Right = math.ceil(Right)
    return Left, Right

def GetRangeMinMaxFloat(Min, Max, Pad=0.05):
    Range = Max - Min
    Left = Min - Pad * Range
    Right = Max + Pad * Range
    if Left == Right:
        if Left > 0.0:
            Left, Right = Left * 0.5, Right * 1.5
        elif Left < 0.0:
            Left, Right = Left * 1.5, Right * 0.5
        else:
            Left, Right = -1.0, 1.0
    return Left, Right

def SetXRangeMinMaxFloat(ax, Min, Max, Pad=0.05):
    Left, Right = GetRangeMinMaxFloat(Min, Max, Pad)
    ax.set_xlim(Left, Right)

def SetYRangeMinMaxFloat(ax, Min, Max, Pad=0.05):
    Left, Right = GetRangeMinMaxFloat(Min, Max, Pad)
    ax.set_ylim(Left, Right)

def SetXRangeMinMaxInt(ax, Min, Max, Pad=0.05):
    Left, Right = GetRangeMinMaxInt(Min, Max, Pad)
    ax.set_xlim(Left, Right)

def SetYRangeMinMaxInt(ax, Min, Max, Pad=0.05):
    Left, Right = GetRangeMinMaxInt(Min, Max, Pad)
    ax.set_ylim(Left, Right)

def SetAxisLocationForAx(ax, XAxisLocation=None, YAxisLocation=None):
    if XAxisLocation is not None:
        if XAxisLocation in ["top"]:
            ax.xaxis.tick_top()
        elif XAxisLocation in ["bottom"]:
            ax.xaxis.tick_bottom()
        else:
            raise Exception(XAxisLocation)
    if YAxisLocation is not None:
        if YAxisLocation in ["right"]:
            ax.yaxis.tick_right()
        elif YAxisLocation in ["left"]:
            ax.yaxis.tick_left()
        else:
            raise Exception(YAxisLocation)

def SetTitleAndLabelForAx(ax, XLabel=None, YLabel=None, Title=None):
    if XLabel is not None:
        ax.set_xlabel(XLabel)
    if YLabel is not None:
        ax.set_ylabel(YLabel)
    if Title is not None:
        ax.set_title(Title)

def SaveFigForPlt(Save=True, SavePath=None, Fig=None):
    if SavePath is not None:
        Save = True
    if SavePath is None and Save is True:
        raise Exception()
    if Save:
        if Fig is not None:
            Fig.savefig(SavePath)
        else:
            DLUtils.EnsureFileDir(SavePath)
            plt.tight_layout()
            # plt.savefig(SavePath, format="svg")
            plt.savefig(SavePath)
            # plt.close() # is closed, plt.show() will not function normally.

def CompareDensityCurve(data1, data2, Name1, Name2, Save=True, SavePath=None):
    fig, ax = plt.subplots()

    data1 = DLUtils.ToNpArray(data1)
    data2 = DLUtils.ToNpArray(data2)

    Min = min(np.min(data1), np.min(data2))
    Max = max(np.max(data1), np.max(data2))

    DLUtils.plot.PlotGaussianDensityCurve(ax, data1)
    DLUtils.plot.PlotGaussianDensityCurve(ax, data2)

    SetXRangeMinMaxFloat(ax, Min, Max)

    if SavePath is None:
        SavePath = DLUtils.GetMainSaveDir() + "%s-%s-GaussianKDE.png"%(Name1, Name2)

    DLUtils.plot.SaveFigForPlt(Save, SavePath)

def PlotMeanAndStdCurve(
        ax=None, Xs=None, Mean=None, Std=None,
        LineWidth=2.0, Color="Black", XTicks="Float",
        Title=None, XLabel=None, YLabel=None,
        Save=False, SavePath=None, **kw
    ):
    if ax is None:
        fig, ax = plt.subplots()

    if XTicks in ["Int"]:
        XTicks, XTicksStr = SetXTicksInt(ax, min(Xs), max(Xs))
    elif XTicks in ["Float"]:
        XTicks, XTicksStr = SetXTicksFloat(ax, min(Xs), max(Xs))

    Mean = DLUtils.ToNpArray(Mean)
    Std = DLUtils.ToNpArray(Std)
    Color = (Color)
    Y1 = Mean - Std
    Y2 = Mean + Std

    YTicks, YTicksStr = SetYTicksFloat(ax, np.nanmin(Y1), np.nanmax(Y2))

    if DLUtils.math.IsAllNaNOrInf(Y1) and DLUtils.math.IsAllNaNOrInf(Y2) and DLUtils.math.IsAllNaNOrInf(Mean):
        ax.text(
            (XTicks[0] + XTicks[-1]) / 2.0,
            (YTicks[0] + YTicks[-1]) / 2.0,
            "All values are NaN or Inf", ha='center', va='center',
        )
    else:
        PlotLineChart(
            ax, Xs, Mean, Color=Color, LineWidth=LineWidth,
            Save=False, **kw
        )
        ax.fill_between(Xs, Y1, Y2)

    SetTitleAndLabelForAx(ax, XLabel, YLabel, Title)
    SaveFigForPlt(Save, SavePath)

def SetXAxisLocationForAx(ax, XAXisLocation):
    if XAXisLocation in ["top"]:
        ax.xaxis.tick_top()
    else:
        raise Exception()

# def PlotTrajectory(ax, XYs, Color="Black"):
#     # XYs: [StepNum, (x, y)]
#     StepNum = XYs.shape[0] - 1
#     Color=(Color)
#     for StepIndex in range(StepNum):
#         DLUtils.plot.PlotArrowFromVertexPairsPlt(
#             ax, XYs[StepIndex, :], XYs[StepIndex+1, :],
#             Color=Color,
#             SizeScale=0.1,
#         )
#     return ax

def ColorImage2GrayImage(images, ColorAxis=-1):
    R = np.take(images, 0, ColorAxis)
    G = np.take(images, 1, ColorAxis)
    B = np.take(images, 2, ColorAxis)
    return 0.30 * R + 0.59 * G + 0.11 * B   

def Norm2Image(data):
    assert len(data.shape)==2 or len(data.shape)==3 # Gray or Color Image

    Min, Max = np.min(data), np.max(data)
    if Min==Max:
        data = np.full_like(data, 128.0)
    else:
        data = (data - Min) / (Max - Min) * 255.0
    data = np.around(data)
    data = data.astype(np.uint8)
    if len(data.shape)==2: # collapsed gray image
        data = np.stack([data, data, data], axis=2)
    return data

def PlotImageLFloat01(Image, SavePath):
    NpArray2ImageFileGreyFloat01PIL(Image, SavePath)
PlotGreyImage = PlotImageGreyFloat = PlotImageGreyFloat01 = PlotImageLFloat01

def PlotImageRGBFloat01(Image, SavePath):
    #NpArray2ImageFile(Image, SavePath)
    raise Exception()
    return
PlotColorImage = PlotImageRGBFloat = PlotImageRGBFloat01

def PlotImage(Image, SavePath):
    Shape = Image.shape
    if len(Shape) == 2:
        NpArray2ImageFileGreyFloat01PIL(Image, SavePath)
    elif len(Shape) == 3:
        PlotImageRGBFloat01(Image, SavePath)
    else:
        raise Exception()
from PIL import Image as Im
def NpArray2ImageFilePIL(Data, SavePath=None):
    # image : np.ndarray, with dtype np.uint8
    ImagePIL = Im.fromarray(Data)
    DLUtils.EnsureFileDir(SavePath)
    ImagePIL.save(SavePath)

def NpArray2ImageFileGreyFloat01PIL(Image, ImageFilePath):
    # Image: (Height, Width). float within range [0.0, 1.0].
    DLUtils.EnsureFileDir(ImageFilePath)
    ImageFilePath = DLUtils.file.RenameFileIfExists(ImageFilePath)
    ImageInt255 = (Image * 255.999).astype(np.uint8)
    _Image = Im.fromarray(ImageInt255)
    _Image = _Image.convert('L')
    _Image.save(ImageFilePath)

import matplotlib.image
def NpArray2ImageFileFloat01MPL(Data, ImageFilePath):
    # requires Data is float, values in [0.0, 1.0].
    # Data: (Height, Width, ChannelNum)
        # ChannelNum: RGB or RGBA
    DLUtils.EnsureFileDir(ImageFilePath)
    ImageFilePath = DLUtils.file.RenameFileIfExists(ImageFilePath)
    matplotlib.image.imsave(ImageFilePath, Data)
    return
NpArray2ImageFile = NpArray2ImageFileFloat01MPL
NpArray2ImageFileFloat = NpArray2ImageFileFloat01MPL
NpArray2ImageFileInt255 = NpArray2ImageFileFloat01MPL

def Tensor2ImageFile(Data, SavePath):
    Data = DLUtils.TorchTensor2NpArray(Data)
    return NpArray2ImageFile(Data, SavePath)

def PlotExampleImage(Images, PlotNum=10, SaveDir=None, SaveName=None):
    PlotIndices = DLUtils.RandomSelect(range(Images.shape[0]), PlotNum)
    PlotNum = len(PlotIndices)
    for Index in range(PlotNum):
        ImageIndex = PlotIndices[Index]
        Image = Images[ImageIndex]
        Image = Norm2Image(Image)
        NpArray2ImageFile(Image, SaveDir + SaveName + "-No.%d.png"%ImageIndex)

# https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes
# answered on Feb 2, 2017 at 23:10 by Mad Physicist
# till 2022.11 matplotlib does not have a method for this function
from operator import sub
def GetAspectOfAx(ax, Positive=True):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    aspect = disp_ratio / data_ratio

    if Positive:
        if aspect < 0.0:
            aspect = - aspect
    return aspect