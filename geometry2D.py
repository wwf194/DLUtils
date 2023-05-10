import math
import cmath
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

T = np.array([[0, -1], [1, 0]])
# @Reference https://blog.csdn.net/daming98/article/details/79561777
# @Reference https://segmentfault.com/a/1190000004457595?ref=myread
def HasIntersection(A, B, C, D, Threshold=1.0e-9): 
    # A, B, C, D np.ndarray [PointNum, (x, y)]
    # Judge whether segment AB intersects CD.
    AC = C - A
    AD = D - A
    BC = C - B
    BD = D - B
    return (np.cross(AC, AD) * np.cross(BC, BD) <= Threshold) * (np.cross(AC, BC) * np.cross(AD, BD) <= Threshold)

def DirectionsLengths2XYsNp(Directions, Lengths):
    return np.stack([np.cos(Directions) * Lengths, np.sin(Directions) * Lengths], axis=1)
DirectionsLengths2XYs = DirectionsLengths2XYsNp

def FlipAroundNorms(Directions, Norms):
    # Norms: must be of length 1. with shape [PointNum, 2]
    Projections = np.sum(Directions * Norms, axis=1)
    Verticals = Directions - Projections
    return Verticals - Projections

def FlipAroundNormsAngle(Directions, Norms):
    return 2 * Norms - Directions + np.pi

def RectangleAContainsRectangleB(RectangleA, RectangelB, Strict=False):
    # RectangleA: list. (XMin, YMin, XMax, YRange)
    # RectangleB: list. (XMin, YMin, XMax, YRange)
    if not Strict:
        Condition1 = RectangleA[0] <= RectangelB[0]
        Condition2 = RectangleA[1] <= RectangelB[1]
        Condition3 = RectangleA[2] >= RectangelB[2]
        Condition4 = RectangleA[3] >= RectangelB[3]
        return Condition1 and Condition2 and Condition3 and Condition4
    else:
        Condition1 = RectangleA[0] < RectangelB[0]
        Condition2 = RectangleA[1] < RectangelB[1]
        Condition3 = RectangleA[2] > RectangelB[2]
        Condition4 = RectangleA[3] > RectangelB[3]
        return Condition1 and Condition2 and Condition3 and Condition4

def SegmentInterceptedByLine(P1, P2, Q1, Q2, P1P2=None):
    # Judge whether p1p2 will be intercepted by ref1-ref2.
    # If yes, gives ratio: p1-InterceptionPoint / ref1-ref2
    # If no, gives ratio 1.0.
    PointNum = P1.shape[0]
    if P1P2 is None:
        P1P2 = P2 - P1
    
    Lambda = np.ones((PointNum), )
    hasIntersection = HasIntersection(P1, P2, Q1, Q2)
    hasIntersectionIndices = np.argwhere(hasIntersection)
    hasIntersectionNum = hasIntersectionIndices.shape[0]
    hasIntersectionIndices = hasIntersectionIndices.reshape(hasIntersectionNum)
    P1WithIntersection = P1[hasIntersectionIndices]
    P2WithIntersection = P2[hasIntersectionIndices]
    intersectionPoins = IntersectionPoints(P1WithIntersection, P2WithIntersection, Q1, Q2)

    LambdaWithIntersection = np.mean((intersectionPoins - P1WithIntersection) / P1P2[hasIntersectionIndices], axis=1)
    Lambda[hasIntersectionIndices] = LambdaWithIntersection
    return Lambda

def SegmentInterceptedByCircle(P1, P2, XYCenter, Radius, P1P2):
    PointNum = P1.shape[0]
    if P1P2 is None:
        P1P2 = P2 - P1
    

def Edges2MidPoints(Edges):
    return Edges2MidPointsNp(np.array(Edges, dtype=np.float32)).tolist()

def Edges2MidPointsNp(EdgesNp): # [EdgeNum, (Point1, Point2), (x, y)]
    return np.mean(EdgesNp, axis=1)

def IntersectionPoint(A, B, C, D):
    return
    
def HasIntersection_(P1, P2, Q1, Q2):
    isBoundaryBoxIntersect = IsBoundaryBoxIntersect(P1, P2, Q1, Q2)    
    return 

def IsBoundaryBoxIntersect(P1, P2, Q1, Q2): # 快速排斥实验
    P1X, P1Y, P2X, P2Y = P1[:, 0], P1[:, 1], P2[:, 0], P2[:, 1],
    Q1X, Q1Y, Q2X, Q2Y = Q1[:, 0], Q1[:, 1], Q2[:, 0], Q2[:, 1],
    return \
    np.min(P1X, P2X, axis=1) <= np.max(Q1X, Q2X, axis=1) * \
    np.min(Q1X, Q2X, axis=1) <= np.max(P1X, P2X, axis=1) * \
    np.min(P1Y, P2Y, axis=1) <= np.max(Q1Y, Q2Y, axis=1) * \
    np.min(Q1Y, Q2Y, axis=1) <= np.max(P1Y, P2Y, axis=1)
    # return \
    # np.min(P1[:, 0], P2[:, 0], axis=1) <= np.max(Q1[:, 0], Q2[:, 0], axis=1) * \
    # np.min(Q1[:, 0], Q2[:, 0], axis=1) <= np.max(P1[:, 0], P2[:, 0], axis=1) * \
    # np.min(P1[:, 1], P2[:, 1], axis=1) <= np.max(Q1[:, 1], Q2[:, 1], axis=1) * \
    # np.min(Q1[:, 1], Q2[:, 1], axis=1) <= np.max(P1[:, 1], P2[:, 1], axis=1)

def IsLineSegmentCross(P1, P2, Q1, Q2): # 跨立实验
    if \
        ((Q1[:, 0]-P1[:, 0])*(Q1[:, 1]-Q2[:, 1])-(Q1[:, 1]-P1[:, 1])*( Q1[:, 0]-Q2[:, 0])) * ((Q1[:, 0]-P2[:, 0])*(Q1[:, 1]-Q2[:, 1])-(Q1[:, 1]-P2[:, 1])*(Q1[:, 0]-Q2[:, 0])) < 0 or \
        ((P1[:, 0]-Q1[:, 0])*(P1[:, 1]-P2[:, 1])-(P1[:, 1]-Q1[:, 1])*(P1[:, 0]-P2[:, 0])) * ((P1[:, 0]-Q2[:, 0])*(P1[:, 1]-P2[:, 1])-(P1[:, 1]-Q2[:, 1])*( P1[:, 0]-P2[:, 0])) < 0:
        return True
    else:
       return False

def _Perpendicular(a):
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:,0]
    return b

def IntersectionPoints(P1, P2, Q1, Q2) :
    P1P2 = P2 - P1
    Q1Q2 = Q2 - Q1
    Q1P1 = P1 - Q1
    P1P2Perpendicular = _Perpendicular(P1P2)
    Denominator = np.sum(P1P2Perpendicular * Q1Q2, axis=1)
    Numerator = np.sum(P1P2Perpendicular * Q1P1, axis=1)
    return (Numerator / Denominator.astype(float))[:, np.newaxis] * Q1Q2 + Q1

def XYsPair2Distance(A, B):
    # if B.shape[0]==512:
    #     print("aaa")
    AB = A[:, np.newaxis, :] - B[np.newaxis, :, :] # [A.PointNum, B.PointNum, (x, y)]
    Lengths = Vectors2Lengths(AB) # [A.PointNum, B.PointNum]
    return Lengths # [A.PointNum, B.PointNum]

def Polar2XY(Radius, Direction):
    # Direction: [-pi, pi)
    return Radius * math.cos(Direction), Radius * math.sin(Direction)
Polar2XY = Polar2XY

def XY2Polar(x, y):
    return cmath.polar(complex(x, y))

def XY2PolarNp(PointsNp):
    # PointsNp: np.ndarray with shape [PointNum, (x, y)]
    Radius = np.linalg.norm(PointsNp, axis=-1)
    PolarsNp = np.stack([np.cos(Radius), np.sin(Radius)], axis=1)
    return PolarsNp

def Vectors2DirectionNp(PointsNp): # [PointNum, (x, y)]
    return np.arctan2(PointsNp[:, 1], PointsNp[:, 0])

def Vertices2VertexPairs(Vertices, close=True):
    VertexNum = len(Vertices)
    VertexPairs = []
    if close:
        for Index in range(VertexNum):
            VertexPairs.append([Vertices[Index], Vertices[(Index + 1) % VertexNum]])
    return VertexPairs

def Vertices2Vectors(Vertices, close=True): 
    return Vertices2EdgesNp(np.array(Vertices, dtype=np.float32), close=close).tolist()

def Vertices2EdgesNp(VerticesNp, close=True):
    if close:
        VerticesNp = np.concatenate((VerticesNp, VerticesNp[0,:][np.newaxis, :]), axis=0)
    VectorsNp = np.diff(VerticesNp, axis=0)
    return VectorsNp

def VertexPairs2Vectors(VertexPairs):
    Vectors = []
    for VertexPair in VertexPairs:
        Vectors.append(VertexPair2Vector(VertexPair))
    return Vectors

def VertexPair2Vector(VertexPair):
    return [VertexPair[1][0] - VertexPair[0][0], VertexPair[1][1] - VertexPair[0][1]]

def VertexPair2VectorNp(VertexPairNp): # ((x0, y0), (x1, y1))
    return np.diff(VertexPairNp, axis=1)

def Vectors2Norms(Vectors):
    return Vectors2NormsNp(np.array(Vectors, dtype=np.float32)).tolist()

def Vectors2GivenLengths(Vectors, Lengths):
    return Vectors * Lengths[:, np.newaxis] / np.linalg.norm(Vectors, axis=1, keepdims=True)

Vectors2GivenLength = Vectors2GivenLengths

def Vectors2LengthsNp(VectorsNp): 
    # @VectorsNp: np.ndarray with shape [..., VectorSize]
    Lengths = np.linalg.norm(VectorsNp, axis=-1, ord=2, keepdims=False)
    return Lengths

Vectors2Lengths = Vectors2LengthsNp

def Vectors2NormsNp(VectorsNp):  # Calculate Norm Vectors Pointing From Inside To Outside Of Polygon
    VectorNum = VectorsNp.shape[0]
    VectorsNorm = np.zeros([VectorNum, 2])
    # (a, b) is vertical to (b, -a)
    VectorsNorm[:, 0] = VectorsNp[:, 1]
    VectorsNorm[:, 1] = - VectorsNp[:, 0]
    # Normalize To Unit Length
    VectorsNorm = VectorsNorm / (np.linalg.norm(VectorsNorm, axis=1, keepdims=True))
    return VectorsNorm

def Vectors2NormsDirectionsNp(Vectors):
    return Vectors2NormsNp(Vectors), Vectors2DirectionsNp(Vectors)

def Vectors2Directions(Vectors):
    return Vectors2DirectionsNp(np.array(Vectors, dtype=np.float32)).tolist()

def Vector2Direction(Vector):
    return np.arctan2(Vector[0], Vector[1])

def Vector2Degree(Vector):
    return Vector2Direction(Vector) * 180.0 / np.pi

def Vectors2RadiansNp(VectorsNp):
    return np.arctan2(VectorsNp[:, 1], VectorsNp[:, 0])

Vectors2DirectionsNp = Vectors2RadiansNp

def Distance2Edges(PointsNp, EdgeVerticesNp, EdgeNormsNp):
    Points2EdgeVertices = EdgeVerticesNp[np.newaxis, :, :] - PointsNp[:, np.newaxis, :]
    # (1, VertexNum, 2) - (PointNum, 1, 2) = (PointNum, VertexNum, 2)
    return np.sum(Points2EdgeVertices * EdgeNormsNp[np.newaxis, :, :], axis=2)
    # (PointNum, VertexNum, 2) * (1, VertexNum, 2)

def StartEndPoints2VectorsNp(PointsStart, PointsEnd):
    return PointsStart - PointsEnd

def StartEndPoints2VectorsDirectionNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2DirectionsNp(Vectors)

def StartEndPoints2VectorsNormNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2NormsNp(Vectors)

def StartEndPoints2VectorsNormDirectionNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2NormsDirectionsNp(Vectors)


def PlotIntersectionTest(SavePath=None, Num=10):
    fig, ax = plt.subplots()

def LatticeXYs(BoundaryBox, ResolutionX, ResolutionY, Flatten=True):
    XYs = np.zeros((ResolutionX, ResolutionY, 2), dtype=np.float32)
    Indices = np.zeros((ResolutionX, ResolutionY, 2), dtype=np.int32)
    for i in range(ResolutionX):
        Indices[i, :, 0] = i
    for i in range(ResolutionY):
        Indices[:, i, 1] = i

    Ys = PixelIndices2XYs(Indices[0, :, :], BoundaryBox, ResolutionX, ResolutionY)[:, 1]
    Xs = PixelIndices2XYs(Indices[:, 0, :], BoundaryBox, ResolutionX, ResolutionY)[:, 0]

    for i in range(ResolutionX):
        XYs[i, :, 0] = Xs[i]
    for i in range(ResolutionY):
        XYs[:, i, 1] = Ys[i]

    if Flatten:
        return XYs.reshape(ResolutionX * ResolutionY, 2)
    else:
        return XYs

def XY2PixelIndex(X, Y, BoundaryBox, ResolutionX, ResolutionY):
    #return int( (x / box_Width + 0.5) * ResolutionX ), int( (y / box_Height+0.5) * ResolutionY )
    XMin, XMax, YMin, YMax = BoundaryBox.XMin, BoundaryBox.XMax, BoundaryBox.YMin, BoundaryBox.YMax
    XRange = XMax - XMin
    YRange = YMax - YMin
    PixelWidth = XRange / ResolutionX
    PixelHeight = YRange / ResolutionY
    PixelHalfWidth = PixelWidth / 2
    PixelHalfHeight = PixelHeight / 2
    XIndex = (X - XMin - PixelHalfWidth) / PixelWidth
    YIndex = (Y - YMin - PixelHalfHeight) / PixelHeight 
    return round(XIndex), round(YIndex)

def XYs2PixelIndices(XYs, BoundaryBox, ResolutionX, ResolutionY):
    # return int( (x / box_Width + 0.5) * ResolutionX ), int( (y / box_Height+0.5) * ResolutionY )
    # print('points shape:'+str(points.shape))
    XMin, XMax, YMin, YMax = BoundaryBox.XMin, BoundaryBox.XMax, BoundaryBox.YMin, BoundaryBox.YMax
    XRange = XMax - XMin
    YRange = YMax - YMin
    PixelWidth = XRange / ResolutionX
    PixelHeight = YRange / ResolutionY
    PixelHalfWidth = PixelWidth / 2
    PixelHalfHeight = PixelHeight / 2
    Xs, Ys = XYs[:, 0], XYs[:, 1]
    XIndices = (Xs - XMin - PixelHalfWidth) / PixelWidth 
    YIndices = (Ys - YMin - PixelHalfHeight) / PixelHeight
    XYIndices = np.stack([XIndices, YIndices], axis=1)
    return np.around(XYIndices).astype(np.int32) # np.astype(np.int) do floor, not round.

def PixelIndex2XYs(xIndex, yIndex, BoundaryBox, ResolutionX, ResolutionY):
    # Indices: np.ndarray with shape [PointNum, (xIndex, yIndex)]
    XMin, XMax, YMin, YMax = BoundaryBox.XMin, BoundaryBox.YRange, BoundaryBox.YMin, BoundaryBox.YMax
    XRange = XMax - XMin
    YRange = YMax - YMin
    PixelWidth = XRange / ResolutionX
    PixelHeight = YRange / ResolutionY
    PixelHalfWidth = PixelWidth / 2
    PixelHalfHeight = PixelHeight / 2
    return xIndex * PixelWidth + XMin + PixelHalfWidth, yIndex * PixelHeight  + YMin + PixelHalfHeight

def PixelIndices2XYs(Indices, BoundaryBox, ResolutionX, ResolutionY):
    # Indices: np.ndarray with shape (PointNum, (xIndex, yIndex))
    XMin, XMax, YMin, YMax = BoundaryBox.XMin, BoundaryBox.XMax, BoundaryBox.YMin, BoundaryBox.YMax
    XRange = XMax - XMin
    YRange = YMax - YMin
    PixelWidth = XRange / ResolutionX
    PixelHeight = YRange / ResolutionY
    PixelHalfWidth = PixelWidth / 2
    PixelHalfHeight = PixelHeight / 2
    Xs = Indices[:, 0] * PixelWidth + XMin + PixelHalfWidth
    Ys = Indices[:, 1] * PixelHeight  + YMin + PixelHalfHeight
    return np.stack([Xs, Ys], axis=1)

def PointInTriangle(P, A, B, C):
    AB = B - A
    BC = C - B
    CA = A - C
    AP = P - A
    BP = P - B
    CP = P - C
    ABCrossAP = Cross(AB, AP)
    BCCrossBP = Cross(BC, BP)
    CACrossCP = Cross(CA, CP)
    return ABCrossAP > 0.0 and BCCrossBP > 0.0 and CACrossCP > 0.0 \
        or ABCrossAP < 0.0 and BCCrossBP < 0.0 and CACrossCP < 0.0

def Triangles2BoundaryBox(ABC):
    # ABC: (TriangleNum, (A, B, C), (x, y))
    XMin = np.min(ABC[:, :, 0], axis=1) # [TriangleNum]
    XMax = np.max(ABC[:, :, 0], axis=1)
    YMin = np.min(ABC[:, :, 1], axis=1)
    YMax = np.max(ABC[:, :, 1], axis=1)
    BoundaryBox = np.stack([XMin, YMin, XMax, YMax], axis=1) # (TriangleNum, 4)