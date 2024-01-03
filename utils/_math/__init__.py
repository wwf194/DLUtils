import DLUtils
import warnings
import random
import sys
import os

def IsNaN(Num):
    return math.isnan(Num)

def NanArrayNp(Shape):
    return np.full(Shape, np.nan)
NanArray = NanArrayNp

def ShuffleList(List, InPlace=False):
    if len(List) == 0:
        return List
    if InPlace:
        _List = List
    else:
        _List = list(List)

    random.shuffle(_List)
    return _List

def RandomByteArray(Num):
    return bytearray(os.urandom(Num))

def RandomSelect(List, Num, Repeat=False):
    if Repeat:
        return RandomSelectFromListWithReplacement(List, Num)
    
    if isinstance(List, int):
        List = range(List)

    if len(List) > Num:
        return random.sample(List, Num)
    else:
        return list(List)
RandomSelectFromList = RandomSelect

def RandomSelectOne(List):
    return RandomSelect(List, 1)[0]

def RandomSelectFromListRepeat(List, Num):
    # return random.choices(List, Num)
    return np.random.choice(List, size=Num, replace=True)
RandomSelectWithReplacement = RandomSelectFromListWithReplacement = RandomSelectFromListRepeat
RandomSelectWithRepeat = RandomSelectRepeat = RandomSelectFromListWithRepeat = RandomSelectFromListRepeat

def RandomIntInRange(Left, Right, IncludeRight=False):
    if not IncludeRight:
        Right -= 1
    #assert Left <= Right 
    return random.randint(Left, Right)

def MultipleRandomIntInRange(Num, *List, AllowSame=False, **Dict):
    if AllowSame:
        return [RandomIntInRange(*List, **Dict) for _ in range(Num)]
    else:
        NumCurrent = 0
        IntSet = set()
        while(len(IntSet) <= Num):
            IntSet.add(RandomIntInRange(*List, **Dict))
        return list(IntSet)

def RandomInt(Positive=True, Max=None): # uniformly sampling from a [0, a large enough integer]
    if Positive:
        if Max is None:
            Max = 2 ** 32 - 1
        return random.randint(0, Max) # range of 32-bit unsigned integer
    else:
        if Max is None:
            Max = 2 ** 31 - 1
        return random.randint(- Max - 1, Max) # range of 32-bit signed integer

def MultipleRandomInt(Num=10, *List, AllowSame=False, **Dict):
    if AllowSame:
        return [RandomInt(*List, **Dict) for _ in range(Num)]
    else:
        NumCurrent = 0
        IntSet = set()
        while(len(IntSet) <= Num):
            IntSet.add(RandomInt(*List, **Dict))
        return list(IntSet)

def SelectSpecificRowsAndColsNp(Data, RowIndexList, ColIndexList):
    DataRow = Data[RowIndexList, :]
    DataRowCol = DataRow[ColIndexList]
    return DataRowCol
SelectSpecificRowsAndCols = SelectSpecificRowsAndColsNp

def GaussianProbDensity(data, Mean, Std):
    Exponent = - 0.5 * ((data - Mean) / Std) ** 2
    return GaussianCoefficient / Std * np.exp(Exponent)

def GaussianCurveValue(data, Amp, Mean, Std):
    Exponent = - 0.5 * ((data - Mean) / Std) ** 2
    return Amp * np.exp(Exponent)

def Float2BaseAndExponent(Float, Base=10.0):
    if Float < 0.0:
        Exponent = math.floor(math.log(-Float, Base))
    else:
        Exponent = math.floor(math.log(Float, Base))
    Coefficient = Float / 10.0 ** Exponent
    return Coefficient, Exponent

Float2BaseExp = Float2BaseAndExponent
try:
    import sklearn
    from sklearn.decomposition import PCA as PCAsklearn # sklearn.decomposition.PCA not supported
except Exception:
    if DLUtils.Verbose:
        warnings.warn("lib sklearn not found.")
else:
    def PCA(data, ReturnType="PyObj"):
        FeatureNum = data.shape[1]
        PCATransform = PCAsklearn(n_components=FeatureNum)
        PCATransform.fit(data) # Learn PC directions
        dataPCA = PCATransform.transform(data) #[SampleNum, FeatureNum]

        if ReturnType in ["Transform"]:
            return PCATransform
        # elif ReturnType in ["TransformAndData"]:
        #     return 
        elif ReturnType in ["PyObj"]:
            return DLUtils.PyObj({
                "dataPCA": DLUtils.ToNpArray(dataPCA),
                "Axis": DLUtils.ToNpArray(PCATransform.components_),
                "VarianceExplained": DLUtils.ToNpArray(PCATransform.explained_variance_),
                "VarianceExplainedRatio": DLUtils.ToNpArray(PCATransform.explained_variance_ratio_)
            })

try:
    import numpy as np
except Exception:
    IsNumpyImported = False
else:
    IsNumpyImported = True

try:
    import scipy
except Exception:
    IsScipyImported = False
else:
    IsScipyImported = True

if IsNumpyImported:
    def NpArrayStatistics(data, verbose=False):
        DataStat = DLUtils.param({
            "Min": np.nanmin(data),
            "Max": np.nanmax(data),
            "Mean": np.nanmean(data),
            "Std": np.nanstd(data),
            "Var": np.nanvar(data)
        })
        return DataStat
    NpStat = NpArrayStat = NpStatistics = NpArrayStatistics

    def NpArrayStatisticsStr(data, verbose=False):
        DataStat = {
            "Min": np.nanmin(data),
            "Max": np.nanmax(data),
            "Mean": np.nanmean(data),
            "Std": np.nanstd(data),
            "Var": np.nanvar(data)
        }
        StrList = []
        for Name, Key in DataStat.items():
            StrList.append(Name)
            StrList.append(": ")
            StrList.append(DLUtils.Float2StrDisplay(Key))
            StrList.append("\n")
        return "".join(StrList)

    def ReplaceNaNOrInfWithZeroNp(data):
        data[~np.isfinite(data)] = 0.0
        return data
    ReplaceNaNOrInfWithZero = ReplaceNaNOrInfWithZeroNp

    def RandomSelectFromListRepeat(List, Num):
        # return random.choices(List, Num)
        # assert isinstance(Num, int)
        return np.random.choice(List, size=Num, replace=True)
    RandomSelectWithReplacement = RandomSelectFromListWithReplacement = RandomSelectFromListRepeat
    RandomSelectRepeat = RandomSelectFromListRepeat

if IsNumpyImported and IsScipyImported:
    from scipy.stats import rankdata
    def GetElementRank(Data, OnNaN="omit", OnSameValue="min"):
        Shape = Data.shape
        
        DataRank = rankdata(Data, method=OnSameValue, nan_policy=OnNaN)
        DataRank = DataRank.reshape(Shape)
        return DataRank

try:
    import numpy as np
    GaussianCoefficient = 1.0 / (2 * np.pi) ** 0.5
    try:
        import torch
        import math
        import pandas as pd
        import scipy
        import sklearn
        from sklearn.decomposition import PCA as PCAsklearn # sklearn.decomposition.PCA not supported
    except Exception:
        pass
    else:




        def IsAllNaNOrInf(data):
            return (np.isnan(data) | np.isinf(data)).all()

        def RemoveNaNOrInf(data):
            # data: 1D np.ndarray.
            return data[np.isfinite(data)]
    
        def CalculateZScore(Data):
            return scipy.stats.zscore(Data)
    
        def TorchTrainParamtat(tensor, verbose=False, ReturnType="PyObj"):
            statistics = {
                "Min": torch.min(tensor).item(),
                "Max": torch.max(tensor).item(),
                "Mean": torch.mean(tensor).item(),
                "Std": torch.std(tensor).item(),
                "Var": torch.var(tensor).item()
            }
            if ReturnType in ["Dict"]:
                return statistics
            elif ReturnType in ["PyObj"]:
                return DLUtils.PyObj(statistics)
            else:
                raise Exception()

        def CreateNpArray(Shape, Value, DataType):
            return np.full(tuple(Shape), Value, dtype=DataType)



        def UniformAfterSoftmax(Shape):
            if isinstance(Shape, int):
                Shape = (Shape,)
            X = DLUtils.SampleFromUniformDistribution(Shape, Min=1.0e-9, Max=1.0)
            X = X / X.sum() # X sums to one.
            print(X.sum())
            Y = np.log(X) # np.log is ln
            Y = Y - Y.mean()
            return Y
        SampleFromSofmaxUniform = UniformAfterSoftmax

        def EvenlyDistributedAfterSoftmax(Shape):
            if isinstance(Shape, int):
                Shape = (Shape,)
            if len(Shape) == 1:
                Num = Shape[0]
            elif len(Shape) > 1:
                Num = 1
                for Index in len(range(Shape)):
                    Num *= Shape[Index]
            else:
                raise Exception()
            X = np.linspace(1.0 / Num, 1.0, Num)
            X = X / X.sum() # X sums to one.
            print(X.sum())
            Y = np.log(X) # np.log is ln
            Y = Y - Y.mean()
            return Y

        def CosineSimilarityNp(vecA, vecB):
            normA = np.linalg.norm(vecA)
            normB = np.linalg.norm(vecB)
            #normA_ = np.sum(vecA ** 2) ** 0.5
            #normB_ = np.sum(vecB ** 2) ** 0.5
            CosineSimilarity = np.dot(vecA.T, vecB) / (normA * normB)
            return CosineSimilarity

        def Vectors2Directions(Vectors):
            Directions = []
            for Vector in Vectors:
                R, Direction = DLUtils.geometry2D.XY2Polar(*Vector)
                Directions.append(Direction)    
            return Directions

        def Vector2Norm(VectorNp):
            return np.linalg.norm(VectorNp)

        def Vectors2NormsNp(VectorsNp): # VectorsNp: [VectorNum, VectorSize]
            return np.linalg.norm(VectorsNp, axis=-1)

        def Angles2StandardRangeNp(Angles):
            return np.mod(Angles, np.pi * 2) - np.pi

        def IsAcuteAnglesNp(AnglesA, AnglesB):
            return np.abs(Angles2StandardRangeNp(AnglesA, AnglesB)) < np.pi / 2

        def ToMean0Std1Np(data, StdThreshold=1.0e-9):
            std = np.std(data, keepdims=True)
            mean = np.mean(data, keepdims=True)
            if std < StdThreshold:
                DLUtils.AddWarning("ToMean0Std1Np: StandardDeviation==0.0")
                return data - mean
            else:
                return (data - mean) / std

        ToMean0Std1 = ToMean0Std1Np

        def ToRangeMinus1Positive1(Data):
            Min, Max = Data.min(), Data.max()
            return (Data - Min) / (Max - Min)
        def Norm2GivenMeanStdNp(data, Mean, Std, StdThreshold=1.0e-9):
            data = ToMean0Std1Np(data, StdThreshold)
            return data * Std + Mean

        Norm2GivenMeanStd = Norm2GivenMeanStdNp

        def Norm2Mean0Std1Torch(data, axis=None, StdThreshold=1.0e-9):
            std = torch.std(data, dim=axis, keepdim=True)
            mean = torch.mean(data, dim=axis, keepdim=True)
            # if std < StdThreshold:
            #     DLUtils.AddWarning("ToMean0Std1Np: StandardDeviation==0.0")
            #     return data - mean
            # else:
            # To Be Implemented: Deal with std==0.0
            return (data - mean) / std

        def Norm2Mean0Std1Np(data, axis=None, StdThreshold=1.0e-9):
            std = np.std(data, axis=axis, keepdims=True)
            mean = np.mean(data, axis=axis, keepdims=True)
            return (data - mean) / std

        def Norm2Sum1(data, axis=None):
            # data: np.ndarray. Non-negative.
            data / np.sum(data, axis=axis, keepdims=True)

        def Norm2Range01(data, axis=None):
            Min = np.min(data, axis=axis, keepdims=True)
            Max = np.max(data, axis=axis, keepdims=True)
            return (data - Min) / (Max - Min)

        def Norm2Min0(data, axis=None):
            Min = np.min(data, axis=axis, keepdims=True)
            return data - Min

        Norm2ProbDistribution = Norm2ProbabilityDistribution = Norm2Sum1

        def GetGaussianProbDensityMethod(Mean, Std):
            GaussianCoefficient = 1.0 / (2 * np.pi) ** 0.5
            # return Gaussian Probability Density Function
            CalculateExponent = lambda data: - 0.5 * ((data - Mean) / Std) ** 2
            Coefficient = GaussianCoefficient / Std
            ProbDensity = lambda data: Coefficient * np.exp(CalculateExponent(data))
            return ProbDensity

        def GetGaussianCurveMethod(Amp, Mean, Std):
            # return Gaussian Curve Function.
            CalculateExponent = lambda data: - 0.5 * ((data - Mean) / Std) ** 2
            GaussianCurve = lambda data: Amp * np.exp(CalculateExponent(data))
            return GaussianCurve

        def GaussianProbDensity(data, Mean, Std):
            Exponent = - 0.5 * ((data - Mean) / Std) ** 2
            return GaussianCoefficient / Std * np.exp(Exponent)

        def GaussianCurveValue(data, Amp, Mean, Std):
            Exponent = - 0.5 * ((data - Mean) / Std) ** 2
            return Amp * np.exp(Exponent)

        def Floats2BaseAndExponent(Floats, Base=10.0):
            Floats = DLUtils.ToNpArray(Floats)
            Exponent = np.ceil(np.log10(Floats, Base))
            Coefficient = Floats / 10.0 ** Exponent
            return Coefficient, Exponent
        Floats2BaseExp = Floats2BaseAndExponent

        def CalculatePearsonCoefficient(dataA, dataB):
            # dataA: 1d array
            # dataB: 1d array
            dataA = DLUtils.EnsureFlat(dataA)
            dataB = DLUtils.EnsureFlat(dataB)
            # data = pd.DataFrame({
            #     "dataA": dataA, "dataB": dataB
            # })
            # return data.corr()
            dataA=pd.Series(dataA)
            dataB=pd.Series(dataB)
            return dataA.corr(dataB, method='pearson')

        def CalculatePearsonCoefficientMatrix(dataA, dataB):
            # dataA: Design matrix of shape [SampleNum, FeatureNumA]
            # dataB: Design matrix of shape [SampleNum, FeatureNumB]
            FeatureNumA = dataA.shape[1]
            FeatureNumB = dataB.shape[1]
            SampleNum = dataA.shape[0]

            Location = DLUtils.GetGlobalParam().system.TensorLocation

            dataAGPU = DLUtils.ToTorchTensor(dataA).to(Location)
            dataBGPU = DLUtils.ToTorchTensor(dataB).to(Location)

            dataANormed = Norm2Mean0Std1Torch(dataAGPU, axis=0)
            dataBNormed = Norm2Mean0Std1Torch(dataBGPU, axis=0)

            CorrelationMatrixGPU = torch.mm(dataANormed.permute(1, 0), dataBNormed) / SampleNum
            CorrelationMatrix = DLUtils.TorchTensor2NpArray(CorrelationMatrixGPU)
            return CorrelationMatrix

        def CalculateBinnedMeanStd(Xs, Ys=None, BinNum=None, BinMethod="Overlap", Range="MinMax", **Dict):
            if Ys is None:
                Ys = Xs
            if Range in ["MinMax"]:
                XMin, XMax = np.nanmin(Xs), np.nanmax(Xs)
            else:
                raise Exception(Range)
            
            if BinNum is None:
                BinNum = max(round(len(Xs) / 100.0), 5)
            
            if BinMethod in ["Overlap"]:
                BinNumTotal = 2 * BinNum - 1
                BinCenters = np.linspace(XMin, XMax, BinNumTotal + 2)[1:-1]

                BinWidth = (XMax - XMin) / BinNum
                Bins1 = np.linspace(XMin, XMax, BinNum + 1)
                
                BinMeans1, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='mean', bins=Bins1)
                BinStds1, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='std', bins=Bins1)
                BinCount1, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='count', bins=Bins1)

                Bins2 = Bins1 + BinWidth / 2.0
                Bins2 = Bins2[:-1]

                BinMeans2, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='mean', bins=Bins1)
                BinStds2, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='std', bins=Bins1)
                BinCount2, BinXs, BinNumber = scipy.stats.binned_statistic(
                    Xs, Ys, statistic='count', bins=Bins1)

                BinMeans, BinStds, BinCount = [], [], []
                for BinIndex in range(BinNum-1):
                    BinMeans.append(BinMeans1[BinIndex])
                    BinMeans.append(BinMeans2[BinIndex])
                    BinStds.append(BinStds1[BinIndex])
                    BinStds.append(BinStds2[BinIndex])
                    BinCount.append(BinCount1[BinIndex])
                    BinCount.append(BinCount2[BinIndex])
                
                BinMeans.append(BinMeans1[BinNum - 1])
                BinStds.append(BinStds1[BinNum - 1])
                BinCount.append(BinCount1[BinNum - 1])

                BinStats = DLUtils.param({
                    "YMean": DLUtils.List2NpArray(BinMeans),
                    "YStd": DLUtils.List2NpArray(BinStds),
                    "YNum": DLUtils.List2NpArray(BinCount),
                    "Xs": BinCenters,
                })
                return BinStats
            else:
                raise Exception(BinMethod)


        def PCA(data, ReturnType="PyObj"):
            FeatureNum = data.shape[1]
            PCATransform = PCAsklearn(n_components=FeatureNum)
            PCATransform.fit(data) # Learn PC directions
            dataPCA = PCATransform.transform(data) #[SampleNum, FeatureNum]

            if ReturnType in ["Transform"]:
                return PCATransform
            # elif ReturnType in ["TransformAndData"]:
            #     return 
            elif ReturnType in ["PyObj"]:
                return DLUtils.PyObj({
                    "dataPCA": DLUtils.ToNpArray(dataPCA),
                    "Axis": DLUtils.ToNpArray(PCATransform.components_),
                    "VarianceExplained": DLUtils.ToNpArray(PCATransform.explained_variance_),
                    "VarianceExplainedRatio": DLUtils.ToNpArray(PCATransform.explained_variance_ratio_)
                })
            else:
                raise Exception(ReturnType)            
    def ManhattanDistance(Point1, Point2):
        return np.linalg.norm(Point1, Point2, ord=1)
    def EuclideanDistance(Point1, Point2, Norm=2):
        return np.linalg.norm(Point1, Point2, ord=Norm)
    Distance = EuclideanDistance
except Exception:
    pass

from .distribution import *