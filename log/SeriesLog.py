
import DLUtils
class SeriesLog():
    def __init__(self, OutputList=["Console"]):
        # self.Param = DLUtils.Param({})
        # Param = self.Param
        # Log = DLUtils.log.NewLog(Name=None, OutputList=OutputList)
        # Param.Data.Log = Log
        # self.Log = Log
        self.Param = DLUtils.Param([])
    def ToFile(self, FilePath):
        self.Param.ToFile(FilePath)
    def FromFile(self, FilePath):
        self.Param = self.Param.FromFile(FilePath)
    def ToJsonFile(self, FilePath):
        self.Param.ToJsonFile(FilePath)
    def Add(self, Log):
        self.Param.append(Log)
        return self