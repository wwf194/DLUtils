
import DLUtils
class SeriesLog():
    def __init__(self):
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