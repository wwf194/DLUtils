
import sys
sys.path.append("../")
import DLUtils

from ftplib import FTP

def FTPConnect(Host=None, Port=21):
    FTPSession = FTP()
    # Host: IP Address, or domain.
    if Host is None:
        Host = "192.168.128.4"


    # ftp.login() # anonymous login
    FTPSession.dir() # print all folders under current dir.
    FileNmaeList = FTPSession.nlst()
    return
import DLUtils
from ...module import AbstractModule
class FTPSession(AbstractModule):
    def Connect(self, Host=None, Port=21):
        Param = self.Param
        if Host is not None:
            Param.Connect.Host = Host
        else:
            assert Param.Connect.hasattr("Host")
            Host = Param.Connect.Host
        self.Session = FTP()
        try:
            self.Session.connect(Host, Port)
        except Exception():
            print("Error during FTP connect.")
        return self
    def Login(self, UserName=None, PassWord=None):
        Param = self.Param
        if UserName is None:
            UserName = Param.Login.UserName
        else:
            Param.Login.UserName = UserName
        if PassWord is None:
            PassWord = Param.Login.PassWord
        else:
            Param.Login.PassWord = PassWord
        try:
            self.Session.login(UserName, PassWord)
        except Exception:
            print("Error during FTP login.")
        return self
    def SetPath(self, Path):
        self.PathCurrent = Path
    def ListFiles(self):
        return self.Session.nlst()
    
    
def Test():
    FTPSession().Connect("192.168.128.4").Login("wwf194", "wwf5218340").ListFiles()
    
    
if __name__ == "__main__":
    Test()