import warnings
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pymysql
else:
    pymysql = DLUtils.LazyImport("pymysql")

def CreateMySQLSession(Host=None, User=None, Password=None):
    # assert IsPyMySqlImported
    session = pymysql.connect(
        host=Host,
        user=User,
        password=Password
    )
    return session

def GetMySQLVersion(session):
    cursor = session.cursor()
    # 使用 execute()  方法执行 SQL 查询 
    cursor.execute("SELECT VERSION()")
    # 使用 fetchone() 方法获取单条数据.
    version = cursor.fetchone()
    return version

MySQLVersion = GetMySQLVersion


    