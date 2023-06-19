import sqlite3
import DLUtils

def PrintIndex(Operator):
    print(ListIndex(Operator))

def PrintTable(Operator):
    print(ListTable(Operator))

def ListIndex(Operator):
    Query = '''
        SELECT name 
        FROM sqlite_master 
        WHERE type = 'index';
    '''
    Operator.execute(Query)
    IndexList = Operator.fetchall()
    return IndexList

def DeleteTable(Operator, TableName):
    try:
        Operator.execute(
            f"DROP TABLE {TableName};"
        )
    except Exception:
        return False
    return True
DropTable = DeleteTable

def ListTable(Operator):
    # list existing tables
    Query = '''
        SELECT name 
        FROM sqlite_master 
        WHERE type = 'table';
    '''    
    Operator.execute(Query)
    
    TableList = Operator.fetchall()
    
    # [('tableA'), ('tableB')]
    return [_[0] for _ in TableList]  

def CreateSession(DataPath):    
    Session = sqlite3.connect(DataPath)
    return Session

def CreateSessionAndOperator(DataPath):
    Session = CreateSession(DataPath)
    Operator = CreateOperator(Session)
    return Session, Operator

def CreateOperator(Session):
    return Session.cursor()

def TableExists(Operator, TableName, Verbose=False):
    TableList = ListTable(Operator)
    if Verbose:
        print(TableList)
    return TableName in TableList

def CreateDataBaseIfNotExists(FilePath):
    Data = sqlite3.connect(FilePath)
    Data.close()
    return True

def CreateDataBase(FilePath):
    # overwrite if file already exists
    DLUtils.file.RemoveFileIfExists(FilePath)
    return CreateDataBaseIfNotExists(FilePath)


# create table
# simple writing
# Operator.execute("CREATE TABLE heart-beat(title, year, score)")

# hyphen is not allowed in table name.

# QueryCreateTable = '''CREATE TABLE new_developers (
#                         id INTEGER PRIMARY KEY,
#                         name TEXT NOT NULL,
#                         joiningDate timestamp
#                     );'''

