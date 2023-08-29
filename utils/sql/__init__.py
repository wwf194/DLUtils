
import DLUtils
try:
    import DLUtils.utils.sql.sqlite as sqlite
except Exception:
    pass
try:
    import DLUtils.utils.sql.mysql as mysql
    from .mysql import MySQLVersion, GetMySQLVersion, CreateMySQLSession
except Exception:
    pass