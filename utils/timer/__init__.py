import DLUtils
import schedule
import time

from ...module import AbstractModule
class EventTrigger(AbstractModule):
    def SetTime(self):
        return self
    def Run(self):
        while True:
            schedule.run_pending()
            time.sleep(1)
    def FixedInterval(self, Interval, FirstTime, Num):
        return self
    def FixedTime(self):
        
        return self




import apscheduler