import multiprocessing.pool

class ProcessNoDaemon(multiprocessing.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class ContextNoDaemon(type(multiprocessing.get_context())):
    Process = ProcessNoDaemon

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class PoolNoDaemon(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = ContextNoDaemon()
        super(PoolNoDaemon, self).__init__(*args, **kwargs)

def RunProcessPool(
    ProcessNum = None,
    ParallelNumMax = 8,
    Func = None,
    ArgsListList = None,
    ArgsDictList = None,
    IsChildProcessDaemon = False # if True, parent process will try to kill child process on exit
):
    if callable(Func):
        assert isinstance(ProcessNum, int)
        FuncList = [Func for _ in range(ProcessNum)]
    else:
        assert isinstance(Func, list) or isinstance(Func, tuple)
        FuncList = Func
        if ProcessNum is None:
            ProcessNum = len(FuncList)
        else:
            assert isinstance(ProcessNum, int)
            assert len(FuncList) == ProcessNum

    if ArgsDictList is None:
        ArgsDictList = [dict() for _ in range(ProcessNum)]
    if ArgsListList is None:
        ArgsListList = [tuple() for _ in range(ProcessNum)]
    import sys
    import DLUtils
    print("ParentPID: %s ProcessNum: %d ParallelNumMax: %d"%(DLUtils.system.CurrentPID(), ProcessNum, ParallelNumMax), file=sys.__stdout__, flush=True)
    if IsChildProcessDaemon:
        """
        default by multiprocessing package. child process is forbidden to create grandchild process.
        parent process will attempt to kill all child process on exit.
            if parent process is killed by 'kill -9 ParentPID',
                for example
                    parent process is run in background using 'nohup ... &') and killed with 'kill -9 ParentPID'
                then
                    child process will not be killed.
                    'kill -9 PID' kills parent process immediately.
            if parent process is killed by KeyBoardInterupt(such as 'Command+C') when run in terminal, then
                child process is likely to get killed. 
        use 'kill -- -ParentPID' to kill parent and all child process.
            this command kills the process group
        """
        ProcessPool = multiprocessing.Pool(ParallelNumMax)
    else:
        ProcessPool = PoolNoDaemon(ParallelNumMax) # child process will exit on parent exit.
    Result = {}
    for Index, Func in enumerate(FuncList):
        # sys.stdout, sys.stderr should not be in ArgsListList or ArgsDictList.
        # might cause instance exception during apply_sync
        Result[Index] = ProcessPool.apply_async(
            Func,
            args=ArgsListList[Index],
            kwds=ArgsDictList[Index]
        )
    ProcessPool.close()
    print("ProcessPool started.", flush=True)
    ProcessPool.join()
    print("ProcessPool finished.", flush=True)
    for ProcessIndex in range(ProcessNum):
        print("Type: %s"%(str(type(Result[ProcessIndex]))))
        ReturnValue = Result[ProcessIndex].get(timeout=1)
        try:
            DLUtils.PrintTo(sys.stdout, "ReturnValue of Process %d: %s"%(ProcessIndex, ReturnValue))
        except Exception:
            DLUtils.PrintTo(sys.stdout, "Process %d threw an Exception:"%ProcessIndex)
            DLUtils.PrintErrorStackTo(sys.stdout, Indent=1)
    import time
    Num = 0
    while(Num < 20):
        Num += 1
        print("Num: %d"%Num, file=sys.__stdout__, flush=True)
        time.sleep(1.0)
    
    return