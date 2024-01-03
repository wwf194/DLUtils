import numpy as np
import DLUtils

def ReturnGPUDevice(GPUIndex, ReturnType="str"):
    if ReturnType in ["str"]:
        return "cuda:%d"%GPUIndex
    elif ReturnType in ["int"]:
        return GPUIndex
    else:
        raise Exception()
try:
    import pynvml # pip install pynvml
except Exception:
    DLUtils.print("lib pynvml not found.")
else:
    def GPUFreeMemory(GPUIndex):
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(GPUIndex)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.free

    def GetGPUWithLargestAvailableMemory(Verbose=False, ReturnType="str", OutPipe=None):
        GPUNum = DLUtils.torch.GPUNum()
        MemoryFreeLargestIndex = -1
        MemoryFreeLargest = -1
        for GPUIndex in range(GPUNum):
            MemoryFree = GPUFreeMemory(GPUIndex)
            if Verbose:
                print("GPU%d MemoryAvailable: %d"%(GPUIndex, MemoryFree), file=OutPipe, flush=True)
            if MemoryFree > MemoryFreeLargest:
                MemoryFreeLargestIndex = GPUIndex
                MemoryFreeLargest = MemoryFree
        if Verbose:
            print("GPU%d has largest available memory %d"%(MemoryFreeLargestIndex, MemoryFreeLargest), file=OutPipe, flush=True)
        return ReturnGPUDevice(MemoryFreeLargestIndex, ReturnType=ReturnType)
    GetGPUWithLargestFreeMemory = GetGPUWithLargestAvailableMemory

try:
    from .._torch import ReportGPUUseageOfCurrentProcess
except Exception:
    pass

try:
    import nvidia_smi # pip install nvidia-ml-py3
except Exception:
    pass
else:
    def GetGPUWithLargestAvailableUseage(ReturnType="str"):
        nvidia_smi.nvmlInit()
        GPUNum = nvidia_smi.nvmlDeviceGetCount()
        GPUUseageList = []
        for GPUIndex in range(GPUNum):
            GPUHandle = nvidia_smi.nvmlDeviceGetHandleByIndex(GPUIndex)
            GPUUtil = nvidia_smi.nvmlDeviceGetUtilizationRates(GPUHandle)
            GPUUseageCurrent = GPUUtil.gpu / 100.0
            GPUUseageList.append(GPUUseageCurrent)
            print("GPU %d Useage: %.3f%%"%(GPUIndex, GPUUseageCurrent * 100.0))
        GPUUseageMinIndex = np.argmin(GPUUseageList)
        return ReturnGPUDevice(GPUUseageMinIndex, ReturnType=ReturnType)