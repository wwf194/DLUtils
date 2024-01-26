
import DLUtils.backend._torch as torch


try:
    import DLUtils.backend._cuda as cuda
except Exception:
    pass
import DLUtils

from ..utils.system import IsWindows
if IsWindows():
    import DLUtils.backend.win as win