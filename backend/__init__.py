try:
    import DLUtils.backend.torch as torch
except Exception:
    pass

from ..utils.system import IsWindows
if IsWindows():
    import DLUtils.backend.win as win