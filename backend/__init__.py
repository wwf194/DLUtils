try:
    import DLUtils.backend._torch as torch
except Exception:
    pass

try:
    import DLUtils.backend._cuda as cuda
except Exception:
    pass

from ..utils.system import IsWindows
if IsWindows():
    import DLUtils.backend.win as win