import DLUtils.backend.torch as torch

from ..utils.system import IsWindows
if IsWindows():
    import DLUtils.backend.win as win