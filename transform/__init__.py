from .reshape import Reshape
from .norm import Norm
from .norm import NormOnColorChannel

import torch
import numpy as np
import DLUtils
class Sum(DLUtils.module.AbstractOperator):
    def Receive(self, InList):
        return sum(InList)