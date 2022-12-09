

import DLUtils
from .image.classification import ImageClassificationTask
def Task(Type, *Args, **Dict):
    if Type in ["ImageClassification"]:
        return ImageClassificationTask(*Args, **Dict)
    else:
        raise Exception(Type)