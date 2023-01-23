

import DLUtils
from .image.classification import ImageClassificationTask
from .image.classification import mnist, cifar10


DatasetMap = DLUtils.IterableKeyToElement({
    ("MNIST", "mnist"): mnist,
    ("CIFAR", "cifar", "cifar10"): mnist,
    ("ImageClassification"): ImageClassificationTask
})

def Dataset(Name, *Args, **Dict):
    if Name in DatasetMap:
        return DatasetMap[Name](*Args, **Dict)
    else:
        raise Exception()

def Task(Name, *Args, **Dict):
    if Name in DatasetMap:
        return DatasetMap[Name](*Args, **Dict)
    else:
        raise Exception()