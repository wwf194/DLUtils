

import DLUtils
from .image.classification import ImageClassificationTask
from .image.classification import MNIST, CIFAR10

DatasetMap = DLUtils.IterableKeyToElement({
    ("MNIST", "mnist"): MNIST,
    ("CIFAR", "cifar", "cifar10"): CIFAR10,
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
