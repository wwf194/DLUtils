
# for running scripts inside this package
import sys
sys.path.append("../")

from DLUtils import *
if __name__=="__main__":
    obj = utils.EmptyObj()
    print(obj.__dict__)

net = NewNetwork("Transformer")
