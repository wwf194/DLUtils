import torch.multiprocessing as mp
import torch.distributed as dist

import DLUtils
def Train(NodeIndex, Device, NodeNum):
    # each node uses a gpu. puts model on that gpu.
    # train / test process is controlled by 1 separate python process.
                    
    dist.init_process_group(                                   
    	backend='nccl',                                   
   		init_method='env://',                                   
    	world_size=NodeNum,                              
    	rank=NodeIndex                                              
    )

