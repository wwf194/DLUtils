import torch

import DLUtils
def ChangeTorchModuleParameter(module:torch.nn.Module):
    ParamDict = dict(module.named_parameters())
    ParamDict

def ReportTorchInfo(): # print info about training environment, global variables, etc.
    return torch.pytorch_info()


def PrintOptimizerStateDict(Optimizer, Out="Std"):
    StateDict = Optimizer.state_dict()
    
    """
    Adam.state_dict():
        "state": {
            0 : { // key is int, starting from 0.
                "step": tensor(391.),
                "exp_avg": historical average. tensor, same shape as weight tensor
                "exp_avg_sq": historical sq. tensor, same shape as weight tensor
            },
            1 : {
            },
            ...
        },
        "param_groups": [
                {
                    "lr": float,
                    "betas" ...,
                    "eps": ...,
                    "weight_decay": ...,
                    "amsgrad": ...,
                    "maximize": ...,
                    "foreach": ...,
                    "capturable": ...,
                    "params": [
                        0, 1, 2, ... //与Adam.state_dict()["state"].keys()一致
                        //为与Adam.param_groups()[0]列表的index
                    ]
                }, //param_group_1
                {
                }, //param_group_2
                {
                }
            ]
    Adam.param_groups():
        [
            {
                "lr": float,
                "betas" ...,
                "eps": ...,
                "weight_decay": ...,
                "amsgrad": ...,
                "maximize": ...,
                "foreach": ...,
                "capturable": ...,
                "params": [ //Adam.state_dict()["state"].keys()就是这个列表的index
                    tensor0, 
                    tensor1,
                    tensor2,
                    ... 
                ]
            }, //param_group_1
            {
            }, //param_group_2
            {
            }
        ]
    """
    DLUtils.PrintDict(StateDict, Out=Out)


    return

def GetLR():
    SGD.state_dict()['param_groups'][0]['lr']