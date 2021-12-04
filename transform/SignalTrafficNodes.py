import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

from utils_torch.module.AbstractModules import AbstractModule
class SignalHolder(AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        kw.setdefault("HasTensor", False)
        utils_torch.transform.InitForModule(self, param, data, ClassPath="utils_torch.transform.SignalHolder", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
    def Receive(self, Obj):
        self.cache.Content = Obj
    def Send(self):
        return self.cache.Content
    def Clear(self):
        utils_torch.attrs.RemoveAttrIfExists(self.cache, "Content")
#utils_torch.transform.SetMethodForTransformModule(SignalHolder, HasTensor=False)

class SerialSender(AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        #super(SerialSender, self).__init__()
        utils_torch.transform.InitForModule(self, param, data,  ClassPath="utils_torch.transform.SerialSender", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        param = self.param
        cache = self.cache
        cache.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
    def SetSendMethod(self):
        param = self.param
        EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self.SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        EnsureAttrs(param, "Receive.Method", default="Default")
        method = GetAttrs(param.Receive.Method)
        if method in ["Default"]:
            self.Receive = self.ReceiveDefault
        elif method in ["Lambda", "eval"]:
            self.Receive = eval(GetAttrs(param.Receive.Args))
        else:
            raise Exception(method)
        return
    def ReceiveDefault(self, content):
        cache = self.cache
        cache.ContentList = content
        cache.NextSendIndex = 0
    def _SendDefault(self, ContentList, Index):
        return ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        cache = self.cache
        Content = self._Send(cache.ContentList, Index=cache.NextSendIndex)
        cache.NextSendIndex += 1
        return Content
#utils_torch.transform.SetMethodForTransformModule(SerialSender, HasTensor=False)

class SerialReceiver(AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        utils_torch.transform.InitForModule(self, param, data, ClassPath="utils_torch.transform.SerialReceiver", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        cache = self.cache
        self.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
        return
    def SetSendMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self._SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            EnsureAttrs(param, "Receive.Method", default="Default")
        method = GetAttrs(param.Receive.Method)
        if method in ["Default"]:
            self.Receive = self.ReceiveDefault
        elif method in ["Lambda", "eval"]:
            self.Receive = eval(GetAttrs(param.Receive.Args))
        else:
            raise Exception(method)
        return
    def ReceiveDefault(self, content):
        self.ContentList.append(content)
    def _SendDefault(self, List):
        return List
    def Send(self):
        result = self._Send(self.ContentList)
        self.ContentList = []
        return result
    def SendWithoutFlush(self):
        return self.ContentList
#utils_torch.transform.SetMethodForTransformModule(SerialReceiver, HasTensor=False)