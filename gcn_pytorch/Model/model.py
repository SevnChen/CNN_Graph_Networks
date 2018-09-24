# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU, Linear, LSTMCell, Module
from torch.autograd import Variable


class meta_module(Module):
    def __init__(self, **kwargs):
        # 参数处理
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        # 日志
        logging = kwargs.get('logging', False)
        self.logging = logging


class GCN(meta_module):
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
