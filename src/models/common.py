from torch.nn.utils.rnn import PackedSequence
from torch.nn import Module, Embedding
import torch.nn.functional as FF
import torch

cuda = torch.cuda.is_available()

def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

class OneHotPacked(Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(_, x):
        return simple_elementwise_apply(FF.one_hot, x)

class EmbeddingPacked(Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = Embedding(**kwargs)

    def forward(self, x):
        return simple_elementwise_apply(self.embedding, x)
