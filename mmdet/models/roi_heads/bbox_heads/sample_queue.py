import collections

_QUEUE_POS_MAX_SIZE = 100
_QUEUE_NEG_MAX_SIZE = 100

"""
保存所有类别样本队列的字典集合
对于类别i

_Buffer = {
    i: {
        "pos": [(similarity, feature),],
        "neg": [(similarity, feature),],
    }
}
"""

_Buffer = collections.defaultdict(default_factory=dict(
    pos=list(),
    neg=list(),
))


def set_queue_pos_max_size(size):
    global _QUEUE_POS_MAX_SIZE
    _QUEUE_POS_MAX_SIZE = size


def set_queue_neg_max_size(size):
    global _QUEUE_NEG_MAX_SIZE
    _QUEUE_NEG_MAX_SIZE = size


def get_pos_samples(cat):
    return _Buffer[cat]["pos"]


def get_neg_samples(cat):
    return _Buffer[cat]["neg"]


def set_pos_samples(cat, samples):
    _Buffer[cat]["pos"] = samples


def set_neg_samples(cat, samples):
    _Buffer[cat]["neg"] = samples
