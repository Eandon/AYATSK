from torch import tensor, stack, ones_like


def adasoftmin3(x, dim=0):
    """
    adaptive Ln-Exp softmin
    the index parameter is adaptively determined according to x
    :param x: inputs, tensor type, get the minimum from them
    :param dim: {int}, get the minimum on which dimension
    :return:
    """
    x = x.double()
    q = -700 / x.data.max(dim=dim).values

    return (x * q.unsqueeze(dim=dim)).exp().sum(dim=dim).log() / q


def yager(x, lam=1, dim=0):
    """
    Yager T-norm
    :param x: inputs, tensor type, get the minimum from them
    :param lam: parameter of this operator, (0, +∞)
    :param dim: {int}, get the minimum on which dimension
    :return:
    """
    return 1 - (1 - x).pow(lam).sum(dim=dim).pow(1 / lam).minimum(tensor(1.0))


def yager_simple(x, lam=1, dim=0):
    """
    simplified Yager T-norm, without minimum operator
    :param x: inputs, tensor type, get the minimum from them
    :param lam: parameter of this operator, (0, +∞)
    :param dim: {int}, get the minimum on which dimension
    :return:
    """
    return 1 - (1 - x).pow(lam).sum(dim=dim).pow(1 / lam)


def ale_softmin_yager(x, lam=1, dim=0):
    """
    ALE-softmin based Yager T-norm, where ALE-softmin is used to replace the minimum operator in Yager T-norm
    :param x: inputs, tensor type, get the minimum from them
    :param lam: parameter of this operator, (0, +∞)
    :param dim: {int}, get the minimum on which dimension
    :return:
    """
    x = (1 - x).pow(lam).sum(dim=dim).pow(1 / lam)
    return 1 - adasoftmin3(stack([ones_like(x), x]))
