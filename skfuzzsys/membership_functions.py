"""
membership functions used in the fuzzy system
"""


def gauss(x, m, sigma):
    """
    gaussian membership function
    exp(-(x-m)^2/(2*sigma^2))
    :param x: independent variable
    :param m: center
    :param sigma: spread
    :return: membership values
    """
    return (-(x - m) ** 2 / (2 * sigma ** 2)).exp()


def cemf(x, m, sigma, k=10):
    """
    composite exponential membership function (CEMF)
    :param x: input
    :param m: center
    :param sigma: spread
    :param k: parameter controlling the lower bound
    :return:
    """
    return k ** (-1 + (x - m).pow(2).div(2 * sigma ** 2).neg().exp())
