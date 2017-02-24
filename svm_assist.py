#encoding:UTF-8

from numpy import *

#----------------------------------------------------------------------
def differences(a, b):
    """"""
    c = a[a!=b]
    d = b[a!=b]
    nums = nonzero(a!=b)[0]
    return concatenate((mat(nums), c, d)).T