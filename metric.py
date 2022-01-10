# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 16:04:11 2022

@author: Yeeun Kim
"""

from math import e, log

def harmonic_number(n):
    if n == 1:
        return 0
    else:
        return log(n) + e