# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
# Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
# from scipy.stats import chisqprob

def compare_distrib(ds1, ds2, bins=10, ddof=0):
    """Compares distribution functions of two arrays

    Args:
        ds1 (np.array): 
        ds2 (np.array): 
        bins (int, optional): number of bins to split arrays to. Defaults to 10.
        ddof (int, optional): Delta Degrees of Freedom for chi square test. Defaults to 0.

    Returns:
        float: probability of chi sqaure test
    """
    hist1,bins= np.histogram(ds1, bins=bins)
    hist2,_= np.histogram(ds2, bins=bins)

    #remove zeroe freq bins
    mask=(hist1>0) | (hist2>0)
    hist1=hist1[mask]
    hist2=hist2[mask]
    bins=len(hist1)

    chi=(np.power(np.sqrt(len(ds1)/len(ds2))*hist2-np.sqrt(len(ds2)/len(ds1))*hist1, 2)/(hist1+hist2)).sum()


    res=stats.chisqprob(chi, bins - 1 - ddof)
    return res


def psi(x, y, bins=50):
    """Calculates Population stability index - difference in distribution of x (reference array) and y

    Args:
        x (np.array): reference array
        y (np.array): array to calculate PSI for
        bins (int, optional): number of bins to split x to. Defaults to 50.

    Returns:
        float: PSI (Population stability index) of y related to x
    """
    freqs, bins = np.histogram(x, bins = bins)
    freqs = 1. * freqs / np.sum(freqs)

    freqs2, _ = np.histogram(y, bins)
    freqs2 = 1. * freqs2 / np.sum(freqs2)

    psi = np.sum((freqs - freqs2) * np.log(freqs / freqs2))
    return psi
