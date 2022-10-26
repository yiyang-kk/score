"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller, 
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import numbers


def is_number(s):
    """Returns True if argument is numerical, i.e. can be trasferred to float data type.

    Args:
        s (any type): variable that we want to decide about if it is a numeber

    Returns:
        bool: True if argument is number.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def optimize_dtypes(df, str2cat=True, floatminbits=32, optimize_int=True):
    """Optimizes datatypes in pandas dataframe:
    - objects (strings) are casted as categorical datatype
    - bits of float datatypes are reduced to such extent number of unique values don't change
    - bits of integers are reduced to such extents there is no overflow

    Args:
        df (pd.DataFrame): dataframe to optimize data types in
        str2cat (bool, optional): True if strings should be casted as categorical. Defaults to True.
        floatminbits (int, optional): Minimal number of bits of datatype floats should be stored in. Defaults to 32.
        optimize_int (bool, optional): True if bits of ints should be reduced. Defaults to True.

    Returns:
        pd.DataFrame: df with datatypes optimized
    """
    types = {}
    if floatminbits <= 16:
        floatformats = ['float16', 'float32', 'float64']
    elif floatminbits <= 32:
        floatformats = ['float32', 'float64']
    else:
        floatformats = ['float64']
    for x in df.dtypes.iteritems():
        if (optimize_int and (x[1].name.startswith('int') or x[1].name.startswith('uint'))):
            flag = False
            for t in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64']:
                if df[x[0]].min() >= np.iinfo(t).min and df[x[0]].max() <= np.iinfo(t).max:
                    types[x[0]] = t
                    flag = True
                    break
            if not flag:
                types[x[0]] = 'int64'
        elif x[1].name.startswith('float'):
            flag = False
            cnt_orig = df[x[0]].nunique()
            for t in floatformats:
                if df[x[0]].astype(t).nunique() == cnt_orig:
                    types[x[0]] = t
                    flag = True
                    break
            if not flag:
                types[x[0]] = 'float64'
        elif (str2cat and
              x[1] == 'O' and
              df[x[0]].apply(lambda x:
                             type(x) == str or
                             isinstance(x, numbers.Number) and
                             np.isnan(x) or
                             x is None).all()):
            types[x[0]] = 'category'
        elif (str2cat and x[1] == 'O'):
            types[x[0]] = 'category'
    df = df.astype(types)
    return df
