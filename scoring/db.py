# -*- coding: utf-8 -*-
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

import sqlalchemy
from sqlalchemy import create_engine, types
import pandas as pd
import numpy as np
import os
import warnings
import traceback
from hashlib import sha512
import os
import sys
import numbers
import numpy as np

os.environ['NLS_LANG'] = '.UTF8'


def get_engine(filename=None):
    """Creates SQLAlchemy engine (as singleton) based on connection string from ~/.ora_cnn or filename if specified.

    Args:
        filename: name of file with connection string
    
    Returns:
        SQLAlchemy engine
    """
    global _engine
    if filename is None:
        filename=os.path.join(os.path.expanduser('~'), '.ora_cnn')
    need_reconnect = False
    if '_engine' in globals() and _engine is not None:
        try: 
            pd.read_sql_query('select * from dual', con=_engine)
        except sqlalchemy.exc.DatabaseError as e:           
            warnings.warn(traceback.format_exc())
            need_reconnect = True
    if need_reconnect or '_engine' not in globals() or _engine is None:  
        with open(filename) as f:
            ora_cnn = f.read()
        _engine = create_engine(ora_cnn, echo=False, encoding='utf-8')
    return _engine


def to_sql(df, name, con = None, chunksize=10000, if_exists='replace', **kwargs):
    """
    Writes pandas.DataFrame to database.

    See params of pandas.DataFrame.to_sql. This function generate proper column names (they should less than
    30 characters). And fixes performance issues by using varchar instead of BLOB used by default.

    Args:
        df:
        name:
        con:
        chunksize:
        if_exists:
    """
    if con is None:
        con = get_engine()
    index_cols = [c[:30].lower() for c in (df.index.names if type(df.index) == pd.MultiIndex else [df.index.name]
                  if df.index.name is not None else [])]
    if sys.version_info[0] < 3:
        columns=[unicode(c[:30].lower()) for c in df.columns]
    else:
        columns=[c[:30].lower() for c in df.columns]

    if len(set(index_cols+columns)) != len(index_cols+columns):
        raise ValueError('Index/column names are not unique after truncation to 30 characters and converting to '
                         'lowercase')

    df_copy = df.rename(columns=dict(zip(df.columns, columns)))
    if len(index_cols) == 1:
        df_copy.index.rename(index_cols[0], inplace=True)
    elif len(index_cols) > 1:
        df_copy.index.rename(index_cols, inplace=True)

    dtyp = dict([(i,types.VARCHAR(i.str.len().max())) for i in
                 ([ii.name for ii in df_copy.index.levels] if type(df_copy.index) == pd.MultiIndex
                  else [df_copy.index] if df_copy.index.name is not None else []) if i.dtype == 'object' or
                  i.dtype.name=='category'] +
                  [(c,types.VARCHAR(df_copy[c].str.len().max())) for c in df_copy.columns
                   if df_copy[c].dtype == 'object' or df_copy[c].dtype.name == 'category'])
    df_copy[columns].to_sql(name, con=con, chunksize=chunksize, if_exists=if_exists, dtype=dtyp, **kwargs)


def read_sql(sql, con = None, refresh=False, optimize_types=False, str2cat=True, **kwargs):
    """Caching data read from database to csv (use refresh=True to refresh case)
    Optimizing data types (optimize_types=True)

    Args:
        sql: SQL query
        con: existing SQLAlchemy connection, will be creaye via get_engine if None passed
        refresh: should cache refreshed from re-reading from database
        optimize_types: should type optimization applied
        str2cat: convert string columns to pandas.category

    Returns:
        pandas.DataFrame

    """
    if con is None:
        con = get_engine()
    dir = 'db_cache'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if sys.version_info[0] < 3:
        hash_filename= sha512(sql).hexdigest()
    else:
        hash_filename= sha512(sql.encode('utf8')).hexdigest()    

    path=os.path.join(dir, hash_filename)
    if os.path.exists(path) and not refresh:
        df = pd.read_pickle(path)
        return df
        
    df = pd.read_sql_query(sql, con=con, **kwargs)
    if optimize_types:
        # types optimization works only for first reading from database or for regresh=True !!!
        new_types = get_optimized_types(df, str2cat=str2cat)
        for x in new_types.items():
            df[x[0]] = df[x[0]].astype(x[1])

    df.to_pickle(path)
    return df


def read_csv(filepath_or_buffer, optimize_types=False, sample_nrows=None, str2cat=True, minimalfloat='float64', **kwargs):
    """Reading CSV with types optimization.

    Args:
        filepath_or_buffer: file path ot file object
        optimize_types: do types optimization
        sample_nrows: number of rows to use for type optimization, all if None
        str2cat: do conversion of str to pandas.Category
        minimalfloat: minimal bit size of float 

    Returns:
        pandas.DataFrame
    """
    if not optimize_types:
        return pd.read_csv(filepath_or_buffer, **kwargs)

    df = pd.read_csv(filepath_or_buffer, nrows=sample_nrows, **kwargs)
    new_types = get_optimized_types(df, str2cat=str2cat, minimalfloat=minimalfloat)

    if sample_nrows is not None: # subsample based types inference - takes less memory
        df = pd.read_csv(filepath_or_buffer, dtype=new_types, **kwargs)
    else: # types optimization AFTER reading entire DataFrame
        for x in new_types.items():
            df[x[0]] = df[x[0]].astype(x[1])
    return df


def get_optimized_types(df, str2cat=True, minimalfloat='float64'):
    """Detects "minimum" types of DataFrame columns.

    Args:
        df: DataFrame
        str2cat: should str be converted to pandas.Category
        minimalfloat: minimal bit size of float 

    Returns:
        dcit: dict of column name -> type
    """
    new_types = {}
    for x in df.dtypes.iteritems():
        if x[1].name.startswith('int') or x[1].name.startswith('uint'):
            flag = False
            for t in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64']:
                if df[x[0]].min() >= np.iinfo(t).min and df[x[0]].max() <= np.iinfo(t).max:
                    new_types[x[0]] = t
                    flag = True
                    break
            if not flag:
                raise ValueError()
        elif x[1].name.startswith('float'):
            new_t = x[1]
            # restric possible float sizes, using float64 will result in empty list
            # and no attempts for smaller floats will occur
            possible_floats = [f for f in ['float32', 'float16'] if int(f[-2:]) >= int(minimalfloat[-2:])]
            for t in possible_floats:
                unique_num = df[x[0]].unique().shape[0]
                if df[x[0]].astype(t).unique().shape[0] < unique_num:
                    break
                new_t = t
            new_types[x[0]] = new_t
        elif str2cat and x[1] == 'O' and df[x[0]].apply(lambda x: type(x) == str or isinstance(x, numbers.Number) and
                                                        np.isnan(x) or x is None).all():
            new_types[x[0]] = 'category'
    return new_types



def get_optimal_numerical_type(series, float_type='float32'):
    """Returns optimal numerical type for pd.Series.

    Integer types will be chosen based on min/max values. Unsigned version if Series contains no negative numbers.
    Float32 is default unless converting to it would lower number of unique values in the Series. In that case float64 will be used.

    Args:
        series (pd.Series): Pandas series
        float_type (str, optional): can be used to force float64 (default: 'float32')

    Returns:
        str: optimal dtype as string

    Raises:
        ValueError when series is no numerical
    """
    if pd.api.types.is_numeric_dtype(series.values.dtype):
        pd_type = pd.to_numeric(series).dtype.name
    else:
        raise ValueError('Series \'{0}\':[{1}] is not numerical.'.format(series.name, series.dtype))

    if 'int' in pd_type:
        if series.min() >= 0:
            for t in ['uint8', 'uint16', 'uint32', 'uint64']:
                if series.max() < np.iinfo(t).max:
                    break
        else:
            for t in ['int8', 'int16', 'int32', 'int64']:
                if series.max() < np.iinfo(t).max and series.min() > np.iinfo(t).min:
                    break
    else:
        if series.astype(np.float32).nunique() == series.nunique() and float_type=='float32':
            t = 'float32'
        else:
            t = 'float64'
    return t