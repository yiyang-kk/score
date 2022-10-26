
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


import re


def xgb2sql(filename_xgb, filename_sql):
    """Plain XGBoost model dump file converter to SQL script.
    
    Args:
        filename_xgb (str): name of file with xgb model in native format (saved with booster.dump_model(...)).
        filename_sql (str): name of file to store sql script.
    """
    
    lines=open(filename_xgb) #input file from xgb (saved by booster.dump_model())

    def mp(var):
        mapping={

        }
        if var in mapping:
            return mapping[var]
        return var

    def f(s):

        # -------------------------------------------------------------------- #
        # if line consists of header (beginning of new tree) go to next line
        while(True):
            line=next(lines)
            res=re.findall('^booster\[\d+\]:$', line)
            if not res:
                break
            else:
                boost_num = re.findall('\d+', res[0])
                if int(boost_num[0]) > 0:
                    s.append('+')
        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # append leaf value in format: \t...\score+=value
        res=re.findall('(\t*)(\d+):leaf=(.*)$', line)
        if res:
            tabs,numb,score=res[0]

            try:
                ss=float(score)
                score='{:.10f}'.format(ss)
            except ValueError:
                None
            s[-1] += str(score)
            return
        # -------------------------------------------------------------------- #

        res=re.findall('(\t*)\d+:\[([_a-zA-Zа-яА-Я0-9/\.+ -]+)([><=]+)(.+)\] yes=(\d+),no=(\d+)', line)
        if res:
            tabs,var,sign,const,yes,no=res[0]
            #var=translit(var, 'ru', reversed=True).replace('/','_')
            res=re.findall(',missing=(\d+)', line)
            missing=None
            if res:
                missing=res[0]

            try:
                ss=float(const)
                const='{:.10f}'.format(ss)
            except ValueError:
                None

            if not missing:
                s.append('{0:}case'.format(tabs))
                s.append('{0:}\twhen {1:} {2:} {3:} then '.format(tabs, var, sign, const))
            elif missing==yes:
                s.append('{0:}case'.format(tabs))
                s.append('{0:}\twhen {1:} is null or {1:} {2:} {3:} then '.format(tabs, var, sign, const))
            elif missing==no:
                s.append('{0:}case'.format(tabs))
                s.append('{0:}\twhen {1:} {2:} {3:} then '.format(tabs, var, sign, const))

            f(s)
            s.append(''.join(tabs)+'\telse ');       
            f(s)
            s.append(''.join(tabs)+'end');
            return
        print(line)
        raise ValueError

    s=['/* XGB */',
    ]

    try:
        while(True):
            f(s)
    except StopIteration:   
        s[-1] += ' as xgb_score'
        pass

    with open(filename_sql, 'w', newline='') as f:
        f.write('\n'.join(s))


def xgb2blz(filename_xgb, filename_blz, cat_columns=[]):
    """Plain XGBoost model dump file converter to Blaze script.
    
    Arguments:
        filename_xgb (str): name of file with xgb model in native format (saved with booster.dump_model(...)).
        filename_blz (str): name of file to store Blaze script.
        cat_str_columns (list of str, optional): list of categorical columns. Defaults to [].
    """
    
    lines=open(filename_xgb) #input file from xgb (saved by booster.dump_model())

    def mp(var):
        mapping={

        }
        if var in mapping:
            return mapping[var]
        return var

    def f(s):

        while(True):
            line=next(lines)
            res=re.findall('^booster\[\d+\]:$', line)
            if not res:
                break

        res=re.findall('(\t*)(\d+):leaf=(.*)$', line)
        if res:
            tabs,numb,score=res[0]

            try:
                ss=float(score)
                score='{:.10f}'.format(ss)
            except ValueError:
                None
            s.append('{}score+={};'.format(tabs,score))
            return

        res=re.findall('(\t*)\d+:\[([_a-zA-Zа-яА-Я0-9/\.+ -]+)([><=]+)(.+)\] yes=(\d+),no=(\d+)', line)
        if res:
            tabs,var,sign,const,yes,no=res[0]
            #var=translit(var, 'ru', reversed=True).replace('/','_')
            res=re.findall(',missing=(\d+)', line)
            missing=None
            if res:
                missing=res[0]


            flag = False
            for c in cat_columns:
                if var.startswith(c):
                    const=var[len(c)+1:]               
                    var=c 
                    if const=='nan':
                        const=None
                    else:
                        try:
                            ss=float(const)
                            if ss==int(ss):
                                const='{}'.format(int(ss))
                        except ValueError:
                            None

                        if var in cat_columns:
                            #print(var)
                            const='"'+const+'"'
                    flag=True
                    break
            var=mp(var)
            if flag:
                if const==None:
                    s.append('{0:}if({1:} is known) then {{'.format(tabs, var))
                else:
                    s.append('{0:}if({1:} is unknown or {1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                '''
                elif not missing:
                    s.append('{0:}if({1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                elif missing==yes:
                    s.append('{0:}if({1:} is unknown or {1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                elif missing==no:
                    s.append('{0:}if({1:} is known and {1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                '''
            else:

                try:
                    ss=float(const)
                    const='{:.10f}'.format(ss)
                except ValueError:
                    None


                if not missing:
                    s.append('{0:}if({1:} {2:} {3:}) then {{'.format(tabs, var, sign, const))
                elif missing==yes:
                    s.append('{0:}if({1:} is unknown or {1:} {2:} {3:}) then {{'.format(tabs, var, sign, const))
                elif missing==no:
                    s.append('{0:}if({1:} is known and {1:} {2:} {3:}) then {{'.format(tabs, var, sign, const))

            f(s)
            s.append(''.join(tabs)+'}');
            s.append(''.join(tabs)+'else {');       
            f(s)
            s.append(''.join(tabs)+'}');
            return
        print(line)
        raise ValueError

    s=['score is an real initially 0;',
    ]

    try:
        while(True):
            f(s)
    except StopIteration:        
        pass

    with open(filename_blz, 'w', newline='') as f:
        f.write('\n'.join(s))


def xgb_to_sql(filename_xgb, filename_sql, table_name, cat_numeric_columns=[], cat_str_columns=[]):
    """Plain XGBoost model dump file converter to SQL script.
    
    Args:
        filename_xgb (str): name of file with xgb model in native format (saved with booster.dump_model(...)).
        filename_sql (str): name of file to store sql script.
        table_name (str): name of table with predictors.
        cat_numeric_columns (list of str, optional): list of numerical categorical columns. Defaults to [].
        cat_str_columns (list of str, optional): list of string categorical columns. Defaults to [].
    """
    cnt = 0

    def f():
        nonlocal cnt
        while True:
            line = next(lines)
            line = line[:-1]
            res = re.findall('^booster\[\d+\]:$', line)
            if not res:
                break
            else:
                if len(s)>1:
                    cnt += 1
                    s.append('+')
                    s.append('---------------------{}'.format(cnt))
        res=re.findall('(\t*)(\d+):leaf=(.*)$', line)
        if res:
            tabs, numb, score=res[0]
            s.append('{}{}'.format(tabs,score))
            return

        res = re.findall('(\t*)\d+:\[([_a-zA-Zа-яА-Я0-9/\.+ -]+)([><=]+)(.+)\] yes=(\d+),no=(\d+)', line)
        # print('x{}y'.format(res))
        if res:
            tabs, var, sign, const, yes, no=res[0]
            res = re.findall(',missing=(\d+)', line)
            missing = None
            if res:
                missing = res[0]

            flag = False
            for c in cat_columns:
                if var.startswith(c):
                    const = var[len(c)+1:]
                    var = c
                    if const == 'nan':
                        const = None
                    elif c in cat_str_columns:
                        const = '\'' + const + '\''
                    flag = True
                    break
            # var=mp(var)
            if flag:
                if const is None:
                    s.append('{0:}case when {1:} is not null then'.format(tabs, var))
                else:
                    s.append('{0:}case when {1:} is null or {1:} {2:} {3:} then'.format(tabs, var, '<>', const))
                '''
                elif not missing:
                    s.append('{0:}if({1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                elif missing==yes:
                    s.append('{0:}if({1:} is unknown or {1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                elif missing==no:
                    s.append('{0:}if({1:} is known and {1:} {2:} {3:}) then {{'.format(tabs, var, '<>', const))
                '''
            else:
                if not missing:
                    s.append('{0:}case when {1:} {2:} {3:} then'.format(tabs, var, sign, const))
                elif missing == yes:
                    s.append('{0:}case when {1:} is null or {1:} {2:} {3:} then'.format(tabs, var, sign, const))
                elif missing == no:
                    s.append('{0:}case when {1:} {2:} {3:} then'.format(tabs, var, sign, const))

            f()
            # s.append(''.join(tabs)+'end');
            s.append(''.join(tabs)+'else')
            f()
            s.append(''.join(tabs)+'end')
            return
        raise ValueError
    
    lines = open(filename_xgb)
    cat_columns = cat_numeric_columns + cat_str_columns
    s = ['alter table &table_with_predictors add score_final number;']
    s.append('update &table_with_predictors')
    s.append('set score_final = ')
    s.append('1 - ( 1 / ( 1 + exp ( -  ( ')
    try:
        while True:
            f()           
    except StopIteration:        
        pass
    #s.append('from {}'.format(table_name))
    s.append('))));')
    with open(filename_sql, 'w') as f_out:
        f_out.write('\n'.join(s))
    #uncomment to debug
    #print('\n'.join(s))
