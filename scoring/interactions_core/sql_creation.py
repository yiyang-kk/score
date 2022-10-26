import re
REGEXP = r'\(([-\d{inf}]+\.*\d*);([-\dinf]+\.*\d*)[\)\]]'

# this is unparsing the already created column and index names using regexp and thus obtaining the interval values.
# i decided just to wrap everything in a function which can be called externally


def sql_condition(isnumeric, var_name, **kwargs):
    sql_string = ''

    if isnumeric:
        if len(kwargs) != 2:
            raise TypeError(
                'Function sql_condition() needs to have 4 arguments if numerical variable is being processed. {} arguments passed to the function.'.format(len(kwargs)+2))
        brdrs = sorted([val for key, val in kwargs.items()])
        include_and = 0
        if brdrs[0] != '-inf':
            sql_string += '{} > {}'.format(var_name, brdrs[0])
            include_and = 1
        if brdrs[1] != 'inf':
            if include_and:
                sql_string += ' and '
            sql_string += '{} <= {}'.format(var_name, brdrs[1])
    else:
        if len(kwargs) != 1:
            raise TypeError(
                'Function sql_condition() needs to have 3 arguments if categorical variable is being processed. {} arguments passed to the function.'.format(len(kwargs)+2))
        categories = [val for key, val in kwargs.items()]
        sql_string += '{} in (\'{}\')'.format(var_name, '\',\''.join(categories))

    return sql_string


def create_sql(interaction_variable, row_variable, column_variable, row_isnumeric, column_isnumeric, categories, row_cat_alias, col_cat_alias):
    col_brdrs = []
    row_brdrs = []
    pattern = re.compile(REGEXP)
    categories
    if len(categories.columns) == 1 and categories.columns[0] == 'null':
        print('Warning:   Columns contain only null values. Interaction SQL code not saved!')
    if len(categories.index) == 1 and categories.index[0] == 'null':
        print('Warning:   Rows contain only null values. Interaction SQL code not saved!')

    if column_isnumeric:
        for col in categories.columns:
            if col != 'null':
                interval_borders = pattern.search(col)
                low_brdr = interval_borders.group(1)
                upp_brdr = interval_borders.group(2)

                if len(col_brdrs) == 0:
                    col_brdrs += [low_brdr, upp_brdr]
                else:
                    col_brdrs += [upp_brdr]
            else:
                col_brdrs += ['null']

    if row_isnumeric:
        for row in categories.index:
            if row != 'null':
                interval_borders = pattern.search(row)
                low_brdr = interval_borders.group(1)
                upp_brdr = interval_borders.group(2)

                if len(row_brdrs) == 0:
                    row_brdrs += [low_brdr, upp_brdr]
                else:
                    row_brdrs += [upp_brdr]
            else:
                row_brdrs += ['null']

    sql_string = 'case\n'
    conditions = []
    for row in range(len(categories.index)):
        categ = 0
        if row_isnumeric:
            if categories.index[row] == 'null':
                row_condition = '{} is null'.format(row_variable)
            else:
                row_low = row_brdrs[row]
                row_upp = row_brdrs[row+1]
                row_condition = sql_condition(row_isnumeric, row_variable, row_low=row_low, row_upp=row_upp)
        else:
            if categories.index[row] == 'null':
                row_condition = '{} is null'.format(row_variable)
            else:
                if row_cat_alias:
                    row_condition = sql_condition(row_isnumeric, row_variable, [
                                                  key for key, val in row_cat_alias.items() if val == categories.index[row]])
                else:
                    row_condition = sql_condition(row_isnumeric, row_variable, [categories.index[row]])

        if len(row_condition) > 0:
            include_and = 1
        else:
            include_and = 0

        for col in range(len(categories.columns)):
            if column_isnumeric:
                if col == 0:
                    col_low = col_brdrs[col]
                    col_upp = col_brdrs[col+1]
                    categ = categories.iloc[row, col]
                elif categories.columns[col] == 'null':
                    conditions += [(row_condition, col_condition, categ)]
                    col_condition = '{} is null'.format(column_variable)
                    conditions += [(row_condition, col_condition, categories.iloc[row, col])]
                elif categ == categories.iloc[row, col]:
                    col_upp = col_brdrs[col+1]
                else:
                    col_condition = sql_condition(column_isnumeric, column_variable, col_low=col_low, col_upp=col_upp)
                    conditions += [(row_condition, col_condition, categ)]
                    categ = categories.iloc[row, col]
                    col_low = col_upp
            else:
                if col == 0:
                    categ = categories.iloc[row, col]
                    if col_cat_alias:
                        col_categs = [key for key, val in col_cat_alias.items() if val == categories.columns[col]]
                    else:
                        col_categs = categories.columns[col]
                elif categories.columns[col] == 'null':
                    col_condition = sql_condition(column_isnumeric, column_variable, categories=col_categs)
                    conditions += [(row_condition, col_condition, categ)]
                    col_condition = '{} is null'.format(column_variable)
                    conditions += [(row_condition, col_condition, categories.iloc[row, col])]
                elif categ == categories.iloc[row, col]:
                    if col_cat_alias:
                        col_categs += [key for key, val in col_cat_alias.items() if val == categories.columns[col]]
                    else:
                        col_categs += categories.columns[col]
                else:
                    col_condition = sql_condition(column_isnumeric, column_variable, categories=col_categs)
                    conditions += [(row_condition, col_condition, categ)]
                    categ = categories.iloc[row, col]
                    if col_cat_alias:
                        col_categs = [key for key, val in col_cat_alias.items() if val == categories.columns[col]]
                    else:
                        col_categs = categories.columns[col]

    conditions = sorted(conditions, key=lambda x: x[2])

    for row_condition, col_condition, categ in conditions:
        if len(row_condition) == 0 and len(col_condition) == 0:
            sql_string += '    when {} is not null and {} is not null\tthen {}\n'.format(
                row_variable, column_variable, categ)
        elif len(row_condition) > 0 and len(col_condition) == 0:
            sql_string += '    when {} and {} is not null\tthen {}\n'.format(row_condition, column_variable, categ)
        elif len(row_condition) == 0 and len(col_condition) > 0:
            sql_string += '    when {} is not null and {}\tthen {}\n'.format(row_variable, col_condition, categ)
        elif len(row_condition) > 0 and len(col_condition) > 0:
            sql_string += '    when {} and {}\tthen {}\n'.format(row_condition, col_condition, categ)

    sql_string += 'end as i_{}_{}'.format(row_variable, column_variable)
    return sql_string
