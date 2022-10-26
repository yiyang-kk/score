# Easing the Scorecard Data Download

- _Author: [Jan Hynek](jan.hynek@homecredit.eu)_  
- _TESTED ON:_
   - India
   - Indonesia

# Vision:
 __Make the data download as seamless (and thus easy) as possible.__
# Mission:
- Connect Python with Oracle SQL database.
- Create tables from the __vector__ master table in several Python commands. 
- Provide a fast way to download them.

# Explanation:
__DataCreator__ is the class that orchestrates the data creation (duuh).

The old way to download the data was to use SQL scripts, which are more or less standardised.  You needed to change dates on several places, change the data families, change the dates from-to, for which you need to download.

However, these scripts are run several times, with some small, slight changes.  
And all these small changes had to be inputted by hand, and it was easy to miss all the spots, where all the changes should be. 


# Requirements
 - Python:
     ```
     pandas
     sqlalchemy
     xlrd
     cx_Oracle
     openpyxl
     ```
 - Oracle SQL __64-bit__ Instant Client
     - I know, I know. It is a pain.
     - However most of our infrastructure is running Oracle.
     - And this step needs to be solved only once.
  
     


![](https://media1.tenor.com/images/6987362b140a83a2cba19942a48ddb20/tenor.gif?itemid=9258253)


## Install Oracle SQL __64-bit__ Instant Client:
  - [Download from here.](https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html)
  - TESTED: 11.2 and 19.3 works on WIN10. 19.3 does not need oracle registration.
  - TESTED: ONLY 11.2 works on WIN7.
  - __Installation:__
    - Unzip folder on some convenient place, _which you will not change in near future_
    - Open start, and look up 'Environment Variables' Click on Environment Variables.
    - If possible, edit system variable 'Path' (requires admin account). 
    - If not, edit user variable 'Path'.
    - __IMPORTANT__: DO NOT DELETE ANYTHING FROM THIS 'Path' VARIABLE. APPEND ONLY.
    - Add to the 'Path' the folder where you unzipped the Oracle SQL Instant client. Separate it using semicolon.
    - to test the installation: open commandline (start -> cmd)
    - run `sqlplus`
    - you should see something like this:
    
```
SQL*Plus: Release 11.2.0.2.0 Production on Wed Apr 15 13:56:54 2020

Copyright (c) 1982, 2010, Oracle.  All rights reserved.
```

## Quick run

- modify `config/db_config.cfg` with your info
   - input your username, password, database name
- modify `config/script_config.cfg
   - input base table (which does have column `SKP_CREDIT_CASE`)
   - input from/to dates
- run following:
    ```
    dc = DataCreator(
        db_config=r"config\db_config_template.cfg",
        script_config=r"config\script_config.cfg",
        log_level=10
    )
    dc.create_attributes()
    ```
  and get all attributes in `attributes.xlsx`.
- but wait, there is more - just follow the code here.


# __DataCreator__ structure & documentation

## Templates
It consists from `templates`. These templates are the original SQL scripts, with _placeholders_. You can find them in the folder of the same name.

## Configs
However, __IMPORTANT__ are configs.

There are two mandatory configs.  
One is optional, if you have [tnsnames.ora](http://oradmin.homecredit.net/tnsnames.ora) correctly specified in your PC.  
All of them can be found in folder `configs`.

___

### __DATABASE CONFIG__: `config/db_config.cfg` <a class="anchor" id="db-config"></a>

#### Example:
```
[db]
user = MY_USERNAME[AP_UWI]
pw = mY_sECRET_pASSWORD
database = HDWIN.HOMECREDIT.IN
schema = ap_uwi
```

#### Individual config elements:
- `user`
    - the user name used for logging in the database.
    - can have the prespecified schema as well.
- `pw`
    - your database password.
- `database`
    - if you have `tnsnames.ora` correctly specified:
        - just input the name of the database.
    - if you don't:
        - either install `tnsnames.ora` from here: [tnsnames.ora](http://oradmin.homecredit.net/tnsnames.ora)
            - and ask EmbedIT for help
        - or [add a new connection string](#new-connection-string-addition).

 - `schema`
     - schema where should all final tables be stored. 
     - usually `ap_risk`, `ap_uwi`

___


### __CONNECTION STRINGS__: `config/connection_strings.py`

 - optional with correctly specified `tnsnames.ora`
 
#### Example:

```
from textwrap import dedent

CONNECTION_STRINGS = {
    "HDWHQ.HOMECREDIT.NET": "(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=DBHDWHQ.HOMECREDIT.NET)(PORT=1521)))(CONNECT_DATA=(SERVICE_NAME=HDWHQ.HOMECREDIT.NET)))",
    "HDWIN.HOMECREDIT.IN": dedent(
        """
            (DESCRIPTION =
            (ADDRESS=(PROTOCOL = TCP)(HOST = INCL02.IN.PROD)(PORT = 1521))
            (CONNECT_DATA =
                (UR = A)
                (SERVICE_NAME = HWIN_USR_DEV.HOMECREDIT.IN)
                (SERVER = DEDICATED)
            )
            )
        """
    ),
}
```
#### Individual config elements:
- Single python element `CONNECTION_STRINGS` 
    - dictionary with structure: `<name>` : `<connection string>`
    
#### New Connection string addition <a class="anchor" id="new-connection-string-addition"></a>:

- Find your database here: [tnsnames.ora](http://oradmin.homecredit.net/tnsnames.ora)
- copy the connection string, such as this one:
```
HDWIN.HOMECREDIT.IN=
(DESCRIPTION =
  (ADDRESS=(PROTOCOL = TCP)(HOST = INCL02.IN.PROD)(PORT = 1521))
  (CONNECT_DATA =
    (UR = A)
    (SERVICE_NAME = HWIN_USR_DEV.HOMECREDIT.IN)
    (SERVER = DEDICATED)
  )
 )
```
- edit the connection string, to be valid for dictionary, such as:
```
    "HDWIN.HOMECREDIT.IN": dedent(
        """
            (DESCRIPTION =
            (ADDRESS=(PROTOCOL = TCP)(HOST = INCL02.IN.PROD)(PORT = 1521))
            (CONNECT_DATA =
                (UR = A)
                (SERVICE_NAME = HWIN_USR_DEV.HOMECREDIT.IN)
                (SERVER = DEDICATED)
            )
            )
        """
```
- add such config to the dictionary.
- NOTE: 
    - `textwrap.dedent` function is used to remove the whitespace.
    - You can add the whole connection string in a single line, that would work as well.

___

### __SCRIPT CONFIG__: `config/script_config.cfg` <a class="anchor" id="script-config"></a>
 
#### Example:
     
```
[predefined]
base = ap_uwi.jh_super_new_scorecard_base
attribute_selection_date_decision_start = 2020-01-15
attribute_selection_date_decision_end = 2020-01-30
prefix = qx_

[dataset]
date_decision_start = 2020-01-15
date_decision_end = 2020-01-30

[definitions]
families = {'static': 'vector_depth == 0',
    'prevAppl': '(path_0 == "applicantData") & path_1 == "previousApplicationDetails[]"',
    'credit': 'path_0 == "credit"',
    'other': 'family.isna()'}
name_replacements = {"\[]": "",
    "\.": "_",
    "address": "addr",
    "utility": "util",
    "employment": "empl",
    "registered": "regist",
    "payment": "pmt",
    }
```
#### Individual config elements:
- __predefined__
    - `base`
        - name of the base table, which defines for which credit cases will data be downloaded.
        - must contain following columns
            - `SKP_CREDIT_CASE`
            - `DATE_DECISION`
    - `attribute_selection_date_decision_start`
        - date limiting the dataset for the attribute list - start
    - `attribute_selection_date_decision_end`
        - date limiting the dataset for the attribute list - end
        - NOTE
            - this is to define such attributes which are currently used in the vector.
            - one month of data is usually enough
    - `prefix`
        - prefix to be used in the names of individual SQL tables.
        - should be initials.
- __dataset__
    - `date_decision_start`
    - `date_decision_end`
        - limits of the data. from - to
- __definitions__
    - `families`
        - specification of individual families
        - Python dictionary
        - using [pandas `query` method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)
        - these queries are used upon [import_table](#import-table)
    - `name_replacements`
        - replacements of the names which are too long for the database
        - Python dictionary
        - `<part_of_name_that_is_long>` : `<replacement>`
        - as this dictionary is parsed as a text, some elements need escaping.
            - i.e. 
                - `[` must be written as `\[`
                - `.` -> `\.`
