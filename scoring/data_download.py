"""
##################################
Easing the Scorecard Data Download
##################################

- _Author: [Jan Hynek](jan.hynek@homecredit.eu)_  
- _TESTED ON:_
   - India
   - Indonesia

*******
Vision:
*******

 __Make the data download as seamless (and thus easy) as possible.

********
Mission:
********

- Connect Python with Oracle SQL database.
- Create tables from the __vector__ master table in several Python commands. 
- Provide a fast way to download them.

************
Explanation:
************

__DataCreator__ is the class that orchestrates the data creation.

The old way to download the data was to use SQL scripts, which are more or less standardised.  
You needed to change dates on several places, change the data families,  
change the dates from-to, for which you need to download.

However, these scripts are run several times,  
with some small, slight changes.  
And all these small changes had to be inputted by hand,  
and it was easy to miss all the spots, where all the changes should be. 


Requirements
************

 - Python:
     ```
     pandas
     sqlalchemy
     xlrd
     cx_Oracle
     openpyxl
     ```
 - Oracle SQL 64-bit Instant Client
     - I know, I know. Installation is pain.
     - However most of our infrastructure is running Oracle.
     - And this step needs to be solved only once.
  


*********************************************
Install Oracle SQL 64-bit Instant Client:
*********************************************
  - [Download from here.](https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html)
  - TESTED: 11.2 and 19.3 works on WIN10. 19.3 does not need oracle registration.
  - TESTED: ONLY 11.2 works on WIN7.
  - Installation:
    - Unzip folder on some convenient place, **which you will not change in near future**
    - Open start, and look up 'Environment Variables' Click on Environment Variables.
    - If possible, edit system variable 'Path' (requires admin account). 
    - If not, edit user variable 'Path'.
    - IMPORTANT: DO NOT DELETE ANYTHING FROM THIS 'Path' VARIABLE. APPEND ONLY.
    - Add to the 'Path' the folder where you unzipped the Oracle SQL Instant client. Separate it using semicolon.
    - to test the installation: open commandline (start -> cmd)
    - run `sqlplus`
    - you should see something like this:
    
``
SQL*Plus: Release 11.2.0.2.0 Production on Wed Apr 15 13:56:54 2020

Copyright (c) 1982, 2010, Oracle.  All rights reserved.
``

## Quick run

- modify `config/db_config.cfg` with your info
   - input your username, password, database name
- modify `config/script_config.cfg
   - input base table (which does have column `SKP_CREDIT_CASE`)
   - input from/to dates
- run following:
    ``
    dc = DataCreator(
        db_config=r"config\db_config_template.cfg",
        script_config=r"config\script_config.cfg",
        log_level=10
    )
    dc.create_attributes()
    ``
  and get all attributes in `attributes.xlsx`.


#####################################
DataCreator structure & documentation
#####################################

*********
Templates
*********

It consists from `templates`. These templates are the original SQL scripts, with _placeholders_. You can find them in the folder of the same name.

*******
Configs
*******

However, IMPORTANT are configs.

There are two mandatory configs.  
One is optional, if you have [tnsnames.ora](http://oradmin.homecredit.net/tnsnames.ora) correctly specified in your PC.  
All of them can be found in folder `configs`.



"""
import os
import ast
import configparser

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from textwrap import dedent

from .data_download_core._utils import create_logger
from .data_download_core.config.connection_strings import CONNECTION_STRINGS
from .data_download_core.templates.attribute_selection import ATTRIBUTE_SELECTION_TEMPLATE
from .data_download_core.templates.generic_table import (
    GENERIC_TABLE_CREATE_TEMPLATE,
    GENERIC_TABLE_INSERT_TEMPLATE,
)
from .data_download_core.templates.temp_table import TEMP_TABLE_TEMPLATE
from tqdm import tqdm


class DataCreator:
    """
    Class to create data from Oracle SQL DB

    ################
    DATABASE CONFIG:
    ################

    **Example:**

    ``
    [db]
    user = MY_USERNAME[AP_UWI]
    pw = mY_sECRET_pASSWORD
    database = HDWIN.HOMECREDIT.IN
    schema = ap_uwi
    ``

    ***************************
    Individual config elements:
    ***************************
    - ``user``
        - the user name used for logging in the database.
        - can have the prespecified schema as well.
    - ``pw``
        - your database password.
    - ``database``
        - if you have ``tnsnames.ora`` correctly specified:
            - just input the name of the database.
        - if you don't:
            - either install ``tnsnames.ora`` from here: [tnsnames.ora](http://oradmin.homecredit.net/tnsnames.ora)
                - and ask EmbedIT for help
            - or [add a new connection string](#new-connection-string-addition).

    - ``schema``
        - schema where should all final tables be stored. 
        - usually ``ap_risk``, ``ap_uwi``

    #############
    SCRIPT CONFIG
    #############

    **Example:**
        
    ``
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
    ``

    ***************************
    Individual config elements:
    ***************************
    - __predefined__
        - ``base``
            - name of the base table, which defines for which credit cases will data be downloaded.
            - must contain following columns
                - ``SKP_CREDIT_CASE``
                - ``DATE_DECISION``
        - ``attribute_selection_date_decision_start``
            - date limiting the dataset for the attribute list - start
        - ``attribute_selection_date_decision_end``
            - date limiting the dataset for the attribute list - end
            - NOTE
                - this is to define such attributes which are currently used in the vector.
                - one month of data is usually enough
        - ``prefix``
            - prefix to be used in the names of individual SQL tables.
            - should be initials.
    - __dataset__
        - ``date_decision_start``
        - ``date_decision_end``
            - limits of the data. from - to
    - __definitions__
        - ``families``
            - specification of individual families
            - Python dictionary
            - using [pandas ``query`` method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)
            - these queries are used upon [import_table](#import-table)
        - ``name_replacements``
            - replacements of the names which are too long for the database
            - Python dictionary
            - ``<part_of_name_that_is_long>`` : ``<replacement>``
            - as this dictionary is parsed as a text, some elements need escaping.
                - i.e. 
                    - ``[`` must be written as ``\[``
                    - ``.`` -> ``\.``
    
    """
    def __init__(
        self, db_config, script_config, download_tnsnames_ora=True, log_name="DataCreator", **logger_kwargs
    ):
        self.logger = create_logger(log_name=log_name, **logger_kwargs)
        self.connection_strings = self._get_tnsnames_ora() if download_tnsnames_ora else CONNECTION_STRINGS
        self.db_config = self._get_config(db_config)
        self.script_config = self._get_config(script_config)

        user = self.db_config["db"]["user"]
        pw = self.db_config["db"]["pw"]
        db = self.db_config["db"]["database"]

        connection_string = f"oracle://{user}:{pw}@{CONNECTION_STRINGS.get(db,db)}"
        self.engine = create_engine(connection_string
            , echo=False
        )
        self.logger.info(f"Created connection:\n{connection_string}")

        self.prefix = self.script_config["predefined"]["prefix"]

        self.schema = self.db_config["db"]["schema"]
        self.attributes = None
        self.import_table = None
        self.tables = {}
        self.temp_table = f"{self.prefix}temp_table"

        self.excel_address = "attributes.xlsx"
        self.csv_used = False
        self.import_column = "IMPORT_THIS_ATTRIBUTE"

    def _get_config(self, config_address):
        config = configparser.ConfigParser()
        config.read(config_address)
        self._log_config(config=config, config_name=os.path.basename(config_address))
        return config


    def _log_config(self, config, config_name):
        if config:
            config_string = ""
            for part in config:
                if part == "DEFAULT":
                    continue
                config_string += f"\n\n\t[{part}]"
                for argument in config[part]:
                    config_string += f"\n\t\t{argument} = {config[part][argument][:100]}"
                    if len(config[part][argument]) > 100:
                        config_string += "... (truncated)."

            self.logger.info(
                dedent(
                    f"""Imported '{config_name}' with parameters:
                    {config_string}
                    """
                )
            )
        else:
            message = dedent(
                f"""
            Incorrect '{config_name}'. 
            Have you checked whether you specified correct config address?
            """
            )
            self.logger.error(message)
            raise ValueError(message)


    def _get_tnsnames_ora(self):
        self.logger.info("Downloading tnsnames.ora ...")
        import requests

        r = requests.get("http://oradmin.homecredit.net/tnsnames.ora")
        tnsnames_ora = r.content.decode("utf-8")
        tnsnames_ora_dict = {}

        # fmt: off
        for string in (
            tnsnames_ora
            .replace("# Migrated  https://rtp.embedit.cz/", "")
            .split("#")[1:]):
        # fmt: on
            splitted = string.replace("\r", "").split("\n")
            tnsnames_ora_dict[splitted[1][:-1]] = "\n".join(splitted[2:])
      
        self.logger.info("Downloaded")
        return tnsnames_ora_dict
    def create_attributes(self, attributes_path=None, limit=None, overwrite_attributes=False):
        # TODO: get rid off csv saving - xlsx only (?)

        try:
            import openpyxl
            import xlrd
        except ImportError as e:
            self.logger.warning(str(e))
            self.logger.warning("Attributes will be saved as csv instead of xlsx")



        if self.attributes is None:
            if os.path.exists(self.excel_address) and not overwrite_attributes:

                self.attributes = pd.read_excel(self.excel_address)
                self.logger.info(dedent(f"""
                Atrributes already exists (in '{self.excel_address}')
                and `overwrite_attributes` argument is False - reading from already saved file.
                """))
                return
            else:

                self.logger.info("Downloading attributes")
                self.attributes = self._download_attributes_from_db(limit)

        try:
            if attributes_path is not None:
                self.excel_address = os.path.join(attributes_path, self.excel_address)
            self.attributes.to_excel(self.excel_address)
            
        except ModuleNotFoundError as e:
            if attributes_path is not None:
                self.excel_address = os.path.join(attributes_path, "attributes.csv")
            self.logger.warning(str(e))
            self.excel_address = "attributes.csv"
            self.csv_used = True
            self.attributes.to_csv(self.excel_address, index=False)
        self.logger.info(f"Saving excel file into {self.excel_address}")
        print(
            f"Please modify '{self.import_column}' column "
            f"in file '{self.excel_address}'"
        )
        return

    def _download_attributes_from_db(self, limit):
        sql = ATTRIBUTE_SELECTION_TEMPLATE.format(
            base=self.script_config["predefined"]["base"],
            attribute_selection_date_decision_start=self.script_config["predefined"][
                "attribute_selection_date_decision_start"
            ],
            attribute_selection_date_decision_end=self.script_config["predefined"][
                "attribute_selection_date_decision_end"
            ],
            where_condition="" if limit is None else f"where rownum < {limit}",
        )
        self.logger.debug(sql)
        attributes = (
            pd.read_sql(sql=sql, con=self.engine)
            .set_index("skp_scoring_vector_attribute")
            .sort_values("text_vector_attr_full_path")
        )

        attributes[self.import_column] = 1
        return attributes
        

    def load_attributes(self, excel=None, csv=None):
        """
        Function to load attributes from prespecified table

        Args:
            excel (str, optional): Address of the excel file. Defaults to None, then csv is read.
            csv (str, optional): Address of the attributes file in excel form. Defaults to None.
            When neither is set, it reads from pre-specified excel file (read during previous actions)

        Returns:
            pd.DataFrame: dataframe with read attributes
        """
        if excel is None and csv is None:
            self.logger.info("Reading from already saved excel")
            if self.csv_used:
                self.logger.warning("Reading from csv")
                self.attributes = pd.read_csv(self.excel_address)
            else:
                self.attributes = pd.read_excel(self.excel_address)
        elif excel:
            self.attributes = pd.read_excel(excel)
        elif csv:
            self.attributes = pd.read_csv(excel)

        return self.attributes

    def get_import_table(self, attributes=None):
        """Create filtered and preformatted table for attributes, 
        which were marked to be downloaded from db

        Raises:
            AttributeError: When no attributes were downloaded before

        Returns:
            [type]: [description]
        """

        if self.attributes is None and attributes is None:
            raise AttributeError("You need to obtain attributes first")

        self._create_raw_import_table()

        self._split_fullpath()

        self._assign_families()

        self._replace_colnames()

        self._create_partitions()

        self._register_tables()

        self._check_column_names()
        self.logger.info(
            "Created Python import table (with specified families and colnames)"
        )
        return self.import_table

    def _check_column_names(self):
        for colname in self.import_table["final_colname"]:
            if len(colname) > 30:
                self.logger.error(
                    f"\n\n'{colname}' is too long for the database.\n"
                    " Please modify 'script_config.cfg' and shorten this name."
                )

    def _create_raw_import_table(self):
        self.import_table = self.attributes.query(
            "IMPORT_THIS_ATTRIBUTE == 1"
        ).reset_index(drop=True)
        self.import_table["vector_depth"] = self.import_table[
            "text_vector_attr_full_path"
        ].str.count("\[\]")
        self.import_table.loc[:, "final_colname"] = [
            items[-1]
            for items in self.import_table["text_vector_attr_full_path"]
            .str.split(".")
            .tolist()
        ]

    def _create_partitions(self):
        self.import_table["dt_block"] = (
            self.import_table["family"]
            + "_"
            + self.import_table["vector_depth"].astype(str)
        )

    def _replace_colnames(self):
        replacements = ast.literal_eval(
            self.script_config["definitions"]["name_replacements"]
        )
        self.import_table["final_colname"] = self.import_table[
            "final_colname"
        ].str.lower()
        self.import_table["final_colname"] = self.import_table["final_colname"].replace(
            replacements, regex=True
        )

        self.import_table["final_colname"] = (
            self.import_table["final_colname"]
            + "_"
            + self.import_table["skp_scoring_vector_attribute"].astype(str)
        )

    def _assign_families(self):
        self.import_table.loc[:, "family"] = "other"
        families = ast.literal_eval(self.script_config["definitions"]["families"])
        for family, query in families.items():
            self.import_table.loc[self.import_table.eval(query), "family"] = family

    def _split_fullpath(self):
        splitted = pd.DataFrame(
            self.import_table["text_vector_attr_full_path"].str.split(".").tolist()
        )

        splitted.columns = [f"path_{c}" for c in list(splitted)]
        self.import_table = self.import_table.join(splitted)

    def send_import_table_to_db(self):
        """Sends import table to database. 
        Import table prespecifies, which arguments will be downloaded from db.
        It also prespecifies table partitioning, and thus which tables are going to be created.
        """
        from sqlalchemy.types import Integer, String

        if self.import_table is None:
            _ = self.get_import_table()

        self.logger.info("Sending selected attributes into DB (~ETA: 1 minute)")
        table_name = f"{self.prefix}vector_item_names"
        self.import_table[
            ["skp_scoring_vector_attribute", "final_colname", "dt_block"]
        ].to_sql(
            name=table_name,
            con=self.engine,
            schema=self.schema,
            if_exists="replace",
            index=False,
            dtype={
                "skp_scoring_vector_attribute": Integer,
                "final_colname": String(64),
                "dt_block": String(64),
            },
        )
        self.logger.info(f"Data successfully sent into DB as '{table_name}'.")

    def create_temp_table_db(self):
        """Creates subset of vector table in database.
        """
        partition_string = ",\n".join(
            [f"partition values('{b}')" for b in self.import_table["dt_block"].unique()]
        )
        sql = TEMP_TABLE_TEMPLATE.format(
            prefix=self.prefix,
            schema=self.schema,
            temp_table_name=self.temp_table,
            base=self.script_config["predefined"]["base"],
            partition_string=partition_string,
            date_decision_start=self.script_config["dataset"]["date_decision_start"],
            date_decision_end=self.script_config["dataset"]["date_decision_end"],
        )


        self._execute_or_drop_and_execute(sql, self.temp_table)

    def _register_tables(self):
        for dt_block in self.import_table["dt_block"].unique():
            self.logger.debug(
                f"Table {self.__get_generic_table_name(dt_block)} will be created"
            )
            self.tables[self.__get_generic_table_name(dt_block)] = {
                "created": False,
                "dt_block": dt_block,
                "vector_depth": int(dt_block[-1]),
            }

    def __get_generic_table_name(self, dt_block):
        return f"{self.prefix}data_vector_{dt_block}"

    def _execute_or_drop_and_execute(self, sql, table_name):

        try:
            self.logger.debug(sql)
            self.engine.execute(text(sql).execution_options(autocommit=True))
        except sqlalchemy.exc.SQLAlchemyError as err:
            self.logger.debug(str(err))
            drop_sql = f"drop table {table_name}"
            _ = self.engine.execute(text(drop_sql).execution_options(autocommit=True))
            self.logger.debug(drop_sql)
            self.engine.execute(text(sql).execution_options(autocommit=True))
            self.logger.debug(sql)

    def create_data_vector_table(
        self, dt_block, vector_depth, table_name, variable_string
    ):
        """Create feature table for given data vector subset and given import table.


        Args:
            dt_block (str): partitioning
            vector_depth (num): number specifying whether the client attributes is 1:1, 1: array, or 1: array of arrays
            table_name (str): name of the final table
            variable_string (str): casting of variables, to not have the apostrophes in names
        """

        sql = GENERIC_TABLE_CREATE_TEMPLATE.format(
            prefix=self.prefix,
            num_group_creation_string=self._get_num_group_string(
                vector_depth, "1 as num_group{i},\n"
            ),
            table_name=table_name,
            dt_block=dt_block,
            variable_string=variable_string,
        )
        self._execute_or_drop_and_execute(sql, table_name=table_name)
        self.engine.execute(
            text(f"truncate table {table_name}").execution_options(autocommit=True)
        )
        self.logger.debug(f"truncate table {table_name}")

    @staticmethod
    def _get_num_group_string(vector_depth, string_to_format):
        if vector_depth == 0:
            return ""
        else:
            result = ""
            for i in range(1, vector_depth + 1):
                result += string_to_format.format(i=i)
            return result


    def _get_variable_string(self, dt_block):
        return ",\n".join(
            [
                f"\t\t'{name}' as {name}"
                for name in self.import_table.query(f'dt_block == "{dt_block}"')[
                    "final_colname"
                ]
            ]
        )

    def insert_into_data_vector_table(
        self, dt_block, vector_depth, table_name, variable_string
    ):
        """Insert data to pre-cretaed tables with given column names

        Args:
            dt_block (str): data partition
            vector_depth (num): number specifying whether the client attributes is 1:1, 1: array, or 1: array of arrays
            table_name (str): name of the final table
            variable_string (str): sql sttmnt casting of variables into correct format
        """

        sql = GENERIC_TABLE_INSERT_TEMPLATE.format(
            table_name=table_name,
            num_group_insert_string_1=self._get_num_group_string(
                vector_depth, "t.num_group{i},\n"
            ),
            num_group_insert_string_2=self._get_num_group_string(
                vector_depth, "sv.num_group_position_{i} as num_group{i},\n",
            ),
            base=self.script_config["predefined"]["base"],
            temp_table=self.temp_table,
            date_decision_start=self.script_config["dataset"]["date_decision_start"],
            date_decision_end=self.script_config["dataset"]["date_decision_end"],
            dt_block=dt_block,
            prefix=self.prefix,
            variable_string=variable_string,
        )
        self.logger.debug(sql)
        self.engine.execute(sql)

    def orchestrate_data_table_creation(self):
        """
        Orchestrate creation of the final data tables.
        Needs inputting of the configuation files.
        """

        for table_name, table_attributes in self.tables.items():
            dt_block = table_attributes["dt_block"]
            variable_string = self._get_variable_string(dt_block)
            vector_depth = table_attributes["vector_depth"]

            self.create_data_vector_table(
                table_name=table_name,
                vector_depth=vector_depth,
                dt_block=dt_block,
                variable_string=variable_string,
            )
            self.insert_into_data_vector_table(
                table_name=table_name,
                vector_depth=vector_depth,
                dt_block=dt_block,
                variable_string=variable_string,
            )
            self.logger.info(f"Created table {table_name}")

    def orchestrate_data_download(self, tables=None, limit=None, chunksize=1000, save_to=None):
        """
        Orchestrate download of all tables.

        Args:
            tables (list, optional): List of all tables to be downloaded. Defaults to None.
            limit (int, optional): max # of rows to be downloaded. Defaults to None.
            chunksize (int, optional): size of the chunks. Defaults to 1000.
            save_to (path-like, optional): where the final data should be saved. If None, dont save. Defaults to None.

        Returns:
            dict[name: pd.DataFrame]: dictionary with name: pd Dataframe of all data
        """

        # TODO: ADD CHUNKING BY DAYS
        if tables is None:
            tables = self.tables
        if isinstance(tables, str):
            tables = [tables]
        self.final_data = {}
        for table_name in tables:
            dataset = self._get_table(table_name, limit, chunksize)
            self.final_data[table_name] = dataset
            if save_to:
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                dataset.to_csv(os.path.join(save_to, f"{table_name}.csv"))
        return self.final_data

    def _get_table(self, table_name, limit, chunksize):
        size = pd.read_sql(
            sql=f"select count(*) from {self.schema}.{table_name}", con=self.engine
        ).iloc[0, 0]
        if size == 0:
            self.logger.warning(f"{self.schema}.{table_name} is empty.")
            return pd.DataFrame()
        if limit:
            size = min(size, limit)
        if not chunksize:
            chunksize = size

        self.logger.info(f"Downloading {size} rows from {self.schema}.{table_name}.")

        start = 0

        steps = list(range(start, size, chunksize)) + [size]
        data_parts = []

        for start, end in tqdm(zip(steps, steps[1:])):
            self.logger.info(f"{start} - {end}")

            part = self._download_data_part(table_name, start, end).copy()
            data_parts += [part]
            # breakpoint()

        return pd.concat(data_parts)

    def _download_data_part(self, table_name, start=None, end=None):
        sql = f"""

        select M.* 
          from (
                select {self.schema}.{table_name}.*, 
                       row_number() over (order by SKP_CREDIT_CASE) as rn
                  from {self.schema}.{table_name} ) M
         where rn >  {start} 
           and rn <= {end}
            """
        self.logger.debug(sql)
        data = pd.read_sql(sql=sql, con=self.engine).set_index('skp_credit_case')
        return data
