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
