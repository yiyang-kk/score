# FAQ:

### One click install does not work.

See [Installation troubleshooting](install_troubleshooting.md).

### I can't see table of contents / manipulation with PSW is hard
_Install Jupyter Extensions._

To get the most out of Jupyter notebooks we recommend using several exntensions. The easiest way is to install a collection put together in this project on GitHub [jupyter_contrib_nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions). If you use Anaconda simply get everything with

` conda install -c conda-forge jupyter_contrib_nbextensions` 

You should now see a tab on your Jupyter homepage named *Nbextensions*. There select which extensions you want to use. Recommended are:
* Table of contents (2)
* ExecutionTime
* Variable Inspector *(can have negative effect on performance)*

### Something does not work.

Report the bug on issue tracker here: https://git.homecredit.net/risk/python-scoring-workflow/issues