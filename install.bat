conda config --append channels conda-forge
conda install --file requirements.txt
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix qgrid