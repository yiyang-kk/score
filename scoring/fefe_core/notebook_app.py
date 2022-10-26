from .front_end import callbacks
import copy
from .front_end.app import app
from .front_end.layout import get_layout
from .front_end.config_handler import CONFIG, CONFIG_CLEAN
from .front_end.tab_callbacks.ratio_callbacks import register_ratio_callbacks
from .front_end.tab_callbacks.simple_callbacks import register_simple_callbacks
from .front_end.tab_callbacks.time_since_callbacks import register_time_since_callbacks

from .structures.metadata import Metadata

NOTEBOOK_FRONT_END_APPLICATION_FIRST_RUN = True


def notebook_front_end(data, port=8051, host="127.0.0.1", **kwargs):
    """
    This function starts the interactive front-end.
    It is actually just a wrapper for several subfunctions
        - metadata creation
        - callback registration (from metadata)
        - calling `dash.app`
    When you would like to continue after the front-end, interrupt the kernel.

    Args:
        data (pd.DataFrame): data for which metadata should be calculated and displayed.
            These data are just quickly processed - they are not displayed
            in the front-end wholly. The processing however can take a while
        host (str, optional): default '127.0.0.1'. either 'auto' or ip address
            if auto, ip address of the current server is obtained.
            Otherwise, this argument is fed to `dash.app` arguments.
        port (int, optional): default 8051. port on which the app should be running
        kwargs: key-word arguments to be passed to `dash.app`

    """
    global NOTEBOOK_FRONT_END_APPLICATION_FIRST_RUN
    global CONFIG

    CONFIG = copy.deepcopy(CONFIG_CLEAN)
    metadata = Metadata(data)
    app.layout = get_layout(metadata)
    if NOTEBOOK_FRONT_END_APPLICATION_FIRST_RUN:
        register_simple_callbacks(metadata)
        register_ratio_callbacks(metadata)
        register_time_since_callbacks(metadata)
        NOTEBOOK_FRONT_END_APPLICATION_FIRST_RUN = False
    if host == "auto":
        import socket

        host = socket.gethostbyname(socket.gethostname())

    app.run_server(debug=False, port=port, host=host, **kwargs)

