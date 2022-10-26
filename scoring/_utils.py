
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


import datetime
import logging
import configparser


DEFAULT_FILE_NAME = 'log_{}.log'.format(datetime.datetime.now().strftime("%y%m%d%H%M"))
DEFAULT_LOGGER_LEVEL = logging.DEBUG
DEFAULT_LOGGER_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def create_logger(logname=None, filename=DEFAULT_FILE_NAME,
                  loglevel=DEFAULT_LOGGER_LEVEL, logformat=DEFAULT_LOGGER_FORMAT):
    """
    Creates logging object and returns it.

    Args:
        logname (str, optional): name of the logging object (default: DEFAULT_LOG_NAME)
        filename (str, optional): file address where the log should be saved (default: DEFAULT_FILE_NAME)
        loglevel (int, optional): logger level which should be stored (default: logging.DEBUG)
        logformat (str, optional): formatting of the logging entries (default: DEFAULT_LOGGER_FORMAT)

    Returns:
        logging.Logger
    """
    # TODO: create log folder and save there
    if not logname:
        logname = __name__
    logger = logging.getLogger(logname)
    logger.setLevel(loglevel)

    # create the logging file handler
    fh = logging.FileHandler(filename)
    sh = logging.StreamHandler()
    formatter = logging.Formatter(logformat)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)\

    # add handler to logger object
    if not len(logger.handlers):
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info('Logger created.')
    return logger


def kill_logger(logger):
    """
    Function for logger killing - removes all handlers.

    Args:
        logger (logging.Logger): logger for which handlers should be closed.
    """

    logger.info('Killing logger...')
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)



