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

import datetime
import logging
import os

DEFAULT_FILE_NAME = "log_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
DEFAULT_LOGGER_LEVEL = logging.INFO
DEFAULT_LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FOLDER = "log"


def create_logger(
    log_name=None,
    log_folder=DEFAULT_LOG_FOLDER,
    log_filename=DEFAULT_FILE_NAME,
    log_level=DEFAULT_LOGGER_LEVEL,
    log_format=DEFAULT_LOGGER_FORMAT,
    handlers="both",
):
    """
    Creates logging object and returns it.

    Args:
        log_name (str, optional): name of the logging object (default: DEFAULT_LOG_NAME)
        log_filename (str, optional): file address where the log should be saved (default: DEFAULT_FILE_NAME)
        log_level (int, optional): logger level which should be stored (default: logging.DEBUG)
        log_format (str, optional): formatting of the logging entries (default: DEFAULT_LOGGER_FORMAT)
        handlers (str, optional): 'both', 'file', 'stream' - which handlers should be added (default: 'both')

    Returns:
        logger: logging.Logger object
    """
    # TODO: create log folder and save there
    if not log_name:
        log_name = __name__
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # create the logging file handler
    formatter = logging.Formatter(log_format)
    if not len(logger.handlers):
        if handlers in ["both", "file"]:
            fh = logging.FileHandler(os.path.join(log_folder, log_filename))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if handlers in ["both", "stream"]:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    logger.info("Logger created.")
    return logger


def kill_logger(logger):
    """
    Function for logger killing - removes all handlers.

    Args:
        logger (logging.Logger): logger for which handlers should be closed.
    """

    logger.info("Killing logger...")
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


import subprocess


def runrealcmd(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        stderr=subprocess.STDOUT,
        bufsize=1,
        close_fds=False,
    )
    for line in iter(process.stdout.readline, b""):
        print(line.rstrip().decode("utf-8"))
    process.stdout.close()
    process.wait()
