#!/usr/bin/env python
# encoding: utf-8
#
# logger.py
#
# Created by José Sánchez-Gallego on 17 Sep 2017.

import datetime
import logging
import os
import pathlib
import re
import shutil
import sys
import traceback
import warnings
from logging.handlers import TimedRotatingFileHandler

import click
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name


def print_exception_formatted(tp, value, tb):
    """A custom hook for printing tracebacks with colours."""

    tbtext = ''.join(traceback.format_exception(tp, value, tb))
    lexer = get_lexer_by_name('pytb', stripall=True)
    formatter = TerminalFormatter()
    sys.stderr.write(highlight(tbtext, lexer, formatter))


def colored_formatter(record):
    """Prints log messages with colours."""

    colours = {'info': ('blue', 'normal'),
               'debug': ('magenta', 'normal'),
               'warning': ('yellow', 'normal'),
               'print': ('green', 'normal'),
               'error': ('red', 'bold')}

    levelname = record.levelname.lower()

    if levelname == 'error':
        return

    if levelname.lower() in colours:
        levelname_color = colours[levelname][0]
        bold = True if colours[levelname][1] == 'bold' else False
        header = click.style('[{}]: '.format(levelname.upper()), levelname_color, bold=bold)

    message = record.getMessage()

    if levelname == 'warning':
        warning_category_groups = re.match(r'^\w*?(.+?Warning): (.*)', message)
        if warning_category_groups is not None:
            warning_category, warning_text = warning_category_groups.groups()

            warning_category_colour = click.style('({})'.format(warning_category), 'cyan')
            message = '{} {}'.format(click.style(warning_text, fg=None), warning_category_colour)

    sys.__stdout__.write('{}{}\n'.format(header, message))
    sys.__stdout__.flush()

    return


class MyFormatter(logging.Formatter):

    base_fmt = '%(asctime)s - %(levelname)s - %(message)s [%(funcName)s @ %(filename)s]'

    ansi_escape = re.compile(r'\x1b[^m]*m')

    def __init__(self, fmt='%(levelname)s - %(message)s [%(funcName)s @ %(filename)s]'):
        logging.Formatter.__init__(self, fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.base_fmt

        elif record.levelno == logging.getLevelName('PRINT'):
            self._fmt = MyFormatter.base_fmt

        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.base_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = MyFormatter.base_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = MyFormatter.base_fmt

        record.msg = self.ansi_escape.sub('', record.msg)

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


my_fmt = MyFormatter()


class LoggerStdout(object):
    """A pipe for stdout to a logger."""

    def __init__(self, level):
        self.level = level

    def write(self, message):

        if message != '\n':
            self.level(message)

    def flush(self):
        pass


class MyLogger(logging.Logger):
    """This class is used to set up the logging system.

    The main functionality added by this class over the built-in
    logging.Logger class is the ability to keep track of the origin of the
    messages, the ability to enable logging of warnings.warn calls and
    exceptions, and the addition of colourised output and context managers to
    easily capture messages to a file or list.

    """

    INFO = 15

    def __init__(self, name):

        self.fh = None
        self.sh = None
        self.log_filename = None

        super().__init__(name)

    def _catch_exceptions(self, exctype, value, tb):
        """Catches all exceptions and logs them."""

        # Now we log it.
        self.error('Uncaught exception', exc_info=(exctype, value, tb))

        # First, we print to stdout with some colouring.
        print_exception_formatted(exctype, value, tb)

    def _set_defaults(self, log_level=logging.INFO, redirect_stdout=False):
        """Reset logger to its initial state."""

        # Disable astropy logging if present because it uses the same override
        # of showwarning than us and messes up things
        try:
            from astropy import log
            log.disable_warnings_logging()
            log.disable_exception_logging()
        except Exception:
            pass

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel(logging.DEBUG)

        # Set up the stdout handler
        self.fh = None
        self.sh = logging.StreamHandler()
        self.sh.emit = colored_formatter
        self.addHandler(self.sh)

        self.sh.setLevel(log_level)

        # Redirects all stdout to the logger
        if redirect_stdout:
            sys.stdout = LoggerStdout(self._print)

        # Catches exceptions
        sys.excepthook = self._catch_exceptions

        warnings.showwarning = self._showwarning

    def _showwarning(self, *args, **kwargs):

        warning = args[0]

        message = '{0}: {1}'.format(warning.__class__.__name__, args[0])

        mod_path = args[2]

        mod_name = None
        mod_path, ext = os.path.splitext(mod_path)

        for name, mod in list(sys.modules.items()):
            try:
                path = os.path.splitext(getattr(mod, '__file__', ''))[0]
            except Exception:
                continue
            if path == mod_path:
                mod_name = mod.__name__
                break

        if mod_name is not None:
            self.warning(message, extra={'origin': mod_name})
        else:
            self.warning(message)

    # def warning(self, msg, category=None, use_filters=True):
    #     """Custom ``logging.warning``.

    #     Behaves like the default ``logging.warning`` but accepts ``category``
    #     and ``use_filters`` as arguments. ``category`` is the type of warning
    #     we are issuing (defaults to `UserWarning`). If ``use_filters=True``,
    #     checks whether there are global filters set for the message or the
    #     warning category and behaves accordingly.

    #     """

    #     if category is None:
    #         category = UserWarning

    #     n_issued = 0
    #     if category in self.warning_registry:
    #         if msg in self.warning_registry[category]:
    #             n_issued = self.warning_registry[category]

    #     if use_filters:

    #         category_filter = None
    #         regex_filter = None
    #         for warnings_filter in warnings.filters:
    #             if issubclass(category, warnings_filter[2]):
    #                 category_filter = warnings_filter[0]
    #                 regex_filter = warnings_filter[1]

    #         if (category_filter == 'ignore') or (category_filter == 'once' and n_issued >= 1):
    #             if regex_filter is None or regex_filter.search(msg) is not None:
    #                 return

    #         if category_filter == 'error':
    #             raise ValueError(msg)

    #     super(MyLogger, self).warning(msg)

    #     if msg in self.warning_registry[category]:
    #         self.warning_registry[category][msg] += 1
    #     else:
    #         self.warning_registry[category][msg] = 1

    def warning(self, *args, **kwargs):

        super().warning(*args, **kwargs)

    def save_log(self, path):
        shutil.copyfile(self.log_filename, os.path.expanduser(path))

    def start_file_logger(self, name, log_file_level=logging.DEBUG, log_file_path='~/'):
        """Start file logging."""

        log_file_path = pathlib.Path(log_file_path).expanduser() / '{}.log'.format(name)
        logdir = log_file_path.parent

        try:
            logdir.mkdir(parents=True, exist_ok=True)

            # If the log file exists, backs it up before creating a new file handler
            if log_file_path.exists():
                strtime = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S')
                shutil.move(str(log_file_path), str(log_file_path) + '.' + strtime)

            self.fh = TimedRotatingFileHandler(str(log_file_path), when='midnight', utc=True)
            self.fh.suffix = '%Y-%m-%d_%H:%M:%S'
        except (IOError, OSError) as ee:
            warnings.warn('log file {0!r} could not be opened for writing: {1}'.format(
                          log_file_path, ee), RuntimeWarning)
        else:
            self.fh.setFormatter(my_fmt)
            self.addHandler(self.fh)
            self.fh.setLevel(log_file_level)

        self.log_filename = log_file_path


orig_logger = logging.getLoggerClass()
logging.setLoggerClass(MyLogger)
log = logging.getLogger(__name__)
log._set_defaults()  # Inits sh handler
logging.setLoggerClass(orig_logger)
