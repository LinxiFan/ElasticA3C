import logging
import functools
import sys
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

"""
Define more verbosity levels. The higher level number gives more verbosity.
Usage: log.log(<one_of_levels_below, .....)
    or log.log1(), log2() ...
"""
INFO0 = INFO  # default, least verbose
INFO1 = INFO - 1
INFO2 = INFO - 2
INFO3 = INFO - 3
INFO4 = INFO - 4
INFO5 = INFO - 5

# hack the built-in getLevelName function to have nice names for INFO1-5
_level_names = {
    CRITICAL : 'CRITICAL',
    ERROR : 'ERROR',
    WARNING : 'WARNING',
    INFO : 'INFO',
    DEBUG : 'DEBUG',
    INFO1 : 'INFO1',
    INFO2 : 'INFO2',
    INFO3 : 'INFO3',
    INFO4 : 'INFO4',
    INFO5 : 'INFO5'
}


# hack the standard lib function
logging.getLevelName = lambda level : _level_names.get(level, '')


def arg_verbosity(parsed_arg):
    """
    Get logging verbosity level from command line arg, e.g. ``-v`` or ``-vvv`` 
    parsed_arg must have ``verbose`` keyword, and optionally ``debug`` flag
    if --debug, set to DEBUG level
    """
    if not hasattr(parsed_arg, 'verbose'):
        raise KeyError('parsed_arg does not have `verbosity` flag.')
    if hasattr(parsed_arg, 'debug') and parsed_arg.debug:
        return logging.DEBUG
    else:
        return logging.INFO - parsed_arg.verbose


class LoggerFormatAdapter(object):
    """
    logger = LoggerFormatAdapter(logger)
    fully supports positional/keyword new formatter string syntax

    Example:
    
    ::
        logger.info("Float {1:>7.3f} and int {0} with {banana} and {apple:.6f}", 
                    66, 3.141592, apple=7**.5, banana='str')
        # if the first arg isn't a format string, treat as if print statement
        # you can specify a `sep=` arg under this case. `sep` defaults to space
        logger.info({3:'apple', 4:'banana'}, 4.5, 'asdf')
        logger.info('I am not a format string', 66, {'key':'value'}, sep=', ')
    
    Custom verbosity level. The higher the more strings printed. 
    log.info1(), .info2() ... info5(). 
    corresponding levels: INFO1 to INFO5
    """

    def __init__(self, logger):
        self.logger = logger
        CLS = LoggerFormatAdapter # convenient alias
        # generate these methods dynamically
        for name in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(CLS, name, CLS.__gen_log_methods(name))

        # generate verbosity methods info1(), info2() ... info5() dynamically
        # the higher level, the more strings printed
        for level_n in range(5):
            setattr(CLS, 'info{}'.format(level_n),
                    functools.partial(self.log, logging.INFO - level_n))

    @staticmethod
    def unwrap(logger):
        """
        If it's LoggerFormatAdapter, get internal. Otherwise do nothing.
        """
        if isinstance(logger, LoggerFormatAdapter):
            return logger.logger
        else:
            return logger


    @staticmethod
    def __gen_log_methods(name):
        "Meta function that generates log.debug(), .info(), .error(), etc."
        def _log(self, msg, *args, **kwargs):
            # e.g. logging.DEBUG
            level = getattr(logging, name.upper())
            if self.logger.isEnabledFor(level):
                msg, kwargs = self.__process_msg(msg, *args, **kwargs)
                getattr(self.logger, name)(msg, **kwargs)
        return _log
    
    
    def __process_msg(self, msg, *args, **kwargs):
        if isinstance(msg, str) and '{' in msg and '}' in msg:
            fmt_kwargs = {}
            for key, value in kwargs.items():
                if not key in ['level', 'msg', 'args', 'exc_info', 'extra']:
                    fmt_kwargs[key] = value
                    # must remove unsupported keyword for internal args
                    kwargs.pop(key)
            msg = msg.format(*args, **fmt_kwargs)
        else:
            # if `msg` isn't a format string, then treat it like the first arg to print
            args = (msg,) + args
            # if 'sep' is provided, we will use the custum separator instead
            if 'sep' in kwargs:
                sep = kwargs['sep']
                kwargs.pop('sep')
            else:
                sep = ' '
            # e.g. "{}, {}, {}" if sep = ", "
            msg = sep.join([u'{}'] * len(args)).format(*args)
        return msg, kwargs


    def exception(self, msg, *args, **kwargs):
        """
        Logs a message with level ERROR on this logger. 
        Exception info is always added to the logging message. 
        This method should only be called from an exception handler.
        """
        if self.logger.isEnabledFor(logging.ERROR):
            msg, kwargs = self.__process_msg(msg, *args, **kwargs)
            kwargs["exc_info"] = 1
            self.logger.error(msg, **kwargs)
    
    
    def log(self, level, msg, *args, **kwargs):
        """
        Log with user-defined level, e.g. INFO_V0 to INFO_V5
        """
        if self.logger.isEnabledFor(level):
            msg, kwargs = self.__process_msg(msg, *args, **kwargs)
            self.logger.log(level, msg, **kwargs)
    
    
    def section(self, msg=None, level=INFO, sep='=', repeat=20):
        """
        Display a section segment line
        Args:
          msg: to be displayed in the middle of the sep line
          level: defaults to INFO
          sep: symbol to be repeated for a long segment line
          repeat: 'sep' * repeat, the length of the segment string
        """
        if self.logger.isEnabledFor(level):
            if msg:
                msg = ' {} '.format(msg)
            else:
                msg = ''
            self.logger.log(level, 
                            u'{0}{msg}{0}'.format(sep * repeat, msg=msg))
        

    def setLevel(self, level):
        self.logger.setLevel(level)


def configure_logger(logger, 
                     level=None, 
                     log_file=None,
                     overwrite=False,
                     print_level=False,
                     time_format='hm',
                     stream=sys.stderr):
    """
    Mutator function. Supports both plain and LoggerFormatAdapter loggers. 
    Args:
      level: None to retain the original level of the logger
      log_file: None to print to console only
      overwrite: if False, append to the log log_file, otherwise overwrite it
      print_level: if True, display `INFO> ` before the message
      time_format:
        - `full`: %m/%d %H:%M:%S
        - `hms`: %H:%M:%S
        - `hm`: %H:%M
        - if contains '%', will be interpreted as a format string
        - None
      stream:
        - defaults to sys.stderr
        - None: do not print to any stream
    
    """
    logger = LoggerFormatAdapter.unwrap(logger)
        
    if level:
        logger.setLevel(level)

    levelname = '%(levelname)s> ' if print_level else ''
    if not time_format:
        formatter = logging.Formatter(levelname + u'%(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s '+levelname+u'%(message)s',
                                      datefmt=(time_format 
                                               if '%' in time_format
                                               else {'full':'%m/%d %H:%M:%S', 
                                                     'hms': '%H:%M:%S', 
                                                     'hm': '%H:%M'}
                                                     .get(time_format, ' ')))
    if stream:
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, 'w' if overwrite else 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class LoggerManager(object):
    """
    log = LoggerManager().get_logger('name') to ensure that the same 
    logger is used across modules. 
    """
    _loggers = {}
    _default_handler = logging.StreamHandler()
    
    def __init__(self):
        raise NotImplementedError('Singleton class cannot be instantiated')
    

    @staticmethod
    def get_logger(name, apply_format_adapter=True):
        """
        If name already exists, get the existing singleton logger
        If doesn't, create a new basic logger with no extra config. 

        Args:
          apply_format_adapter: wrap the basic logger with LoggerFormatAdapter
            to enable new style formatter string
        """
        if name not in LoggerManager._loggers:
            logger = logging.getLogger(name)
            logger.addHandler(LoggerManager._default_handler)
            logger.setLevel(logging.DEBUG)
            if apply_format_adapter:
                logger = LoggerFormatAdapter(logger)
            LoggerManager._loggers[name] = logger
            
        return LoggerManager._loggers[name]


    @staticmethod
    def configure(name, *config_args, **config_kwargs):
        """
        Args:
          name:
            - str: find the associated logger with the str name
            - logger instance: if exist in LoggerManager, directly configure it
          *config_args, **config_kwargs:
            identical to configure_logger() function
        """
        if isinstance(name, str):
            if name not in LoggerManager._loggers:
                raise KeyError('{} logger not created yet.'.format(name))
            # remove the old handler
            logger = LoggerManager._loggers[name]
        else:
            if name not in LoggerManager._loggers.values():
                raise KeyError('logger instance not found.')
            logger = name
                
        (LoggerFormatAdapter.unwrap(logger)
             .removeHandler(LoggerManager._default_handler))
        configure_logger(logger, *config_args, **config_kwargs)
    