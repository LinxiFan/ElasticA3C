"""
Printing utils:

- Context manager-based printing streams.
- Convert data structures to pretty string representation. 
"""

import os
import sys
import datetime
from io import StringIO
from .filesys import f_expand
from ..common import include_exclude

def stdnull():
    "/dev/null stream"
    return open(os.devnull, 'w')


def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ================ Convert types to pretty strings ==============
def time_now_str(fmt_str='%m-%d_%H.%M.%S'):
    """
    https://docs.python.org/2/library/time.html#time.strftime
    %m - month; %d - day; %y - year
    %H - 24 hr; %I - 12 hr; %M - minute; %S - second; %p - AM or PM
    """
    return datetime.datetime.now().strftime(fmt_str)


def seconds_str(seconds):
    "Convert seconds to str `HH:MM:SS`"
    return datetime.timedelta(seconds=seconds)


def dict_str(D, 
             sep='=',
             item_sep=', ',
             key_format='',
             value_format='',
             enclose=('{', '}')):
    """
    Pretty string representation of a dictionary. Works with Unicode.

    Args:
      sep: "key `sep` value"
      item_sep: separator between key-value pairs
      key_format: same format string as in str.format()
      value_format: same format string as in str.format()
      enclose: a 2-tuple of enclosing symbols
    """
    assert len(enclose) == 2
    itemstrs = []
    for key, value in D.items():
        itemstrs.append(u'{{:{}}} {} {{:{}}}'
                        .format(key_format, sep, value_format)
                        .format(key, value))
    return enclose[0] + item_sep.join(itemstrs) + enclose[1]


def list_str(L, 
             sep=', ',
             item_format='',
             enclose=None):
    """
    Pretty string representation of a list or tuple. Works with Unicode.

    Args:
      sep: separator between two list items
      item_format: same format string as in str.format()
      enclose: a 2-tuple of enclosing symbols. 
          default: `[]` for list and `()` for tuple.
    """
    if enclose is None:
        if isinstance(L, tuple):
            enclose = ('(', ')')
        else:
            enclose = ('[', ']')
    else:
        assert len(enclose) == 2
    item_format = u'{{:{}}}'.format(item_format)
    itemstr = sep.join(map(lambda s: item_format.format(s), L))
    return enclose[0] + itemstr + enclose[1]


def attribute_str(obj, sep=',\n', mapsep='=', 
                  include_filter=None, exclude_filter=None):
    """
    Args:
      obj: an object with various attrs
      sep: separator for each attribute (lines)
      mapsep: name `mapsep` value (e.g. x => 3)
      include_filter, exclude_filter: see include_exclude()

    Returns:
      pretty printed string of all attributes of the object,
      sorted by alphabetical order.
    
    Sample usage: an object of argparse.Namespace, returned by parse_arg()
    """
    all_attrs = sorted(include_exclude(vars(obj), 
                                       include_filter=include_filter, 
                                       exclude_filter=exclude_filter))
    return sep.join(['{} {} {}'.format(attr, mapsep, getattr(obj, attr))
                     for attr in all_attrs])


# ============= Print redirections ================
class PrintRedirection(object):
    """
    Context manager: temporarily redirects stdout and stderr
    """
    def __init__(self, stdout=None, stderr=None):
        """
        Args:
          stdout: if None, defaults to sys.stdout, unchanged
          stderr: if None, defaults to sys.stderr, unchanged
        """
        if stdout is None:
            stdout = sys.stdout
        if stderr is None:
            stderr = sys.stderr
        self._stdout, self._stderr = stdout, stderr

    def __enter__(self):
        self._old_out, self._old_err = sys.stdout, sys.stderr
        self._old_out.flush()
        self._old_err.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        return self
            
    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        # restore the normal stdout and stderr
        sys.stdout, sys.stderr = self._old_out, self._old_err
    
    def flush(self):
        "Manually flush the replaced stdout/stderr buffers."
        self._stdout.flush()
        self._stderr.flush()


class PrintToFile(PrintRedirection):
    """
    Print to file and save/close the handle at the end.
    """
    def __init__(self, out_file=None, err_file=None):
        """
        Args:
          out_file: file path
          err_file: file path. If the same as out_file, print both stdout 
              and stderr to one file in order.
        """
        self.out_file, self.err_file = out_file, err_file
        if out_file:
            out_file = f_expand(out_file)
            self.out_file = open(out_file, 'w')
        if err_file:
            err_file = f_expand(err_file)
            if err_file == out_file: # redirect both stdout/err to one file
                self.err_file = self.out_file
            else:
                self.err_file = open(f_expand(err_file), 'w')
        
        PrintRedirection.__init__(self, 
                                  stdout=self.out_file, 
                                  stderr=self.err_file)
    
    def __exit__(self, *args):
        PrintRedirection.__exit__(self, *args)
        if self.out_file:
            self.out_file.close()
        if self.err_file:
            self.err_file.close()


def PrintSuppress(no_out=True, no_err=True):
    """
    Args:
      no_out: stdout writes to sys.devnull
      no_err: stderr writes to sys.devnull
    """
    out_file = os.devnull if no_out else None
    err_file = os.devnull if no_err else None
    return PrintToFile(out_file=out_file, err_file=err_file)


class PrintString(PrintRedirection):
    """
    Redirect stdout and stderr to strings.
    """
    def __init__(self):
        self.out_stream = StringIO()
        self.err_stream = StringIO()
        PrintRedirection.__init__(self, 
                                  stdout=self.out_stream, 
                                  stderr=self.err_stream)
    
    def stdout(self):
        "Returns: stdout as one string."
        return self.out_stream.getvalue()
    
    def stderr(self):
        "Returns: stderr as one string."
        return self.err_stream.getvalue()
        
    def stdout_by_line(self):
        "Returns: a list of stdout line by line, ignore trailing blanks"
        return self.stdout().rstrip().split('\n')

    def stderr_by_line(self):
        "Returns: a list of stderr line by line, ignore trailing blanks"
        return self.stderr().rstrip().split('\n')

