"""
Debugging utils.
"""
import timeit
import sys
import pprint
import inspect
from contextlib import contextmanager
from .io.printing import PrintSuppress


try:
    "Inject REPL() at any place in your code to start interactive debugging"
    from IPython import embed as REPL
except ImportError:
    print('IPython not found. Use a less powerful REPL instead.', file=sys.stderr)
    import code
    def REPL():
        # collect all variables outside this scope
        local = {}
        # set stack context to 0 to avoid the slow loading of source file
        for sinfo in inspect.stack(0):
            local.update(sinfo[0].f_globals)
            local.update(sinfo[0].f_locals)
        code.interact(local=local)


def timecode(statement, **kwargs):
    """
    Same as timeit.timeit, except that the statement can contain any variable 
    in the outer scope. Suppress warning.
    """
    with PrintSuppress(no_out=False, no_err=True):
        return timeit.timeit(statement, 
                             setup='from __main__ import *', 
                             **kwargs)


def pp(*objs, h='', **kwargs):
    """
    Args:
      *objs: objects to be pretty-printed
      h: header string
      **kwargs: other kwargs to pass on to ``pprint()``
    """
    if h:
        print('=' * 20, h, '=' * 20)
    for obj in objs:
        pprint.pprint(obj, indent=4, **kwargs)
    if h:
        print('=' * (42 + len(h)))


def dprint(*varnames):
    """debug-print

    Auto-find local variable's name and print its value.
    """
    record=inspect.getouterframes(inspect.currentframe())[1]
    frame=record[0]
    for name in varnames:
        print(name, '==>', eval(name,frame.f_globals,frame.f_locals))


@contextmanager
def noop_context(*args, **kwargs):
    """
    For debugging purposes. 
    Placeholder context manager that does nothing.
    """
    yield
    
    
def halt():
    input('enter to continue ... ')


def stop():
    sys.exit(0)
