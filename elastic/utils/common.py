"""
Common utility functions

TODO:
  need more cleanup
"""
from __future__ import print_function
import os
import sys
import functools
import traceback
import argparse
import random
import inspect
import collections
import itertools
from contextlib import contextmanager


NaN = float('nan')
Inf = float('inf')


def is_python3():
    return sys.version_info.major >= 3


if is_python3():
    long = int
    iteritems = dict.items
else:
    iteritems = dict.iteritems


def import_parent_dir():
    """
    Import scripts from parent directory.
    """
    sys.path.insert(0, '..')
    

def merge_dicts(*dicts):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dic in dicts:
        result.update(dic)
    return result


def merge_attr_dicts(*dicts):
    """
    Returns: merged AttributeDict
    """
    return AttributeDict(merge_dicts(*dicts))


def reverse_dict(dic):
    """
    Convert {key: value} to {value: key}
    """
    return dict((value, key) for key, value in iteritems(dic))


def pop_head(array, n):
    """
    pop n items from the head of the list. Input will be modified.
    """
    ans = array[:n]
    del array[:n]
    return ans


def pop_tail(array, n):
    """
    pop n items from the tail of the list. Input will be modified.
    """
    ans = array[-n:]
    del array[-n:]
    return ans


def list_dim(lis):
    """
    Return: dimensions of deep nested sequences. Only consider the 
    dim of the first element. Strings are not sequences. 
    """
    if is_sequence(lis):
        if len(lis) == 0:
            inner_dims = []
        else:
            inner_dims = list_dim(lis[0])
        return [len(lis)] + inner_dims
    else:
        return []


def parse_int_list(ints):
    """
    For command line int list parsing
    syntax: 
    - `1-5` => [1,2,3,4,5] inclusive
    - `3-10^5,9` => [3,4,6,7,8,10]  3-10 excluding 5 and 9. 
    - `3-12^5-8` => [3,4,9,10,11,12]  3-12 excluding 5-8. 
    - `2,4,7` => [2,4,7]
    - `0` => [0]
    """
    ints = ints.strip()
    if '^' in ints:
        parts = map(str.strip, ints.split('^'))
        assert len(parts) == 2
        assert '-' in parts[0]
        return sorted(list(set(parse_int_list(parts[0]))
                           - set(parse_int_list(parts[1]))))
    elif '-' in ints:
        ints = map(int, ints.split('-'))
        assert len(ints) == 2
        return list(range(ints[0], ints[1]+1))
    else:
        return map(int, ints.split(','))


def enlist(obj, N=1, check=False):
    """
    Useful for repeating default option multiple times
    Args:
      check: if True, check len(obj) == N
    Returns:
      If obj is a list, return after checking
      Otherwise, repeat it into N-element list
    """
    if isinstance(obj, list):
        if check and len(obj) != N:
            raise ArgumentError('length != {}'.format(N))
        else:
            return obj
    else:
        return [obj] * N


def cum_sum(seq):
    """
    Cumulative sum (include 0)
    """
    s = 0
    cumult = [0]
    for n in seq:
        s += n
        cumult.append(s)
    return cumult
#     return [sum(seq[:i]) for i in range(len(seq) + 1)]
    

def include_exclude(array, include_filter=None, exclude_filter=None):
    """
    Include an element if:
    1. include_filter is None
    2. include_filter is a list and contains the element
    3. include_filter is a lambda and returns True on the element
    AND then exclude an element if:
    1. exclude_filter is not None
    2. exclude_filter is a list and contains the element
    3. exclude_filter is a lambda and returns True on the element
    """
    ans = []
    for a in array:
        if ((include_filter is None 
                 or is_sequence(include_filter) 
                    and a in include_filter 
                 or callable(include_filter) 
                    and include_filter(a))
            and (exclude_filter is None
                 or is_sequence(exclude_filter) 
                    and not a in exclude_filter 
                 or callable(exclude_filter) 
                    and not exclude_filter(a))):
            ans.append(a)
    return ans


def min_at(values):
    "Returns: (min, min_i)"
    if not values:
        return None, None
    return min( (v, i) for i, v in enumerate(values) )


def max_at(values):
    "Returns: (max, max_i)"
    if not values:
        return None, None
    return max( (v, i) for i, v in enumerate(values) )


def sum_pow(p, N_begin, N_end):
    """
    Power summation, N inclusive.
    \sum_{n=N_begin}^{N_end} p^n
    """
    return (p**(N_end+1) - p**N_begin) / (p - 1.0)


def ceildiv(a, b):
    """
    Ceiling division, equivalent to math.ceil(1.0*a/b) but much faster.
    ceildiv(19, 7) == 3
    ceildiv(21, 7) == 3
    ceildiv(22, 7) == 4
    """
    return - (-a // b)


def is_div(a, b):
    " Returns: bool - does `a` divide `b`. "
    return int(a) % int(b) == 0


def rand_choice(*choices):
    """
    Args:
      choices: randomly return an element from the list
      can be either packed or unpacked
    """
    choices = pack_args(choices)
    if len(choices) <= 1:
        raise ArgumentError('must have at least one choice')
    return choices[random.randint(0, len(choices) - 1)]


def rint(*N):
    """
    One arg N:
        Random int from [0, N] inclusive
    Two args N_low, N_high:
        Random int from [N_low, N_high] inclusive
    """
    N = pack_args(N)
    assert 1 <= len(N) <= 2
    if len(N) == 1:
        N = [0, N[0]]
    return random.randint(*N)


def shuffled_indices(N):
    indices = list(range(N))
    random.shuffle(indices)
    return indices


def flatten2d(list2d):
    return list(itertools.chain.from_iterable(list2d))


def pack_args(args):
    """
    Process `*args` variable-length positional args
    Enable to function to accept either unpacked or packed args
    
    def f1(*args):
        args = pack_args(args)
    
    Now f1 can be used in all the following ways: 
    - f1(a, b, c)
    - f1([a, b, c])
    - f1(*[a, b, c])
    - f1([a,b,c],[a2,b2,c2]) -> will raise error
    """
    assert is_sequence(args)
    if args and is_sequence(args[0]):
        assert len(args) == 1, 'only 1 iterable allowed in varargs'
        return args[0]
    else:
        return args


class CheckInterval(object):
    def __init__(self, interval, init_value=0):
        self.interval = interval
        self.counter = init_value

    
    def trigger(self, value):
        """
        Returns: True if the value is over an interval
        """
        if value > self.counter + self.interval:
            self.counter = value
            return True
        else:
            return False


class CheckPeriodic(object):
    def __init__(self, period, first=False):
        """
        first: if True, triggers at the first time
        """
        self.period = period
        self.counter = -1 if first else 0

    
    def trigger(self):
        """
        Returns: True if `trigger()` has been called `period` times
        """
        self.counter += 1
        return self.counter % self.period == 0


# ======================== ArgParser ========================
class _SingleMetavarFormatter(argparse.HelpFormatter):
    "Helper for better metavar display in ArgParser"
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is `-s, --long`
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is:
            #    -s <METAVAR>, --long
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                ## THIS IS THE PART REPLACED
                # for option_string in action.option_strings:
                    # parts.append('%s %s' % (option_string, args_string))
                parts.extend(action.option_strings)
                # treat nargs different
                if action.nargs and action.default:
                        parts[-1] += ' default={}'.format(action.default)
                parts[0] += ' ' + args_string
            return ', '.join(parts)
        
class ArgParser(object):
    """
    example
    ArgParser = ArgParser(
                   [('b', 'bar'), 23, 'will be int, float will error'],
                   ['foo', ['opt1', 'opt2', 'opt3'], 'arg choices'],
                   [('a', 'app', 'apple'), 34.5, 'myarg.app to get value'],
                   ['dud', None, '--dud is a required string'],
                   [('u', 'ununun'), False, 'store_true to ununun'],
                   ['gug', (None, float), '--gug is a required float'],
                   [('p', 'pollo'), (44, float), 'myarg.pollo saved as float'],
                   ['evy', ArgParser.Nargs([1,2,3]), 'variable number of args']
                   description='This is my hello world program',
                   epilogue='goodbye zai jian sayonala')
    myarg = ArgParser.parse()

    Args:
      *arg_infos must be a list of 3 specifiers:
            [name or (names,), 
            default or (default, type) or [list of choices] or ArgParser.Nargs, 
            help_string]
      **config_kwargs
        - any kwargs that can be used in argparse.ArgumentParser constructor
        - e.g. description=, epilog=
        - config_file TODO 
    
    Note: 
      The following options are pre-configured
      --verbosity, or -vvv (number of v's indicate the level of verbosity)
      --debug: turn on debugging mode
      Any option that has underscore in it automatically supports hyphen, and 
      vice versa. E.g. `step_size` supports both `--step_size` and `--step-size`

    Warning:
      Remember to add a comma after each [...], to prevent
      weird error like `TypeError: list indices must be integers, not tuple`
    """

    class Narg(object):
        """
        Variadic number of args for an option. 
        See `nargs=` option in parser.add_argument()
        Args:
          default: a list of default Nargs. If None, this option will be required
          nargs: <N>, '?', '+', '*', will be passed to `add_argument()`
        """
        def __init__(self, default, nargs='*'):
            if not default: # emtpy list or None
                self.dtype = str
            elif isinstance(default, (list, tuple)):
                self.dtype = type(default[0])
            else:
                self.dtype = type(default)
            self.default = default
            self.nargs = nargs
            
    
    def __init__(self, *arg_infos, **config_kwargs):
        config_kwargs['formatter_class'] = _SingleMetavarFormatter
        parser = argparse.ArgumentParser(**config_kwargs)
        required_args = parser.add_argument_group('required named arguments')
        
        for info in arg_infos:
            kwargs = {}
            if not isinstance(info, list) or len(info) not in [3, 4]:
                raise ArgumentError("""
                    Each arg must have 3 specifiers in a list:
                    [name or (names,), default or (default, type), help_string,
                    <optional dict of extra **kwargs for parser.add_argument()]
                    """)
            # item 1, either a string or a tuple of optional strings
            # '-' and '--' will be automatically prefixed
            name = info[0]
            if isinstance(name, str):
                name = [name]
            else:
                name = list(name)
            for i in range(len(name)):
                if not name[i].startswith('-'):
                    # only one char, that's a shorthand
                    if len(name[i]) == 1:
                        prefix = '-'
                    else:
                        prefix = '--'
                    # support both hyphen and underscore
                    name_added = None
                    if '-' in name[i]:
                        name_added = name[i].replace('-', '_')
                    elif '_' in name[i]:
                        name_added = name[i].replace('_', '-')

                    name[i] = prefix + name[i]
                    if name_added:
                        name.append(prefix + name_added)

            # item 2, default value. 
            # if None, set required to True. 
            # if a boolean, use action='store_true'
            values = info[1]
            dtype = None
            if isinstance(values, tuple):
                assert len(values) == 2
                values, dtype = values # let this fall through the `if` below
            if isinstance(values, list):
                # list of choices
                assert len(values) >= 2, 'must have at least 2 choices'
                dtype = type(values[0]) if not dtype else dtype
                default = values[0]
                kwargs['choices'] = values
            elif isinstance(values, ArgParser.Narg):
                default = values.default
                if not dtype:
                    dtype = values.dtype
                kwargs['nargs'] = values.nargs
            else:
                default = values
                if default is None:
                    dtype = str
                else:
                    dtype = type(default)

            # metavar: display --foo <float=0.05> in help string
            if 'choices' in kwargs:
                choices = kwargs['choices']
                kwargs['metavar'] = '<{}: {}>'.format(dtype.__name__, 
                               '/'.join(['{}']*len(choices)).format(*choices))
            elif 'nargs' in kwargs:
                # better formatting handled in _SingleMetavarFormatter
                kwargs['metavar'] = '{}'.format(dtype.__name__)
            else:
                kwargs['metavar'] = '<{}{}>'.format(dtype.__name__, 
                                      '={}'.format(default) if default else '')

            kwargs['default'] = default
            kwargs['type'] = dtype
            kwargs['required'] = default is None
            if isinstance(default, bool):
                kwargs['action'] = 'store_true' # switch on the boolean flag
                # store_true cannot co-exist with type or metavar
                del kwargs['type'], kwargs['metavar']
            else:
                kwargs['action'] = 'store' # normal behavior
            # item 3, help string
            kwargs['help'] = info[2]
            
            # last optional info is a dict that will update kwargs
            if len(info) == 4:
                assert isinstance(info[3], dict)
                kwargs.update(info[3])
            
            if kwargs['required']:
                required_args.add_argument(*name, **kwargs)
            else:
                parser.add_argument(*name, **kwargs)
        # add verbosity count
        parser.add_argument('--verbose', '-v', action='count', default=0,
                            help='Can repeat, e.g. -vvv for level 3 verbosity')
        parser.add_argument('--debug', action='store_true', default=False,
                            help='Turn on debugging mode. ')
        self.parser = parser
    
        
    def add(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

        
    def parse(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)
    
    
# =================== Meta programming and inflection ===================
"Test if a variable is a class type"
is_class = inspect.isclass
is_function = inspect.isfunction


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return (isinstance(obj, collections.Sequence)
            and not isinstance(obj, str))


def is_list_len(obj, length):
    """
    For argument sanity check: assert if the arg is a list/tuple of `len`.
    """
    return isinstance(obj, (tuple, list)) and len(obj) == length


def is_iterable(obj):
    """
    Returns: True if obj is iterable. 
    """
    return hasattr(obj, '__iter__')


def is_str(obj):
    return isinstance(obj, str)


def hasattrs(obj, attrs):
    """
    Multi-attr version of the built-in `hasattr()`
    
    Returns:
      If object has all the attrs specified in the second arg.
    """
    assert is_sequence(attrs)
    return all(hasattr(obj, attr) for attr in attrs)


def transfer_attrs(from_obj, to_obj, 
                  include_filter=None,
                  exclude_filter=None):
    """
    from_obj and to_obj can be dict. 
    If not dict, will ONLY transfer attrs in `vars()`, i.e. __dict__ 
    Transfer an attribute if:
    1. include_filter is None
    2. include_filter is a list and contains the attr
    3. include_filter is a lambda and returns True on the attr
    AND then exclude an attribute if:
    1. exclude_filter is not None
    2. exclude_filter is a list and contains the attr
    3. exclude_filter is a lambda and returns True on the attr
    Note that any attr that startswith '_' is automatically ignored
    """
    if isinstance(from_obj, dict):
        all_attrs = from_obj.keys()
        _get_attr = dict.__getitem__
    else:
        all_attrs = vars(from_obj).keys()
        _get_attr = getattr # use built-in
    
    if isinstance(to_obj, dict):
        _set_attr = dict.__setitem__
    else:
        _set_attr = setattr # use built-in

    all_attrs = filter(lambda attr: not attr.startswith('_'), all_attrs)
    for attr in include_exclude(all_attrs,
                                include_filter=include_filter,
                                exclude_filter=exclude_filter):
        try:
            _set_attr(to_obj, attr, _get_attr(from_obj, attr))
        except AttributeError as e:
            print('Attr', attr, e, file=sys.stderr)
            raise


# No need for this in python 3
def enum(*sequential, **named):
    """
    Numbers = enum('ZERO', 'ONE', 'TWO')
    Numbers.to_str(Numbers.ZERO) -> 'ZERO'
    Numbers.to_enum('ZERO') -> Numbers.ZERO
    Numbers.size == 3
    for n in Numbers():
       # 1, 2, 3
    """
    enum_dict = dict(zip(sequential, range(len(sequential))), **named)
    rev_dict = reverse_dict(enum_dict)
    enum_class = type('Enum', (), enum_dict)
    enum_class.to_str = staticmethod(lambda value: rev_dict[value])
    enum_class.to_enum = staticmethod(lambda key: enum_dict[key])
    enum_class.size = len(enum_dict)
    enum_class.__iter__ = lambda self: (i for i in range(len(enum_dict)))
    return enum_class


class AttributeDict(dict):
    """
    Set AttributeDict.OVERRIDE = True to allow overriding original dict methods.
    - True: `a.keys = 42; print a.keys` returns 42
    - False: `a.keys = 42; print a.keys` returns <function keys>
    
    vars() will return the (hacked) internal dictionary
    Always use dict() to convert if you want to manipulate it as a regular dict
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __OVERRIDE = True
    
    def __getattribute__(self, name):
        # hack to enable vars()
        if name == '__dict__':
            return self
        # if method overidden, return the overidden value 
        elif AttributeDict.__OVERRIDE and name in self:
            return self[name]
        else:
            return object.__getattribute__(self, name)
    
    @staticmethod
    def set_override(o):
        AttributeDict.__OVERRIDE = o


class PrefixProperty(object):
    """
    Decorator for class with prefixed properties
    internally, the class will have self.<prefix>varname
    externally, a read-only property will be created for self.varname
    takes care of '__' name mangling
    """
    def __init__(self, attr_names, prefix='_'):
        self.attr_names = attr_names
        # take care of name mangling
        if prefix == '__':
            self.prefix = (lambda cls_self : 
                           '_{}__'.format(cls_self.__class__.__name__))
        else:
            self.prefix = lambda cls_self: prefix

    def __call__(self, cls):
        assert is_class(cls)
        for attr in self.attr_names:
            setattr(cls, attr, 
                    # `attr=attr` hack to avoid closure, 
                    # otherwise 'attr' value will be changed as the loop goes
                    property(lambda cls_self, attr=attr: 
                             getattr(cls_self, self.prefix(cls_self) + attr)))
        return cls


def inspect_classes(module, include_import=True):
    """
    Args:
      module: either one of the following
      - string name of the module.
      - module variable itself.
      include_import: if False, only list classes being defined in this module 
          and exclude any imported class. 
      
    Returns: 
      list of classes (excluding or including imported ones)
    """
    if is_str(module):
        try:
            module = sys.modules[module]
        except KeyError:
            raise ArgumentError('module `{}` does not exist.'.format(module))
    cls_info = inspect.getmembers(module, inspect.isclass)
    if include_import:
        return [info[1] for info in cls_info]
    
    module = module.__name__
    ans = []
    for _, cls in cls_info:
        if (module == cls.__module__
            # module is part of the long path "xxx.xxx.module.class"
            or ('.'+module) in cls.__module__):
            ans.append(cls)
    return ans


def deepcopy_cls(cls, new_name=None):
    """
    Deep copy a class
    Args:
      cls: 
      new_name: if None, defaults to cls.__name__ + 'Copy'
    """
    assert is_class(cls)
    return type(new_name if new_name else cls.__name__ + 'Copy', 
                cls.__bases__, 
                dict(cls.__dict__))


def partial_kwarg(func, **apply_kwargs):
    """
    If a function has the specified kwargs, apply to it. 
    Otherwise ignore the kwarg.
    If the function itself accepts variable keyword args (**keywords), always 
    apply all of **apply_kwargs to func. 
    
    Returns:
      partially applied function
    """
    spec = inspect.getfullargspec(func)
    if spec.varargs in apply_kwargs:
        raise ArgumentError('*varargs `{}` cannot appear in **apply_kwargs'
                            .format(spec.varargs))
    if spec.varkw is None:
        apply_kwargs = {k: apply_kwargs[k] 
                        for k in apply_kwargs if k in spec.args}
    return functools.partial(func, **apply_kwargs)


# ==================== Exceptions handling ====================
class ArgumentError(Exception):
    """
    Raise if an argument to a function is invalid.
    """
    pass


@contextmanager
def errorsafe(safe_handler, exclude):
    '''
    Safe-handle all errors except for those explicitly excluded
    Args:
      safe_handler: function that takes e and does safe handling
        if None, do nothing
      excluded: dict mapping ExceptionType to function that handles e of that type
    '''
    excluded_excs = tuple(exclude.keys())
    try:
        yield
    except excluded_excs as e:
        exclude[type(e)](e)
    except Exception as e:
        if safe_handler:
            safe_handler(e)
    

def errorsafe_mode(safe_handle_mode='traceback', 
                   exclude_excs=None, 
                   exclude_handle_mode='exit'):
    '''
    Safe-handle all errors except for those explicitly excluded
    
    Args:
      safe_handler_mode: 
      - 'traceback': print raised exception's full traceback
      - 'print:{format-str-with-err}': print formatted error message
            e.g. 'print:oh no {err}!'
      - 'pass': do nothing
      excluded_excs: list of excluded ExceptionType(s)
      exclude_handle_mode: 
      - 'exit': sys.exit(1) terminate the program
      - 'print:{format-str-with-err}': print formatted error message and exit
            e.g. 'print+exit:oh no {err}!'
      - 'raise': raise the exception as-is
    
    Returns:
      context manager.
    '''
    def _errmsg_format(mode):
        errmsg = mode[len('print:'):]
        if errmsg:
            return errmsg
        else: # empty string
            return '{err}'
    
    mode = safe_handle_mode
    if mode == 'traceback':
        def safe_handler(e):
            traceback.print_exc()
    elif mode.startswith('print'):
        errmsg_safe = _errmsg_format(mode)
        def safe_handler(e):
            print(errmsg_safe.format(err=str(e)))
    elif mode == 'pass':
        def safe_handler(e): pass
    else:
        raise NotImplementedError('Safe handler mode {} is not implemented'
                                  .format(mode))
    
    if exclude_excs is None:
        exclude_excs = []
    
    mode = exclude_handle_mode
    if mode == 'exit':
        def exc_handler(e):
            sys.exit(1)
    elif mode.startswith('print'):
        # MUST have a different name than above errmsg_safe, otherwise closure bug
        errmsg_exc = _errmsg_format(mode)
        def exc_handler(e):
            print(errmsg_exc.format(err=str(e)))
            sys.exit(1)
    elif mode == 'raise':
        def exc_handler(e):
            raise e
    else:
        raise NotImplementedError('Excluded handler mode {} is not implemented'
                                  .format(mode))
    
    return errorsafe(safe_handler=safe_handler, 
                     exclude={exc : exc_handler for exc in exclude_excs})


# =================== Simple Email ===================
import smtplib
import email.mime.text
import email.utils

class Emailer(object):
    
    def __init__(self, sender_name, 
                 sender_email, 
                 recv_email, 
                 passwd,
                 smtp_addr):
        self.sender_name = sender_name
        self.sender_email = sender_email
        self.recv_email = recv_email
        self.passwd = passwd
        self.smtp_addr = smtp_addr
        
        # under debug mode, doesn't send any email
        self._debug = False
    
    
    def set_debug(self, debug):
        self._debug = debug
    

    def send_multiple(self, *messages, dryrun=False):
        '''
        Send multiple emails.
        Args:
          messages: Each msg is a tuple (subject, bodytext)
          dry_run=True to print the msg out without actually sending it. 
        '''
        try:
            server = smtplib.SMTP(self.smtp_addr, 587)
            server.ehlo()
            server.starttls()
            server.login(self.sender_email, self.passwd)

            for msgtuple in messages:
                if type(msgtuple) != tuple or len(msgtuple) != 2:
                    raise ArgumentError(
                        "Each message must be a tuple (subject, bodytext)")
                
                subject, textbody = msgtuple
                msg = email.mime.text.MIMEText(textbody)
                
                msg['Subject'] = subject
                msg['From'] = email.utils.formataddr(
                                        (self.sender_name, self.sender_email))
                msg['To'] = self.recv_email
                msg = msg.as_string()
                
                if dryrun:
                    print('======== [Emailer dry run msg] ========')
                    print(msg)
                    print('======== [end msg] ========\n')
                else:
                    server.sendmail(self.sender_email, [self.recv_email], msg)

            server.quit()

        except Exception as e:
            print('[Emailer error]:', str(e))


    def send(self, subject, textbody):
        self.send_multiple((subject, textbody))
        
        
# ==================== Unittest ====================
def SET_DOC(cls):
    """
    Call SET_DOC(cls) in `def setUpClass`.
    """
    def shortDescription(self):
        doc = self._testMethodDoc
        if not doc: return doc
        doc = doc.strip().replace('\n', '\n\t')
        return '>>\t' + doc
    # override unittest.TestCase.shortDescription
    cls.shortDescription = shortDescription


def SET_ASSERT(self, gs):
    """
    Inside a test method, call 
    SET_ASSERT(self, globals())
    """
    gs['EQ'] = self.assertEqual
    gs['NEQ'] = self.assertNotEqual
    gs['FEQ'] = self.assertAlmostEqual
    gs['LT'] = self.assertLess
    gs['LEQ'] = self.assertLessEqual
    gs['GT'] = self.assertGreater
    gs['GEQ'] = self.assertGreaterEqual
    gs['ITEM_EQ'] = self.assertItemsEqual
    
    gs['TRUE'] = self.assertTrue
    gs['FALSE'] = self.assertFalse
    gs['NONE'] = self.assertIsNone
    gs['NOT_NONE'] = self.assertIsNotNone
    gs['IN'] = self.assertIn
    gs['NOT_IN'] = self.assertNotIn
    gs['TYPE'] = self.assertIsInstance
    gs['NOT_TYPE'] = self.assertNotIsInstance
    gs['RAISE'] = self.assertRaises