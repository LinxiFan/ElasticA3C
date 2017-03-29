"""
Decorators, higher-order functionals, functools module extensions.
"""
import sys
import functools
import dis
import inspect
from .common import is_sequence, is_class, hasattrs, deepcopy_cls, pack_args


def meta_wrap(decor):
    """
    a decorator decorator, allowing the wrapped decorator to be used as:
    @decorator(*args, **kwargs)
    def callable()
      -- or --
    @decorator  # without parenthesis, args and kwargs will use default
    def callable()
    
    Args:
      decor: a decorator whose first argument is a callable (function or class
        to be decorated), and the rest of the arguments can be omitted as default.
        decor(f, ... the other arguments must have default values)

    Warning:
      decor can NOT be a function that receives a single, callable argument. 
      See stackoverflow: http://goo.gl/UEYbDB
    """
    single_callable = (lambda args, kwargs: 
                       len(args) == 1 and len(kwargs) == 0 and callable(args[0]))
    @functools.wraps(decor)
    def new_decor(*args, **kwargs):
        if single_callable(args, kwargs):
            # this is the double-decorated f. 
            # It should not run on a single callable.
            return decor(args[0])
        else:
            # decorator arguments
            return lambda real_f: decor(real_f, *args, **kwargs)

    return new_decor


@meta_wrap
def noop_decorator(func, *args, **kwargs):
    """
    For debugging purposes. 
    A decorator that does nothing. Compatible with `meta_wrap` style.  
    """
    return func


@meta_wrap
def deprecated(func, msg='', action='warning'):
    """
    Function/class decorator: designate deprecation.
    
    Args:
      msg: string message. 
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing
      - 'raise': raise DeprecatedError(`msg`)
    """
    action = action.lower()
    if action not in ['warning', 'noop', 'raise']:
        raise ValueError('unknown action type {}'.format(action))
    if not msg:
        msg = 'This is a deprecated feature.'

    # only does the deprecation when being called
    @functools.wraps(func)
    def _deprecated(*args, **kwargs):
        if action == 'warning':
            print(msg, file=sys.stderr)
        elif action == 'raise':
            raise DeprecationWarning(msg)
        return func(*args, **kwargs)
    return _deprecated


@meta_wrap
def experimental(func, msg='', action='warning'):
    """
    Function/class decorator: warn user of experimental feature.
    
    Args:
      msg: string message
      action: string mode
      - 'warning': (default) prints `msg` to stderr
      - 'noop': do nothing
    """
    action = action.lower()
    if action not in ['warning', 'noop']:
        raise ValueError('unknown action type {}'.format(action))
    msg = 'experimental feature: ' + msg

    # only issues warning when being called
    @functools.wraps(func)
    def _experimental(*args, **kwargs):
        if action == 'warning':
            print(msg, file=sys.stderr)
        return func(*args, **kwargs)
    return _experimental


def next_iterable(cls):
    """
    Decorator to make a class iterable with `next()` given `__iter__()`
    Compatible with both python 2 and 3.
    
    Args:
      cls: must define __iter__() (returns a generator or has `yield` statement)
    """
    assert is_class(cls)
    assert hasattr(cls, '__iter__'), '__iter__ needs to return a generator'
    assert not hasattrs(cls, '_next_iterator')
    old_init = cls.__init__
    @functools.wraps(old_init)
    def _init(self, *args, **kwargs):
        self._next_iterator = self.__iter__()
        old_init(self, *args, **kwargs)
    # override the subclass' constructor
    cls.__init__ = _init
    
    def _next(self):
        return next(self._next_iterator)
    
    cls.next = cls.__next__ = _next
    return cls


@meta_wrap
def circular_iterable(cls, cycles=0):
    """
    Decorator to make a class cyclic iterable with for-loop.
    Compatible with both python 2 and 3. 
    Meta-wrap style.
    
    Args:
      cls: must define __iter__() (returns a generator or has `yield` statement)
      cycles: 
      - 0: infinitely iterable
      - N: repeat a specific number of times
    """
    assert is_class(cls)
    assert hasattr(cls, '__iter__'), '__iter__ needs to return a generator'
    assert not hasattrs(cls, '_circular_iterator')
    
    # must make a deep copy of the class so that the old class is not modified
    cls = deepcopy_cls(cls)
    old_iter = cls.__iter__
    @functools.wraps(old_iter)
    def _new_iter(self):
        it = old_iter(self)
        is_inf = cycles == 0 # infinitely iterable
        c = cycles
        while is_inf or c > 0:
            try:
                yield next(it)
            except StopIteration:
                it = old_iter(self)
                c -= 1
    cls.__iter__ = _new_iter
    return cls


# =================== function tools ===================
def compose(*functions):
    """
    Compose multiple functions
    compose(f, g, h)(x) <=> f(g(h(x)))
    """
    return functools.reduce(lambda f, g: 
                              lambda *args, **kwargs: f(g(*args, **kwargs)), 
                            functions)


def chain_and(*predicates):
    """
    Args:
      *predicates: a list of lambdas that take one arg and return bool
      
    Returns:
      a lambda predicate that is true iff all predicates are true.
      employs short-circuit logic. Internally, use a generator instead of 
      list comprehension to enable short-circuit.
    """
    predicates = pack_args(predicates)
    return lambda obj: all((pred(obj) for pred in predicates))


def chain_or(*predicates):
    """
    Args:
      *predicates: a list of lambdas that take one arg and return bool

    Returns:
      a lambda predicate that is true if at least one of the predicates are true
    """
    predicates = pack_args(predicates)
    return lambda obj: any((pred(obj) for pred in predicates))


def apply_map(func, seq):
    assert is_sequence(seq)
    for i in range(len(seq)):
        seq[i] = func(seq[i])


@meta_wrap
def lru_cache(func, maxsize=128, typed=False):
    """
    Can be used with or without parenthesis. See `meta_wrap`'s effect. 
    """
    return functools.lru_cache(maxsize, typed)(func)


# ================ @overrides decorator ================
def overrides(obj):
    """
    Class member method decorator. Mimics the Java `@Override` syntax.
    Used primarily for error checking in inheritance hierarchy. 
    The decorator code is executed while loading class. Using this method
    should have minimal runtime performance implications.
    
    Returns:
      method itself, with possibly added docstring from superclass. 
    
    There are two usage syntax: 
    1. @overrides  # by itself, no parentheses.
       check if the method overrides any of its base classes. 
    2. @overrides(BaseClass)  
       check if the method overrides a specific base class.
    
    class A:
        def f1(): pass
    
    class B(A):
        @overrides
        def f1(): pass
        
    class C(B):
        @overrides(A)
        def f1(): pass
    
    class D:
        @overrides(C)  # raises exception at class definition time. 
        def f1(): pass
    """
    if inspect.isfunction(obj):
        return _overrides_helper(obj, target_class=None)
    elif inspect.isclass(obj):
        return functools.partial(_overrides_helper, target_class=obj)
    else:
        raise ValueError('@overrides invalid type: {}'.format(obj))


def _overrides_helper(method, target_class):
    """
    Helper for @overrides decorator
    source: http://stackoverflow.com/a/8313042/308189

    Args:
      method: must be an instance method.
      target_class: if None, look into call stack and infer the base class(es),
          otherwise check if target_class is properly overridden
    
    Returns:
      method itself, with possibly added docstring from superclass. 
      
    Raises:
      AssertionError if this method does not properly override any base class. 
    """
    # getframe(N): N should change if you have more function redirects
    if target_class:
        frame = sys._getframe(2)
    else:
        # if you use @overrides without any arg
        frame = sys._getframe(3)
    base_classes = get_base_classes(frame, method.__globals__)
    m_name = method.__name__
    
    assert base_classes, 'This class does not have a base class.'
    if target_class is not None:
        assert inspect.isclass(target_class), \
            '`{}` is not of class type.'.format(target_class)
        all_bases = set()
        for base in base_classes:
            all_bases |= set(inspect.getmro(base))
        assert target_class in all_bases, \
            '`{}` is not a base class.'.format(target_class)
        assert hasattr(target_class, m_name), \
            'Base `{}` does not have `{}()`'.format(target_class, m_name)
        if not method.__doc__:
            method.__doc__ = getattr(target_class, m_name).__doc__
        return method
    else:
        # examine all inferred base classes
        for base in base_classes:
            if hasattr(base, m_name):
                if not method.__doc__:
                    method.__doc__ = getattr(base, m_name).__doc__
                return method
        raise AssertionError('No base class method found for `{}()`'
                             .format(m_name))

def get_base_classes(frame, namespace):
    """
    Inside a new class definition, the class itself has not been defined yet.
    We need to hack python code object to obtain its base classes.
    
    Args:
      frame: sys._getframe(N) to obtain the most outer frame
      namespace: XX.__globals__
    
    Returns:
      A list of base classes for the current enclosing class.
    """
    cls = []
    # components: [module1, module2, ..., class]
    for components in _get_base_class_names(frame):
        cl = namespace[components[0]]
        for component in components[1:]:
            cl = getattr(cl, component)
        cls.append(cl)
    return cls
        

def _get_base_class_names(frame):
    """ Get baseclass names from the code object """
#     if is_python3():
    itemint = lambda x : x
#     else:
#         itemint = lambda x : ord(x)

    co, lasti = frame.f_code, frame.f_lasti
    code = co.co_code
    i = 0
    extended_arg = 0
    extends = []
    while i <= lasti:
        c = code[i]
        op = itemint(c)
        i += 1
        if op >= dis.HAVE_ARGUMENT:
            oparg = itemint(code[i]) + itemint(code[i+1])*256 + extended_arg
            extended_arg = 0
            i += 2
            if op == dis.EXTENDED_ARG:
                extended_arg = oparg * 65536
            if op in dis.hasconst:
                if type(co.co_consts[oparg]) == str:
                    extends = []
            elif op in dis.hasname:
                if dis.opname[op] == 'LOAD_NAME':
                    extends.append(('name', co.co_names[oparg]))
                if dis.opname[op] == 'LOAD_ATTR':
                    extends.append(('attr', co.co_names[oparg]))
    items = []
    previous_item = []
    for t, s in extends:
        if t == 'name':
            if previous_item:
                items.append(previous_item)
            previous_item = [s]
        else:
            previous_item += [s]
    if previous_item:
        items.append(previous_item)
    return items

