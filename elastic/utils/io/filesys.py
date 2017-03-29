"""
File system utils.
"""
import os
import sys
import errno
import shutil
import glob
import pwd
import codecs
from socket import gethostname

f_exists = os.path.exists

f_ext = os.path.splitext

f_join = os.path.join

f_expand = os.path.expanduser

f_size = os.path.getsize

is_file = os.path.isfile

is_dir = os.path.isdir

def owner_name(filepath):
    """
    Returns: file owner name, unix only
    """
    return pwd.getpwuid(os.stat(filepath).st_uid).pw_name


def host_name():
    "Get host name, alias with ``socket.gethostname()``"
    return gethostname()


def host_id():
    """
    Returns: first part of hostname up to '.'
    """
    return host_name().split('.')[0]


def utf_open(fname, mode):
    """
    Wrapper for codecs.open
    """
    return codecs.open(fname, mode=mode, encoding='utf-8')


def is_txt(fpath):
    "Test if file path is a text file"
    _, ext = f_ext(fpath)
    return ext == '.txt'


def f_mkdir(fpath):
    "If exist, do nothing."
    try:
        os.mkdir(fpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
    

def f_time(fpath):
    "File modification time"
    return str(os.path.getctime(fpath))


def f_append_before_ext(fpath, suffix):
    """
    Append a suffix to file name and retain its extension
    """
    name, ext = f_ext(fpath)
    return name + suffix + ext


def f_add_ext(fpath, ext):
    """
    Append an extension if not already there
    Args:
      ext: will add a preceding `.` if doesn't exist
    """
    if not ext.startswith('.'):
        ext = '.' + ext
    if fpath.endswith(ext):
        return fpath
    else:
        return fpath + ext


def f_join_expand(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    return f_expand(f_join(*fpaths))


def f_remove(fpath):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    for f in glob.glob(fpath):
        try:
            shutil.rmtree(f)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                os.remove(f)


def f_copy(fsrc, fdst):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    for f in glob.glob(fsrc):
        try:
            shutil.copytree(f, fdst)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                shutil.copy(f, fdst)


def f_size(fpath):
    return os.path.getsize(fpath)


def script_dir():
    """
    Returns: the dir of current script
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def parent_dir(location):
    """
    Args:
      location: current directory or file
    
    Returns: parent directory absolute path
    """
    return os.path.abspath(os.path.join(location, os.pardir))