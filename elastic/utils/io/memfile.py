"""
Memory-optimized file types
"""
import struct
from enum import Enum
from .filesys import f_exists, f_expand, f_size
from ..common import is_sequence
from ..functional import overrides

"""
See `file.seek` in standard Python library
- 0: seek relative to the beginning of the file.
- 1: relative to the current read head.
- 2: relative to the end of the file.
"""
class SEEK(Enum):
    Begin = 0
    Current = 1
    End = 22

STR_EOF = ''


class AbstractFile(object):
    """
    Abstract interface for file.
    """
    def __init__(self, file_path, mode):
        """
        Constructs and *opens* the file handle.
        self.handle holds the internal file handle object.
        
        Args:
          file_path
          mode: regular file mode str like'w', 'r', 'a', etc.
        """
        self.file_path = f_expand(file_path)
        self.handle = None
        self.mode = mode
        if self.is_read() and not f_exists(self.file_path):
            raise IOError(self.file_path + ' does not exist.')

        self.open()
    
    
    def open(self, mode=None):
        """
        Args:
          mode: if specified, override the mode in __init__
        """
        self.check_handle(False)
        self.handle = open(self.file_path, 
                           mode if mode else self.mode)
        return self

    
    def check_handle(self, should_open):
        """
        Check if the file handle is `should_open` (bool).
        """
        if should_open and self.handle is None:
            raise IOError('BinaryFile: file handle not open.')
        if not should_open and self.handle is not None:
            raise IOError('BinaryFile: old file handle must be closed first.')
    
    
    def write(self, obj):
        """
        Writes obj to file.
        """
        raise NotImplementedError('write(obj)')


    def write_n(self, obj_list):
        """
        Writes a list of objs to file.
        Calls self.write() multiple times.
        """
        assert is_sequence(obj_list)
        for obj in obj_list:
            self.write(obj)

    
    def _get_position(self, position, seek_mode):
        """
        if seek_mode is None: 
            - position >= 0: seek_mode set to SEEK.Begin
            - position < 0: seek_mode set to SEEK.End
        """
        if seek_mode is None:
            if position >= 0:
                seek_mode = SEEK.Begin
            else:
                seek_mode = SEEK.End
        else:
            if seek_mode == SEEK.Begin:
                if position < 0:
                    raise IndexError('Out of range: seek before beginning: '
                                     '{}'.format(position))
            elif seek_mode == SEEK.End:
                if position > 0:
                    raise IndexError('Out of range: seek after EOF: '
                                     '{}'.format(position))
            # assert seek_mode in SEEK, \
            #     '{} is not a valid SEEK mode.'.format(seek_mode)
        return position, seek_mode
    
    
    def seek(self, position, seek_mode=None):
        """
        Seek to a position in the file. 
        
        Args:
          position: the actual byte location = position * self.size
          seek_mode: see python's built-in `file.seek`. Use SEEK enum.
          - 0: from the beginning of file
          - 1: from current position
          - 2: from the end of file
          - None: infer. if position < 0, set seek to SEEK.End
        """
        self.check_handle(True)
        self.handle.seek(*self._get_position(position, seek_mode))
    
    
    def tell(self):
        raise NotImplementedError('return file position')
    
    
    def read(self, *args, **kwargs):
        """
        Advances the file pointer forward.
        
        Returns:
        - object one at a time
        - self.EOF symbol
        """
        raise NotImplementedError('return obj')
    
    
    def read_n(self, n, *args, **kwargs):
        """
        Repeat `read(*args, **kwargs)` n times. Advances the file pointer forward by n.

        Returns:
          a list of size n. If EOF is encountered midway, truncate the list.
        """
        reads = []
        for _ in range(n):
            obj = self.read(*args, **kwargs)
            if obj == self.EOF:
                return reads
            else:
                reads.append(obj)
        return reads
    
    
    def read_all(self, *args, **kwargs):
        """
        Read all the way till EOF. 
        Calls self.iter()
        """
        return [obj for obj in self.iter(*args, **kwargs)]
    
    
    def iter(self, *args, **kwargs):
        """
        Returns: 
          a generator to read through the data from the current file pos.
        """
        assert self.is_read()
        self.check_handle(True)
        while True:
            dat = self.read(*args)
            # warning: don't use `if dat`: will return false if dat reads `0`
            if dat == self.EOF:
                break
            else: 
                yield dat
    
    
    def __iter__(self):
        """
        Enables for-loop iterator syntax. 
        """
        return self.iter()
    
    
    def iter_from(self, position, seek_mode=None):
        """
        Combines seek() and iter()

        Returns: 
          a generator to read through the data from specified position.
        """
        self.seek(position, seek_mode)
        return self.iter()
    
    
    def peek(self, position=None, seek_mode=None):
        """
        Does not change the file head.
        
        Args:
          position: if None, peek at the current read head.
          seek_mode: see `seek()`

        Returns: 
          entry at that position.
        """
        pos = self.tell()
        if position is not None:
            self.seek(position, seek_mode)
        obj = self.read()
        self.seek(pos)
        return obj
    
    
    def __getitem__(self, position):
        """
        Functionally equivalent to peek(), does not change read head.
        
        Args:
          position: if < 0, seek_mode=SEEK.End, peek from end of file.
        """
        return self.peek(position, None)
        
    
    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        
    
    def is_read(self):
        "If mode is `r` or `rb`"
        return self.mode.startswith('r')
    
    
    @property
    def EOF(self):
        """
        Override this method to have different EOF symbol.
        EOF should be returned from `read()`.
        """
        return STR_EOF


    # enable context manager
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    
    def __len__(self):
        """
        Returns:
          number of entries in the file. 
        """
        raise NotImplementedError('return number of entries in the file')


class TextFile(AbstractFile):
    """
    Read text file line by line.
    """
    @overrides
    def read(self):
        return self.handle.readline()
    
    @overrides
    def read_all(self):
        return self.handle.readlines()
    
    @overrides
    def write(self, obj):
        self.handle.write(obj)
    
    @overrides
    def tell(self):
        raise NotImplementedError('TextFile does not support tell(). '
                                  'Use LineAccessFile instead. ')

    @overrides
    def seek(self, position, seek_mode=None):
        raise NotImplementedError('TextFile does not support seek()')


class BinaryFile(AbstractFile):
    """
    Read and write binary values as if regular file.
    """
    def __init__(self, file_path, mode, data_fmt, convert_type=None):
        """
        Args:
          file_path
          mode: 'w', 'r', or 'a'. Automatically append 'b' to the mode.
          data_fmt: see https://docs.python.org/2/library/struct.html
            i - int; I - unsigned int; q - long long; Q - unsigned long long
            f - float; d - double; s - string; c - char; ? - bool
            b - signed char; B - unsigned char; h - short; H - unsigned short.
            3i - tuple of 3 ints; 'ifb' - tuple of int, float, bool
          convert_type: if you write int, file.read() will return a tuple (3,)
              use convert_type to return convert_type(*read) instead.
        """
        self.data_fmt = data_fmt
        self._size = struct.calcsize(data_fmt)
        self.convert_type = convert_type
        mode = self._get_mode(mode)

        AbstractFile.__init__(self, file_path, mode)
    
    
    @overrides
    def open(self, mode=None):
        """
        Args:
          mode: if specified, override the mode in __init__
        """
        if mode:
            mode = self._get_mode(mode)
        return AbstractFile.open(self, mode)


    def _get_mode(self, mode):
        return mode if mode.endswith('b') else mode+'b'
        
    
    @overrides
    def write(self, obj):
        self.check_handle(True)
        # must unpack the tuple before passing to struct.pack()
        if not is_sequence(obj):
            obj = (obj,)
        self.handle.write(struct.pack(self.data_fmt, *obj))

    
    @overrides
    def read(self, convert_type=None):
        """
        Args:
          convert_type: if specified, override the convert_type in __init__
        
        Returns:
        - if convert_type, return convert_type(*read)
        - empty string '' when EOF reached
        - tuple from struct.unpack()
        """
        self.check_handle(True)
        convert_type = convert_type or self.convert_type
        dat = self.handle.read(self._size)
        if dat != STR_EOF:
            dat = struct.unpack(self.data_fmt, dat)
            if convert_type:
                return convert_type(*dat)
        return dat 
    
    
    @overrides
    def seek(self, position, seek_mode=None):
        """
        Seek to a position in the file. 
        
        Args:
          position: the actual byte location = position * self.size
          seek_mode: see python's built-in `file.seek`
          - 0: from the beginning of file
          - 1: from current position
          - 2: from the end of file
        """
        AbstractFile.seek(self, position * self.size, seek_mode)
    
    
    @overrides
    def tell(self):
        self.check_handle(True)
        fpos = self.handle.tell()
        assert fpos % self.size == 0
        return fpos // self.size
    

    @property
    def size(self):
        """
        Returns:
          size of one binary data. == struct.calcsize()
        """
        return self._size
    
    @overrides
    def iter(self, convert_type=None):
        """
        Args:
          convert_type: see `self.read()`
        Returns: 
          a generator to read through the data.
        """
        return AbstractFile.iter(self, convert_type)
    
    
    @overrides
    def __len__(self):
        """
        Uses os.path.getsize() to infer the length. 
        If the file doesn't have the right format, the result will be wrong.
        Time complexity: O(1)
        
        Returns:
          number of binary entries in the file.
        """
        fsize = f_size(self.file_path)
        assert fsize % self.size == 0, 'file size is not a multiple of data size.'
        return  fsize // self.size


class InMemoryFile(AbstractFile):
    """
    Read once as an array.
    Helper class for LineAccessFile.
    """
    TEXT = 'TEXT'
    
    def __init__(self, file_path, data_fmt, 
                 block_size=0, convert_type=None):
        """
        Args:
          file_path
          data_fmt:
          - a string of binary format
          - InMemoryFile.TEXT, read file as text delimited by lines.
          block_size: 
          - if 0, read as a plain list [obj1, obj2, ...]
          - if > 0, read as a list of lists, [[block 1], [block 2] ...]
        """
        self.data_fmt = data_fmt
        self.convert_type = convert_type
        self.block_size = int(block_size)
        AbstractFile.__init__(self, file_path, 'r')
    
    
    @overrides
    def open(self):
        self.check_handle(False)
        
        if self.data_fmt == InMemoryFile.TEXT:
            self.handle = TextFile(self.file_path, self.mode)
        # deal with binary file
        else: 
            self.handle = BinaryFile(self.file_path, 
                                     mode=self.mode,
                                     data_fmt=self.data_fmt,
                                     convert_type=self.convert_type) 
        if self.block_size > 0:
            self.memory = []
            while True:
                block = self.handle.read_n(self.block_size)
                if block:
                    self.memory.append(block)
                else:
                    break
        else: # flattened list of objects
            self.memory = self.handle.read_all()
                
        self.pos = 0 # file position pointer
    
    
    @overrides
    def seek(self, position, seek_mode=None):
        position, seek_mode = self._get_position(position, seek_mode)
        if seek_mode == SEEK.Begin:
            self.pos = position
        elif seek_mode == SEEK.Current:
            self.pos += position
        elif seek_mode == SEEK.End:
            self.pos = len(self) + position
        assert self.pos >= 0, 'invalid seek position: {}'.format(self.pos)
        
    
    @overrides
    def tell(self):
        return self.pos
    
    
    @overrides
    def read(self):
        if self.pos >= len(self):
            return self.EOF
        else:
            obj = self.memory[self.pos]
            self.pos += 1 # advance the "file" pointer
            return obj
    

    def clear_memory(self):
        self.pos = 0
        self.memory = []

    
    @property
    @overrides
    def EOF(self):
        if self.block_size == 0:
            return STR_EOF
        # in block mode, self.memory is a list of list.
        else:
            return []
        
        
    @overrides
    def __len__(self):
        return len(self.memory)
    

"""
File load mode: 
- 0: hold the entire file in-memory.
- 1: use `seek` to access the file on-disk.
"""
class LoadMode(Enum):
    InMem = 0
    OnDisk = 1


class LineAccessFile(AbstractFile):
    """
    Use an auxiliary binary file to enable random access by line.
    """
    def __init__(self, file_path, 
                 block_size, 
                 aux_suffix='.line',
                 aux_mode=LoadMode.InMem,
                 aux_regenerate=False):
        """
        Args:
          file_path
          block_size: number of lines per block. Lines delimited by '\n'
          aux_suffix: auxiliary binary line file = `file` + aux_suffix
          aux_mode:
            - MEM (0): read entire auxiliary line file into memory.
            - DISK (1): use `seek` to read line info from the auxiliary file.
          aux_regenerate: if True, generate auxiliary file even if it already exists.
        """
        assert aux_mode in LoadMode, 'invalid LoadMode'

        AbstractFile.__init__(self, file_path, 'r')

        self.aux_path = f_expand(self.file_path + aux_suffix)
        self.block_size = block_size

        if aux_regenerate or not f_exists(self.aux_path):
            self.generate_line_file()
            assert f_exists(self.aux_path), 'line file generation failed.'
        
        if aux_mode is LoadMode.InMem:
            # read into an in-memory array (fake as a file handle)
            self.aux_handle = InMemoryFile(self.aux_path,
                                                 data_fmt='Q', 
                                                 convert_type=int)
        else:
            self.aux_handle = self._line_fhandle('r')
    
    
    def _line_fhandle(self, mode):
        return BinaryFile(self.aux_path, 
                          data_fmt='Q', 
                          mode=mode,
                          convert_type=int)
    
    
    def generate_line_file(self):
        """
        Generates the auxiliary line file that has position info for each block.
        Regenerate even if the file already exists.

        Returns:
          number of blocks processed.
        """
        l = 0
        with open(self.file_path, 'r') as f, \
            self._line_fhandle('w') as aux:
            while True:
                pos = f.tell()
                line = f.readline()
                if line == '':
                    break # EOF reached
                # write after `break` to avoid writing EOF position at last.
                if l % self.block_size == 0:
                    aux.write(pos)
                l += 1
    
    @overrides
    def seek(self, position, seek_mode=None):
        "Sets the current block position."
        self.aux_handle.seek(position, seek_mode)
    
    
    @overrides
    def tell(self):
        "Returns: the current block position"
        return self.aux_handle.tell()
    
    
    @overrides
    def read(self):
        """
        Returns:
          Reads a single block at the current read head. 
        """
        index = self.aux_handle.read()
        if index == self.aux_handle.EOF:
            return self.EOF
         
        self.handle.seek(index, SEEK.Begin)
        lines = []
        for _ in range(self.block_size):
            line = self.handle.readline() 
            if line == STR_EOF:
                break
            lines.append(line)
        return lines
        
    
    @property
    @overrides
    def EOF(self):
        return []
    
    
    @overrides
    def close(self):
        AbstractFile.close(self)
        self.aux_handle.close()
    
    
    @overrides
    def __len__(self):
        """
        See: BinaryFile.__len__
        Time complexity: O(1), relies on the auxiliary file
        
        Returns:
          number of blocks in the file. 
        """
        return len(self.aux_handle)
    
