"""
Implement I/O functionalities, mostly for Data class and subclass

StorageEntry (abstact) and subclasses: 
    classes that make the interface to some storage entry
        implement __init__(url), open('r'|'w'), read, write and close()
        
    FileEntry: default storage-entry
    
...
map-storage:
    class that manageS a set of entries each related to an id (name)
    implement __init__(entry-class, entry-class param)
              
"""

import cPickle as cPickle
import os as _os
import urlparse as _urlparse

from rhizoscan.tool import _property
from rhizoscan.tool.path import assert_directory as _assert_directory

_PICKLE_PROTOCOL_ = -1 

class PickleSerializer(object):
    """
    basic pickle serializer
    
    It has static `dump` and `load` functions, using protocol=-1 and 
    extension='.pickle'
    """
    _PICKLE_PROTOCOL_ = -1 # pickle saving (dump) are done with the latest protocol
    
    extension = '.pickle'  # file extension that indicate the file content type
    
    @staticmethod
    def dump(obj,stream):
        return cPickle.dump(obj, stream, protocol=_PICKLE_PROTOCOL_)
    @staticmethod
    def load(stream):
        return cPickle.load(stream)
        

class _txt_stream(object):
    """ virtual writable text stream """
    def __init__(self):              
        self.txt = ''
    def write(self, txt):
        self.txt += txt
    def get_text(self):
        return self.txt
    def close(self):
        self.txt = ''

class RegisteredEntry(type):
    """
    Metaclass used to register StorageEntry w.r.t url scheme
    
    To register a class, it must have the following attributes:
    
        `__metaclass__ = RegisteredEntry`
        `__scheme__`   a tuple of scheme name. e.g. 'file', 'http'
        
        (Note `__metaclass__` is set by default to StorageEntry subclasses)
        
    To get the StorageEntry subclass registered for a scheme, use 
      `RegisteredEntry.get(scheme_name)`
      
    
    If the constructor does not have the required API, i.e. `__init__(url)`), 
    then it can have an `__entry_constructor__` attribute that point to a static
    method which implement it.
    
    
    Note: Only the last registered StorageEntry for a specific scheme is kept.
    
    
    See also: `create_entry(url)`
    """
    register = {}
    def __init__(cls, name, *args, **kargs):
        schemes = getattr(cls, '__scheme__',[])
        constructor = getattr(cls, '__entry_constructor__', cls)
        for scheme in schemes:
            RegisteredEntry.register[scheme] = constructor
            
    @staticmethod
    def get(scheme):
        if scheme not in RegisteredEntry.register:
             raise LookupError('no entry class has been registered for {} scheme'.format(scheme))
        return RegisteredEntry.register.get(scheme)

class StorageEntry(object):
    """
    Abstract class that define the API of a storage entry:
      - load: retrieve the data stored in the entry
      - save: store the data in the entry
      - exist: return True if the entry exist, or False otherwise
      
    StorageEntry has a `url` properties that store the url string of the entry
    
    See also `create_entry()`
    """
    __metaclass__ = RegisteredEntry
    
    def load(self,):      raise NotImplementedError('StorageEntry is an abstract class')
    def save(self,data):  raise NotImplementedError('StorageEntry is an abstract class')
    def exist(self):      raise NotImplementedError('StorageEntry is an abstract class')
        
    def __str__(self):
        return self.__class__.__name__ + '(' + self.url + ')'
    def __repr__(self):
        return self.__str__()
        
    @_property
    def url(self):
        return self.__url__
    @url.setter
    def url(self, url):
        self.__url__ = url
        
def create_entry(url):
    """
    Create the suitable Storage entry for `url`
    
    If `url`is a StorageEntry instance, return it
    If `url` does not contain a scheme (eg. file://'), file type is used.
    
    Note: currently, only `file:` scheme is recognized.
    """
    if isinstance(url, StorageEntry):
        entry = url
    else:
        scheme = _urlparse.urlsplit(url).scheme
        cls    = RegisteredEntry.get(scheme)
        entry  = cls(url)
    
    return entry

class FileEntry(StorageEntry):
    """
    One-file storage with implicit serialization
    
    A FileEntry object has the following methods:
    
      - open(mode):  open the file (file name is given to constructor)
      - read():      read and deserialize the content of the file
      - write(data): write (replace) serialized `data` in the file
      - close():     close the file
      
      - load(): same as open('r')-read-close
      - save(): same as open('w')-write-close
      
    :Serializer:
        By default, FileEntry use `PickleSerializer` (which use `cPickle`) to 
        serialize/deserialize data. However, `load` and `save` have suitable 
        parameter to provide alternate serializer.
        
        See also: `PickleSerializer`, `StorageEntry`
    """
    __scheme__ = ('file', '')
    
    def __init__(self, filename):
        """
        Create a FileEntry object related to `filename`
        """
        ## `read`, `write`, `save` and `load` to have serializer arg?
        
        if len(filename):
            self.url = _os.path.abspath(_urlparse.urlsplit(filename).path)
        else:
            self.url = filename
        self._stream = None
        
    def open(self,mode='r'):
        """
        mode='w' means write operation is done in a buffered, then saving is 
        done by the `close` method.
        """
        if self._stream is not None:
            raise IOError('File stream already open')
        if mode=='w':
            self._stream = _txt_stream()
        else:
            if 'a' in mode or 'w' in mode: _assert_directory(self.url)
            self._stream = open(self.url, mode=mode)
        return self
    def read(self, serializer=None):
        if serializer is None: serializer = PickleSerializer
        return serializer.load(self._stream)
    def write(self, data, serializer=None):
        if serializer is None: serializer = PickleSerializer
        serializer.dump(data, self._stream)
    def close(self):
        if isinstance(self._stream, _txt_stream):
            _assert_directory(self.url)
            with open(self.url, mode='w') as f:
                f.write(self._stream.get_text())
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            
    def exist(self):
        return _os.path.exists(self.url)
        
    # to allow use of "with"
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
        
    def load(self, submode='b', serializer=None):
        """
        open, read and return the object entry
        
        `submode` can be 
          - 'b' for binary file (or not for text content),
          - 'U' to assimilate all newline characters as '\n'
         """
        with self.open(mode='r'+submode):
            return self.read(serializer=serializer)
            
    def save(self,data, serializer=None):
        with self.open(mode='w'):
            return self.write(data, serializer=serializer)
            

def _get_extension(obj):
    return reduce(getattr,['__serializer__','extension'],obj)
    
    
class MapStorage(object):
    """
    A MapStorage allows automated IO to storage entries related to a unique id
    The iterface is done using `set_data(name,data)` and `load_data(name)`
    
    MapStorage use a url generator function to relate identifiers to entry url  
    """
    def __init__(self, url_generator):
        """
        Create a MapStorage based `url_generator`
        
        `url_generator` should either be:
          - A string that can be used with the python format functionality 
            Eg. "file://some/directory/data_{}"
          - A string to which is append the name of data to be saved.
            i.e. same as url_generator + "{}"
          - a function that returns the url with a key as input
            i.e. entry_url = url_generator(key)
        """
        if isinstance(url_generator, basestring):
            if not '{' in url_generator or not '}' in url_generator:
                url_generator += "{}"
            self.url_generator = url_generator.format
        else:
            self.url_generator = url_generator
        
    def set_data(self, name, data):
        """ stores `data` to the storage entry related to key = `name` """
        entry = self.get_entry(name, extension=_get_extension(data))
        entry.write(data)
        
    def get_data(self,name):
        """ retrieve data from the MapStorage object (i.e. load it) """
        entry = self.get_entry(name)
        entry.load(data)


    def has_entry(self, key):
        return self.__dict__.has_key(key)
        
    def get_entry(self, key, extension=''):
        """
        Return the entry related to `key`
        
        If it does not exist, its entry is automatically generated by this
        MapStorage url_generator, to which `extension` is appended.
        """
        if not self.has_entry(key):
            url = url_generator(key)+extension
            entry = create_entry(url)
            self.__dict__[key] = entry
        return self.__dict__[key]
        

