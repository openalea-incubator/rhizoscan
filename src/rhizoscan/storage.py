"""
Implement serialization functionalities for Data class and subclass

storage-entry: 
    class/concept to make the interface to some storage entry
        implement __init__(url), open('r'|'w'), read, write and close()
        
    FileEntry: default storage-entry
    
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
    _PICKLE_PROTOCOL_ = -1 # pickle saving (dump) are done with the latest protocol
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

class _RegisteredEntry(type):
   registered = {}
   def __init__(cls, name, *args, **kargs):
       for scheme in cls.__scheme__:
           _RegisteredEntry.registered[scheme] = cls
   @staticmethod
   def get(scheme):
       if scheme not in _RegisteredEntry.registered:
            raise LookupError('no entry class has been registered for {} scheme'.format(scheme))
       return _RegisteredEntry.registered.get(scheme)

class StorageEntry(object):
    """
    Abstract class that define the API of a storage entry.
    
    Also implement `create_entry()` that returns a suitable entry from an url.
    """
    def load(self,):      raise NotImplementedError('StorageEntry is an abstract class')
    def save(self,data):  raise NotImplementedError('StorageEntry is an abstract class')
    
    @staticmethod
    def create_entry(url): ## serializer?
        """
        Create the suitable Storage entry based on `url`
        
        If `url`is a StorageEntry instance, return it
        If `url` does not contain a scheme (eg. file://'), file type is used.
        
        Note: currently, only `file:` scheme is recognized.
        """
        if isinstance(url, StorageEntry): 
            return url
        scheme = _urlparse.urlsplit(url).scheme
        cls = _RegisteredEntry.get(scheme)
        return cls(url)
        
    def __str__(self):
        return self.__class__.__name__ + '(' + self.url + ')'
    def __repr__(self):
        return self.__str__()
        
    @_property
    def url(self):
        return self._url
    @url.setter
    def url(self, url):
        self._url = url
        
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
    """
    __metaclass__ = _RegisteredEntry
    __scheme__ = ('file', '')
    def __init__(self, filename, serializer='PickleSerializer'):
        """
        Create a FileEntry object related to `filename`
                      
        `serializer` is the default serializer object which must implement the 
        `dump` and `load` methods. By default use the `PickleSerializer` class.
        """
        ## `read`, `write`, `save` and `load` to have serializer arg?
        
        self.url   = _os.path.abspath(_urlparse.urlsplit(filename).path)
        if serializer=='PickleSerializer':
            serializer = PickleSerializer
        self._serializer = serializer
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
    def read(self):
        return self._serializer.load(self._stream)
    def write(self, data):
        self._serializer.dump(data, self._stream)
    def close(self):
        if isinstance(self._stream, _txt_stream):
            _assert_directory(self.url)
            with open(self.url, mode='w') as f:
                f.write(self._stream.get_text())
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            
    # to allow use of "with"
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
        
    def load(self, submode='b'):
        """
        open, read and return the object entry
        
        `submode` can be 
          - 'b' for binary file (or not for text content),
          - 'U' to assimilate all newline characters as '\n'
         """
        with self.open(mode='r'+submode):
            return self.read()
    def save(self,data):
        with self.open(mode='w'):
            return self.write(data)
            


class FileStorage(object):
    """
    A FileStorage allows automated IO to file entries related to a unique id
    The iterface is done `set_data(name,data)` and `load_data(name)`
    
    FileStorage use a string.format pattern to relate identifiers to filenames 
    """
    def __init__(self, file_format, serializer='default'):
        """
        Create a file storage on HD filesystem following `file_format`
        
        `file_format` should be a string that can be used with the python 
        format functionality (eg. "directory/data_{}.data") of simply a string
        to which is append the name of data to be saved: file_format+"{}"
        
        Object "url" is stored/retrieved from `file_format.format(obj_id)'
        
        `serializer` is passed to FileEntry, with object url.
        """
        if not '{' in file_format or not '}' in file_format:
            file_format += "{}"
        self.file_format = file_format
        
        if serializer=='cPickle':
            serializer = _cPickle
        self._serializer = serializer
        
    def set_data(self, name, data):
        """ stores `data` to file with key value `name` """
        entry = self.get_entry(name)
        entry.write(data)
        
    def get_data(self,name):
        """ retrieve data from FileStorage (i.e. load the file) """
        entry = self.get_entry(name)
        entry.load(data)

    def has_entry(self, key):
        return self.__dict__.has_key(key)
    def get_entry(self, key):
        if not self.has_entry(key):
            url = file_format.format(key)
            entry = FileEntry(url, self._serializer)
            self.__dict__[key] = entry
        return self.__dict__[key]
        

