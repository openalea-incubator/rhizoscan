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

import cPickle as _cPickle
import os as _os
import urlparse as _urlparse
from tempfile import SpooledTemporaryFile as _TempFile

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
        return _cPickle.dump(obj, stream, protocol=_PICKLE_PROTOCOL_)
    @staticmethod
    def load(stream):
        return _cPickle.load(stream)
        

class _txt_stream(object):
    """ virtual writable text stream """
    def __init__(self):              
        self.txt = ''
        self.pos = 0
    def write(self, txt):
        print type(txt), '|' + txt + '|'
        self.txt += txt
    def get_text(self):
        return self.txt
    def close(self):
        self.txt = ''
    def tell(self):
        print 'tell:', len(self.txt)
        return len(self.txt)
    def seek(self, pos, whence=0):
        print 'seek', pos, whence, len(self.txt)

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
    __scheme__ = ('file', '')  # url scheme managed by the FileEntry class
    __meta_file__ = '__storage__.meta'  # name of file storing metadata 
    
    def __init__(self, filename):
        """
        Create a FileEntry object related to `filename`
        """
        ## `read`, `write`, `save` and `load` to have serializer arg?
        
        if len(filename): self.url = _os.path.abspath(_urlparse.urlsplit(filename).path)
        else:             self.url = filename
        self._stream = None
        
    # StorageEntry API
    # ----------------
    def exist(self):
        return _os.path.exists(self.url)
        
    def _load(self, submode='b', serializer=None):
        """
        open, read and return the object entry
        
        `submode` can be 
          - 'b' for binary file (or not for text content),
          - 'U' to assimilate all newline characters as '\n'
         """
        with self.open(mode='r'+submode):
            return self.read(serializer=serializer)
    def _save(self,data, serializer=None):
        with self.open(mode='w'):
            return self.write(data, serializer=serializer)
            
    def load(self):
        """
        open, read and return the object entry
         """
        stream = open(self.url, mode='r')
        serializer = self.get_metadata().get('serializer',PickleSerializer)
        data = serializer.load(stream)
        stream.close()
        
        #if self.get_metadata().get('serializer', None) is not None:
        #    print serializer.img_scale, serializer.ser_scale
        #    data.__serializer__ = serializer
            
        return data
            
    def save(self,data, serializer=None):
        stream = _TempFile()
        save_meta = True
        if serializer is None:
            serializer = PickleSerializer
            save_meta = False
        serializer.dump(data, stream)
        _assert_directory(self.url)
        with open(self.url, mode='w') as f:
            stream.seek(0)
            f.write(stream.read())
        
        if save_meta:
            self.set_metadata(dict(serializer=serializer)) ## what if write fails?
            
    def remove(self):
        """
        Remove the entry file (and metadata) from the file system
        """
        if _os.path.exists(self.url):
            _os.remove(self.url)
        self.rem_metadata()

    # private IO methods on file
    # --------------------------
    def open(self,mode):
        """
        mode='w' means write operation is done in a buffered, then saving is 
        done by the `close` method.
        """
        if self._stream is not None:
            raise IOError('File stream already open')
        if mode=='w':
            self._stream = _TempFile()
        else:
            _assert_directory(self.url)
            self._stream = open(self.url, mode=mode)
            
        return self
    def read(self):
        serializer = self.get_metadata().get('serializer',PickleSerializer)
        return serializer.load(self._stream)
    def write(self, data, serializer=None):
        if serializer is None:
            serializer = PickleSerializer
        else:
            self.set_metadata(dict(serializer=serializer))
        serializer.dump(data, self._stream)
    def close(self):
        if isinstance(self._stream, _TempFile):
            _assert_directory(self.url)
            with open(self.url, mode='w') as f:
                self._stream.seek(0)
                f.write(self._stream.read())
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            
    # allow use of "with"
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    # manage file entry IO on metadata file
    def _metadata(self):
        """ return metadata file and the key for this entry """
        dirname, filename = _os.path.split(self.url)
        meta_file = _os.path.join(dirname,FileEntry.__meta_file__)
        return meta_file, filename
    @staticmethod
    def _read_metadata_file(meta_file):
        if _os.path.exists(meta_file):
            try:
                f = open(meta_file, 'r')
                meta_head = _cPickle.load(f)
                f.close()
            except EOFError:
                print '\033]31mFileEntry: invalid content in metadata file ' + meta_file + '\033]30m'
                meta_head = dict()
        else:
            meta_head = dict()
        return meta_head
    @staticmethod
    def _write_metadata_file(meta_file, meta_data):
        f = open(meta_file, 'w')
        _cPickle.dump(meta_data, f)
        f.close()
        
    def set_metadata(self, metadata, update=True):
        """
        Save the all (key,value) of `content` in this entry metadata
        
        If update, update metadata with `content`.
        otherwise, replace the whole metadata set.
        
        FileEntry metadata are stored in a unique file in the same directory
        with name given in `FileEntry.__meta_file__`
        """
        meta_file, entry_key = self._metadata()
        meta_head = self._read_metadata_file(meta_file)
        
        if update:
            update = metadata
            metadata = meta_head.get(entry_key,dict())
            metadata.update(update)
        meta_head[entry_key] = metadata
        
        self._write_metadata_file(meta_file, meta_head)
            
    def get_metadata(self):
        """
        return this entry metadata (as a dict)
        """
        meta_file, entry_key = self._metadata()
        meta_head = self._read_metadata_file(meta_file)
        return meta_head.get(entry_key, dict())
        
            
    def rem_metadata(self):
        meta_file, entry_key = self._metadata()
        if _os.path.exists(meta_file):
            f = open(meta_file, 'r')
            meta_head = _cPickle.load(f)
            meta_head.pop(entry_key)
            f.close()
            if len(meta_head)==0:
                _os.remove(meta_file)
            else:
                f = open(meta_file, 'w')
                meta_head = _cPickle.load(f)
                f.close()
        

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
        serializer = getattr(data,'__serializer__', None)
        extension  = getattr(serializer,'extension', '')
        entry = self.make_entry(name, extension=extension)
        entry.save(data, serializer=serializer)
        
    def get_data(self,name):
        """ retrieve data from the MapStorage object (i.e. load it) """
        entry = self.get_entry(name)
        entry.load(data)


    def has_entry(self, key):
        return self.__dict__.has_key(key)
        
    def make_entry(self, key, extension=''):
        """
        Create the storage entry related to `key`
        
        The returned entry is generated with this MapStorage `url_generator`, 
        to which `extension` is appended.
        """
        url = self.url_generator(key)+extension
        entry = create_entry(url)
        self.__dict__[key] = entry
        return entry
        
    def get_entry(self, key):
        """
        Return the entry related to `key` if it exist
        """
        return self.__dict__[key]
        
    def __str__(self):
        return 'MapStorage(' + self.url_generator('{}') + ')'
    def __repr__(self):
        return self.__str__()

