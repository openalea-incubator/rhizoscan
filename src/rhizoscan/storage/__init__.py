"""
Implement I/O functionalities, mostly for Data class and subclass

FileObject: 
    classes that make the interface to some FileEntry instance (w.r.t url)
    implements __init__(url), dump() and load()
        
FileEntry:
    Manages IO for file for given communication protocol (eg. local file, sftp) 
    implements __init__(url), dump() and load()

MapStorage:
    Manages a set of FileObject related to an id (name)
    implements ...
              
"""

import cPickle as _cPickle
import os as _os
import urlparse as _urlparse
from tempfile import SpooledTemporaryFile as _TempFile

from rhizoscan.tool import _property
from rhizoscan.tool import class_or_instance_method as _cls_inst_method
from rhizoscan.tool.path import assert_directory as _assert_directory


class PickleSerializer(object):
    """
    basic pickle serializer
    
    It has static `dump` and `load` functions, using protocol=-1 and 
    extension='.pickle'
    """
    protocol  = -1 # pickle saving (dump) are done with the latest protocol
    extension = '.pickle'  # file extension that indicate the file content type
    
    @_cls_inst_method
    def dump(self, obj,stream):
        return _cPickle.dump(obj, stream, protocol=self.protocol)
    @staticmethod
    def load(stream):
        return _cPickle.load(stream)

class RegisteredEntry(type):
    """
    Metaclass used to register FileEntry w.r.t url scheme
    
    Registered FileEntry class must have the following attributes/constructor:
    
        `__metaclass__ = RegisteredEntry` (by default on FileEntry subclasses)
        `__scheme__`   a tuple of scheme name. e.g. ('file', '') 
        `__init__(url)` constructor that takes one url
        
    To get the FileEntry subclass registered for a scheme, use 
      `RegisteredEntry.get('scheme_name')`
      
    Note: It return the last registered FileEntry for the given scheme_name
    
    
    Base on the url, FileObject constructor automatically select the suitable 
    FileEntry using this register. Only the last registered FileEntry for a 
    given scheme is kept.
    
    See also: `FileEntry`, `FileObject`
    """
    register = {}
    def __init__(cls, name, *args, **kargs):
        schemes = getattr(cls, '__scheme__',[])
        for scheme in schemes:
            RegisteredEntry.register[scheme] = cls
            
    @staticmethod
    def get(scheme):
        if scheme not in RegisteredEntry.register:
             raise LookupError('no entry class has been registered for {} scheme'.format(scheme))
        return RegisteredEntry.register.get(scheme)
        
    @staticmethod
    def scheme():
        """ return the list of registered scheme """
        return RegisteredEntry.register.keys()
        
    @staticmethod
    def create_entry(url):
        """
        Create the suitable FileEntry for the given `url`
        
        If `url` is a FileEntry instance, return it
        If `url` does not contain a scheme (eg. file://'), file type is used.
        """
        if isinstance(url, FileEntry):
            entry = url
        else:
            scheme = _urlparse.urlsplit(url).scheme
            cls    = RegisteredEntry.get(scheme)
            entry  = cls(url)
        
        return entry

class FileEntry(object):
    """
    Class to manage I/O for FileObject - implements IO in local HD
    
    It provides the following API:
     - __init__(url): create an I/O interface for the local file at `url`
     - open(mode):    open and return the file in read 'r' or write 'w' mode
     - exists():      return True if the file exist
     - remove():      delete the file
     - url:           attribute that stores the url
     
     - sibling(filename): create an other FileEntry in the same directory
     
    open with mode='w' should create the file directory if necessary
    """
    __metaclass__ = RegisteredEntry
    __scheme__ = ('file', '')  # manages local file & undefined scheme
    
    def __init__(self, url):
        if len(url): 
            self.url = _os.path.abspath(_urlparse.urlsplit(url).path)
        else:
            print '*** invalid url for FileEntry ***' ## remove undefined url?
            self.url = url
        
    def open(self, mode):
        if 'w' in mode:
            _assert_directory(self.url)
        return open(self.url, mode=mode)
    def remove(self):
        if _os.path.exists(self.url):
            _os.remove(self.url)
    def exists(self):
        return _os.path.exists(self.url)
    
    def sibling(self, filename):
        """
        Create a FileEntry for the file `filename` in the same directory 
        """
        dirname, selfname = _os.path.split(self.url)
        sibling = _os.path.join(dirname,filename)
        return self.__class__(sibling)

class SFTPEntry(FileEntry):
    """
    Class to manage I/O for FileObject through sftp protocol
    
    It provides the following API:
     - __init__(url): create an I/O interface for the local file at `url`
     - open(mode):    open and return the file in read 'r' or write 'w' mode
     - exists():      return True if the file exist
     - remove():      delete the file
     - url:           attribute that stores the url
     
     - sibling(filename): create an other FileEntry in the same directory
     
    :login:
        In order to work, the calling OS should have a SSH key defined and 
        registered on the sftp (ssh) server. It should also have an ssh-agent
        started which mange this key.
        
    :dependencies:
        paramiko
    """
    __scheme__ = ('sftp',)  # manages sftp scheme
    
    connections = {}  # static list of connection already established
    
    def __init__(self, url):
        """
        Create a SFTPEntry from an url or a ParseResult object
        """
        if isinstance(url,_urlparse.ParseResult):
            self.url = url.geturl()
            self.urlparse = url
        else:
            self.url = url
            self.urlparse = _urlparse.urlparse(url)
            
        self._connect()
        
    def _connect(self):
        # connect to server
        if self.urlparse.username:
            user = self.urlparse.username
        else:
            import getpass
            user = getpass.getuser()
            
        key = (user,self.urlparse.hostname)
        if key in SFTPEntry.connections:
            self.server = SFTPEntry.connections[key]
        else:
            if self.urlparse.password:
                passw = self.urlparse.password
            else:
                import paramiko
                passw = paramiko.Agent().get_keys()[0]
                
            from . import pysftp_fork as pysftp
            self.server = pysftp.Connection(self.urlparse.hostname, username=user, private_key=passw)
            SFTPEntry.connections[key] = self.server
        
    def exists(self):
        return self.server.exists(self.urlparse.path)
    def open(self, mode):
        if 'w' in mode:
            dirname = _os.path.dirname(self.urlparse.path)
            if not self.server.exists(dirname):
                self.server.mkdir(dirname)
        return self.server.open(self.urlparse.path, mode=mode)
    def remove(self):
        return self.server.remove(self.urlparse.path)
    
    def sibling(self, filename):
        """
        Create a FileEntry for the file `filename` in the same directory 
        """
        p = self.urlparse
        dirname = _os.path.dirname(p.path)
        sibling = _os.path.join(dirname,filename)
        sibling = _urlparse.ParseResult(scheme=p.scheme, netloc=p.netloc, path=sibling, params=p.params, query=p.query, fragment=p.fragment)
        return self.__class__(sibling)
        
    def __getstate__(self):
        d = self.__dict__.copy()
        d['server'] = None
        return d
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._connect()
        return self
        
    
class FileObject(object):
    """
    Provide simplified I/O interface **with serialization** to & from file
    
    A FileObject instances have the following public methods:
    
      - load():   read and return content of file
      - save():   save data into the file
      - exists(): return True if the file exist
      - remove(): delete the file
      
    :Serializer:
        By default, FileEntry use `PickleSerializer` (which use `cPickle`) to 
        serialize/deserialize data. However, `load` and `save` have suitable 
        parameter to provide alternate serializer.
        
        See also: `PickleSerializer`, `FileEntry`
    """
    __meta_file__ = '__storage__.meta'  # name of 'header' file storing metadata 
    
    def __init__(self, url):
        """
        Create the FileObject related to file at `url`
        """
        self.entry = RegisteredEntry.create_entry(url)
        self._stream = None
        
    @_property
    def url(self):
        return self.entry.url
        
    def exists(self):
        return self.entry.exists()
        
    def load(self, serializer=None):
        """
        open, read and return the object entry
         """
        stream = self.entry.open(mode='r')
        if serializer is None:
            serializer = self.get_metadata().get('serializer',PickleSerializer)
        data = serializer.load(stream)
        stream.close()
        
        return data
            
    def save(self,data, serializer=None):
        stream = _TempFile()
        if serializer is None:
            serializer = PickleSerializer
        serializer.dump(data, stream)
        with self.entry.open(mode='w') as f:
            stream.seek(0)
            f.write(stream.read())
        
        if serializer is not PickleSerializer:
            self.set_metadata(dict(serializer=serializer)) ## what if write fails?
            
    def remove(self):
        """
        Remove the entry file (and metadata) from the file system
        """
        self.entry.remove()
        self.rem_metadata()
        

    # manage file entry IO on metadata file
    # -------------------------------------
    def _metadata(self):
        """ return metadata file and the key for this entry """
        meta_file = self.entry.sibling(FileObject.__meta_file__)
        entry_key = _os.path.split(self.entry.url)[-1]  
        return meta_file, entry_key
        
    def set_metadata(self, metadata, update=True):
        """
        Save the all (key,value) of `metadata` in this entry metadata entry
        
        If update, update metadata with `metadata` content.
        otherwise, replace the whole metadata dictionary.
        
        FileEntry metadata are stored in a unique file in the same directory
        as the FileEntry, with name given by `FileEntry.__meta_file__`
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
        
    @staticmethod
    def _read_metadata_file(meta_file):
        if meta_file.exists():
            try:
                f = meta_file.open(mode='r')
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
        f = meta_file.open(mode='w')
        _cPickle.dump(meta_data, f)
        f.close()
            
    def rem_metadata(self):
        meta_file, entry_key = self._metadata()
        if meta_file.exists():
            f = meta_file.open(mode='r')
            meta_head = _cPickle.load(f)
            meta_head.pop(entry_key)
            f.close()
            if len(meta_head)==0:
                meta_file.remove()
            else:
                f = meta_file.open(mode='w')
                meta_head = _cPickle.load(f)
                f.close()

class MapStorage(object):
    """
    A MapStorage allows automated IO to storage entries related to a unique id
    The interface is done using `set_data(name,data)` and `load_data(name)`
    
    MapStorage use a url generator function to relate identifiers to file url  
    """
    def __init__(self, url_generator):
        """
        Create a MapStorage based `url_generator`
        
        `url_generator` should be a string that can be used with the python 
        format functionality to get the url: 
        
            Eg. "file://some/directory/data_{}"
        
        If '{} are missing, it is added to the end of the string.
        """
        if not '{' in url_generator or not '}' in url_generator:
            url_generator += "{}"
        self.url_generator = url_generator
        
    def set_data(self, name, data):
        """ stores `data` to the FileObject related to key = `name` """
        serializer = getattr(data,'__serializer__', None)
        extension  = getattr(serializer,'extension', '')
        entry = self.make_entry(name, extension=extension)
        entry.save(data, serializer=serializer)
        
    def get_data(self,name):
        """ retrieve data from the MapStorage object (i.e. load it) """
        entry = self.get_file(name)
        entry.load(data)


    def make_entry(self, key, extension=''):
        """
        Create the FileObject related to `key`
        
        The returned entry is generated with this MapStorage `url_generator`, 
        to which `extension` is appended.
        """
        url = self.url_generator.format(key)+extension
        entry = FileObject(url)
        self.__dict__[key] = entry
        return entry
        
    def get_file(self, key):
        """
        Return the entry related to `key` if it exist
        """
        return self.__dict__[key]
        
    def has_entry(self, key):
        return self.__dict__.has_key(key)


    def __repr__(self):
        cls = self.__class__
        cls = cls.__module__ + '.' + cls.__name__
        return cls + "('" + self.url_generator.format('{}') + "')"
    def __str__(self):
        return self.__repr__()

