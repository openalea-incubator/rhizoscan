"""
Implement I/O functionalities, mostly for Data class and subclass

FileObject: 
    classes that make the interface to some FileEntry instance (w.r.t url)
    implements __init__(url), dump() and load()
        
FileEntry:
    Manages IO for file for given communication protocol (eg. local file, sftp) 
    implements __init__(url), dump() and load()

FileStorage:
    Manages a set of FileObject related to an id (name)
    implements ...
   
   
TODO:
  - make a URL class that manages
      - splits of scheme, dirname, filename
      - container
      - FileEntry?  => open/write/exist/etc...
  - make FileEntry static
"""

import cPickle as _cPickle
import os as _os
from tempfile import SpooledTemporaryFile as _TempFile

from rhizoscan.misc.decorators import _property
from rhizoscan.misc.decorators import class_or_instance_method as _cls_inst_method
from rhizoscan.misc.path import assert_directory as _assert_directory

_pjoin  = _os.path.join
_psplit = _os.path.split
_dirname = _os.path.dirname

class Serializer(object):
    """ default serializer which use pickle
    
    It can be used has static `dump` and `load` functions, using protocol=-1 and 
    extension='.pickle'
    """
    _registered = {}
    protocol  = -1 # pickle saving (dump) are done with the latest protocol
    extension = '.pickle'  # file extension that indicate the file content type
    
    @_cls_inst_method
    def dump(self, obj,stream):
        return _cPickle.dump(obj, stream, protocol=self.protocol)
    @staticmethod
    def load(stream):
        return _cPickle.load(stream)

    @staticmethod
    def register(serializer, extensions):
        for ext in extensions:
            Serializer._registered[ext] = serializer
        
    @staticmethod
    def is_registered(extension):
        return Serializer._registered.has_key(extension)
    @staticmethod
    def get_registered(extension, **param):
        serializer = Serializer._registered[extension.lower()]
        if isinstance(serializer,basestring):
            module = serializer.split('.')
            sclass = module[-1]
            module = '.'.join(module[:-1])
            module = __import__(module,fromlist=[''])
            sclass = getattr(module,sclass)
            Serializer._registered[extension] = sclass
            serializer = sclass
        
        return serializer(**param)
        
    @staticmethod
    def get_serializer_for(obj,ext=None):
        """ return the suitable serializer for given `obj` 
        
        First, look for '__serializer__' attribute in `obj`. If found:
          - and it is a serializer-like object, returns it
          - and it is a string, return the serializer registered with that name 
          - and it is a list of string and parameter dict: return the serializer
            constructor registered with that name, and call it with given param
        
        Otherwise, if `ext` is given, look for registered serializer with that 
        extension.
        
        Other, returns the Serializer class
        """
        serializer = getattr(obj,'__serializer__',None)
        if serializer is None:
            if ext is not None and Serializer.is_registered(ext):
                serializer = ext
            else:
                return Serializer
                
        elif hasattr(serializer,'dump') and hasattr(serializer,'load'):
            return serializer
        elif isinstance(serializer,basestring):
            return Serializer.get_registered(serializer)
        elif hasattr(serializer,'__getitem__'):
            return Serializer.get_registered(serializer[0],**serializer[1])
        else:
            raise TypeError("Unable to construct suitable serializer")
        
Serializer.register('rhizoscan.image.PILSerializer',['.png','.jpg','.tiff','.tif','.bmp'])
Serializer.register('rhizoscan.root.graph.mtg.RSMLSerializer',['.rsml'])
        
def _urlsplit(url):
    """
    Split `url` and return scheme and path 
    """
    import urlparse
    split = urlparse.urlsplit(url)
    if len(split.scheme)<2:
        return 'file', url.replace('\\','/')
    else:
        return split.scheme, split.path
    
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
    def create_entry(url, container=None):
        """
        Create the suitable FileEntry for the given `url`
        
        If `url` is a FileEntry instance, return it
        If `url` does not contain a scheme (eg. file://'), file type is assumed
        """
        if isinstance(url, FileEntry):
            entry = url
        else:
            if container: 
                if container is True:
                    container, url = _psplit(url)
                base_url = container
            else:   
                base_url = url
            if hasattr(base_url,'get_url'):
                base_url = base_url.get_url()
                
            scheme = _urlsplit(base_url)[0]
            cls    = RegisteredEntry.get(scheme)
            entry  = cls(url, container)
        
        return entry
        
class Containable(object):
    """ Abstract class for storage object that can have a container """
    
    def set_container(self, container):
        if container is not None:
            container = _os.path.abspath(_urlsplit(container)[1])
        self._container = container
    def get_container(self):
        self._container = getattr(self,'_container',None)  ## deprecated
        return self._container
            

class FileEntry(Containable):
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
    """
    TODO:
      - make this a static class, and store url in FileObject
    """
    __metaclass__ = RegisteredEntry
    __scheme__ = ('file', '')  # manages local file & undefined scheme
    
    def __init__(self, url, container=None):
        self.set_container(container=container)
        self.set_url(url=url)
        
    def set_url(self, url):
        self._url = url
    def get_url(self, full=True):
        from rhizoscan.misc.path import abspath
        url = self._url if hasattr(self,'_url') else self.url   ## deprecated
        if full:
            return _pjoin(self.get_container(),_urlsplit(url)[1])
        else:
            return _urlsplit(url)[1]
        
    def open(self, mode):
        url = self.get_url()
        if 'w' in mode:
            _assert_directory(url)
        return open(url, mode=mode)
    def remove(self):
        url = self.get_url()
        if _os.path.exists(url):
            _os.remove(url)
    def exists(self):
        return _os.path.exists(self.get_url())
    
    def sibling(self, filename):
        """
        Create a FileEntry for the file `filename` in the same directory 
        """
        dirname, selfname = _psplit(self.get_url())
        sibling = _pjoin(dirname,filename)
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
    
    def __init__(self, url,container):
        """ Create a SFTPEntry from an url or a ParseResult object """
        FileEntry.__init__(self, url,container)
        
    def set_url(self, url):
        import urlparse
        if isinstance(url,urlparse.ParseResult):
            self._url = url.geturl()
            self._urlparse = url
        else:
            self._url = url
            self._urlparse = urlparse.urlparse(url)
            
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
            dirname = _dirname(self.urlparse.path)
            if not self.server.exists(dirname):
                self.server.mkdir(dirname)
        return self.server.open(self.urlparse.path, mode=mode)
    def remove(self):
        return self.server.remove(self.urlparse.path)
    
    def sibling(self, filename):
        """
        Create a FileEntry for the file `filename` in the same directory 
        """
        import urlparse
        p = self.urlparse
        dirname = _dirname(p.path)
        sibling = _pjoin(dirname,filename)
        sibling = urlparse.ParseResult(scheme=p.scheme, netloc=p.netloc, path=sibling, params=p.params, query=p.query, fragment=p.fragment)
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
        By default, FileEntry use `Serializer` (which use `cPickle`) to 
        serialize/deserialize data. However, `load` and `save` have suitable 
        parameter to provide alternate serializer.
        
        See also: `Serializer`, `FileEntry`
    """
    __meta_file__ = '__storage__.meta'  # name of 'header' file storing metadata 
    
    def __init__(self, url, container=None):
        """
        Create the FileObject related to file at `url`
        """
        self.entry = RegisteredEntry.create_entry(url,container)
        
    def set_container(self, container):
        self.entry.set_container(container)
        
    def get_url(self, full=True):
        return self.entry.get_url(full=full)
    def get_container(self):
        return self.entry.get_container()
        
    def get_extension(self):
        """ return this fileObject extension """
        return _os.path.splitext(self.get_url())[1]
        
    def exists(self):
        return self.entry.exists()
        
    def load(self, serializer=None):
        """
        open, read and return the object entry
        """
        stream = self.entry.open(mode='rb')
        if serializer is None:
            serializer = self.get_metadata().get('serializer')
        if serializer is None:
            ext = self.get_extension()
            if Serializer.is_registered(ext):
                serializer = Serializer.get_registered(ext)
            else:
                serializer = Serializer
        elif isinstance(serializer,basestring):
            serializer = Serializer.get_registered(serializer)
            
        data = serializer.load(stream)
        stream.close()
        
        return data
            
    def save(self,data):
        stream = _TempFile(mode='w+b')
        serializer = getattr(data,'__serializer__',None)
        serializer = Serializer.get_serializer_for(data, self.get_url())
        head = serializer.dump(data, stream)
        with self.entry.open(mode='wb') as f:
            stream.seek(0)
            f.write(stream.read())
        f.close()
        
        #if head is not None:#
        if serializer is not Serializer:
            self.set_metadata(dict(serializer=serializer))#serializer)) ## what if write fails?
            
    def remove(self):
        """
        Remove the entry file (and metadata) from the file system
        """
        self.entry.remove()
        self.rem_metadata()
        
    def __repr__(self):
        return self.__class__.__name__+"('"+self.get_url()+"')"
        

    # manage file entry IO on metadata file
    # -------------------------------------
    def _metadata(self):
        """ return metadata file and the key for this entry """
        meta_file = self.entry.sibling(FileObject.__meta_file__)
        entry_key = _psplit(self.entry.get_url())[-1]  
        return meta_file, entry_key
        
    def set_metadata(self, metadata, update=True):
        """
        Save all (key,value) of `metadata` in this metadata entry
        
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
                f = meta_file.open(mode='rb')
                meta_head = _cPickle.load(f)
                f.close()
            except EOFError:
                raise EOFError('FileEntry: invalid content in metadata file ' + meta_file.get_url())
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

class FileStorage(Containable):
    """
    A FileStorage allows automated generation of FileObject w.r.t to a unique id
    The interface is done using `get_file(name,data=None)`
    
    FileStorage use a url generator function to relate identifiers to file url  
    """
    def __init__(self, url_generator):
        """
        Create a FileStorage from `url_generator`
        
        `url_generator` should be a string that can be used with the python 
        format functionality to get the url: 
        
            Eg. "file://some/directory/data_{}"
        
        If '{} are missing, it is added to the end of the string.
        """
        if not '{' in url_generator or not '}' in url_generator:
            url_generator += "{}"
        self.url_generator = url_generator
        
    def _gen_url(self, key, extension=''):
        c = self.get_container()
        url = self.url_generator.format(key)+extension
        return _pjoin('' if c is None else c, url)
        
    def get_file(self, key, data=None):
        """
        Return the FileObject related to `key`
        
        If `data` is None: return the respective stored FileObject, if it exists
        Otherwise, create one for data and store it
        """
        if data is not None:
            serializer = Serializer.get_serializer_for(data)
            extension  = getattr(serializer,'extension', '') 
            
            # create file object, and store it
            url = self._gen_url(key=key, extension=extension)
            fobj = FileObject(url)
            self.__dict__[key] = fobj
            
        return self.__dict__[key]
        
    def has_entry(self, key):
        return self.__dict__.has_key(key)

    def __repr__(self):
        cls = self.__class__
        cls = cls.__module__ + '.' + cls.__name__
        return cls + "('" + self._gen_url('{}') + "')"
    def __str__(self):
        return self.__repr__()

MapStorage = FileStorage ## deprecated
