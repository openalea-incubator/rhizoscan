""" Data structures that relates run-time objects to storage files

Data classes
------------
 - Data:     base class of all other Data classes. 
             also provide simple IO interface using pickle.
 - Mapping:  a Data subclass that provide a dictionary-like functionality.
 - Sequence: a Data subclass that provide a list-like functionality.
 
TODO:
-----
    - look into the '__io_mode__' idea ('r','w','u') ?
    - replace the _(un)serialize_ by the set/get_state protocol
       - multi-file serialization (pickle or not)
    - remove Sequence, replace by (buffered) external storage in Mapping
"""
__icon__    = 'datastructure.png'   # package icon (for openalea)    

from copy   import copy as _copy

from .tool     import static_or_instance_method, _property 
from .workflow import node as _node # to declare workflow nodes
from .storage  import FileObject as _FileObject
from .storage  import MapStorage as _MapStorage

class Data(object):
    """ 
    Basic Data class: provide automated storage IO operation
    
    The Data class has two aims:
      1) It can be used to relate some data to a FileObject (eg. a file), and 
         use the Data saving and loading interface (pickle based).
         See the constructor documentation for details.
      2) It can be used as a superclass in order to get the IO functionalities
         of the `Data` class (see below), and provide special behavior for 
         objects being saved or loaded by container Data (`Mapping`, `Sequence`)
         by overriding the `__store__` and `__restore__` method.
         
    **Note for subclassing**:
    
        As stated, one of the main goal of the `Data` class is to be subclassed. 
        Subclasses can either keep or override the dump, load, __store__ 
        and __restore__ methods.
        
        - The Data dump and load methods are simple interface with the pickle 
          modules. The dump method save the objects (more precisely what is 
          returned by the __store__ method). And the load method loads it 
          (more precisely what the loaded object __restore__ method returns).
          
        - Save and load are primarily supposed to be called statically. Thus
          when used as instance methods, it should be asserted that their
          behavior are suitable (especialy load) or require overriding. 
          * See dump and load documentation for details *
          
        - The container Data class do not call the Data's dump and load method. 
          So they can be overriden with no restriction.
          ## this might not be True anymore...
          
        - However they call the __store__ and __restore__ method, which
          are also called by the default implementation of dump and load.
          
        - By default, the __store__ and __restore__ methods simply 
          return the calling instance (i.e. self).
          
        - However, if it is wanted to do some processing prior to saving and/or
          post loading, then it can be done by overriding the __store__ and 
          __restore__ methods.
          
        - The only constraint is to keep the same signature working: they should 
          be callable with only self as input, and return the object to be saved
          and the loaded object, respectively
          In both cases the returned object can be different from the calling
          one. But the saved object returned by __store__ should have an
          __restore__ method if it is wanted to be called at loading.
          
        The main reasons to do such overriding can be:
          - if the merging ability of the load method is not suitable (see load doc)
          - to remove some (e.g. storage-consuming) data from the objects before
            saving. And possibly recompute them at loading.
          - to save/load some data in a different way then using pickle 
          - to define a saving/loading approach that manage un-picklable data:
            typically if the object uses definition (function or class) that are
            not loadable where the object might be loaded (see pickle doc). 
            
      :todo:
        - add read/write/update modes attributes
      """
    ##Data todo:
    ## -  a static dataWrapper method that create a savable/readable class
    ##    (and instance) from any class(object) 
    #       => dynamic superclassing for new-style classes ? Dangerous
    #       => or simply add suitable methods to make them 
    #       => what about old-styles one ?
    ## - make a static method / function isValidData(...) that check if the 
    ##   right attribute exist 
    
    def __init__(self, file_url=None, serializer=None):
        """                      
        Create an empty Data object that can be related to `file_url`
        
        `file_url` can be a filename, a valid url string or a 
        `storage.FileObject` object.
        
        See `datastructure.storage` documentation. 
        """
        self.set_file(file_url)
        self.set_serializer(serializer)
        
    @static_or_instance_method
    def set_file(self, file_url):
        """
        set the FileObject of this Data object
        
        `file_url` can be either:
          - a filename, url string or a FileObject
          - None, to remove storage link
          - -1 for (attempted) removale of the file
          
        See `storage` documentation. 
        """
        ## if file obj is already used: what to do?  
        if file_url is None:
            pass
        elif file_url==-1:
            old_file = self.get_file()
            if old_file:
                old_file.remove()
            file_url = None
        if file_url is not None and not isinstance(file_url,_FileObject):
            file_url = _FileObject(file_url)
            
        self.__file_object__ = file_url
        
    @static_or_instance_method
    def get_file(self):
        """
        return the file of this data for saving and loading
        """
        return getattr(self,'__file_object__',None)
    
    @staticmethod
    def has_IO_API(obj):
        """
        Test if given `obj` has the Data I/O API:
        if it has the attributes `set_file`, `get_file`,
        `dump` and `load`.
        """
        return all(map(hasattr,[obj]*4, ['set_file','get_file','dump','load']))
    @staticmethod
    def has_store_API(obj):
        """
        Test if given `obj` has the "store" API:
        if it has the attributes `__store__` and `__restore__`
        """
        return all(map(hasattr,[obj]*2, ['__store__','__restore__']))
        
    @static_or_instance_method
    def set_serializer(obj_or_self, serializer):
        """
        Set the `obj_or_self` serializer
        
        `serializer` should implement `dump` and `load` functions
        
        This method can be called either
          - statically:     Data.set_serializer(obj, serializer)
          - or on instance: data_obj.set_serializer(serializer)
        """
        return setattr(obj_or_self,'__serializer__',serializer)
        
    @static_or_instance_method                           
    def get_serializer(obj_or_self):
        """
        return the `obj_or_self` serializer if it has it, or None otherwise
        
        This method can be called either
          - statically:     Data.get_serializer(obj)
          - or on instance: data_obj.get_serializer()
        """
        return getattr(obj_or_self,'__serializer__',None)
        
    @static_or_instance_method                           
    def dump(data, file_object=None):
        """ 
        Save input `data` to `file_object` (which can be a file name) 

        This method can either be called as a:
          1. static   method:  Data.dump(non_Data, file_object,   protocol=None)
          2. static   method:  Data.dump(Data_Obj, file_object,   protocol=None)
          3. instance method:  someDataObj.dump(file_object=None, protocol=None)
        
        If the 1st argument has the "store" API, this function saves the value
        returned by its `__store__` method.
       
        :Inputs:
          - `file_object`
            Can be either a `FileObject` instance or the url of a file.
            If it is not a FileObject, the value of `file_object` is passed to 
            `storage.FileObject()` (see FileObject doc)
            
            In the case (1), `file_object` argument is mandatory. 
            In the case (2) and (3),  if `file_object` is not given (None) the 
            instance Data `__file_object__` attribute is used. In this case, 
            `file_object` becomes the `__file_object__` attribute of `data`.
        
        :Outputs:
            Return an empty Data object that can be use to load the stored data
        """
        if 'w' not in getattr(data,'__io_mode__','w'):
            raise IOError("This Data object is not writable. mode is {}".format(getattr(data,'__io_mode__')))
        
        io_api = Data.has_IO_API(data)
        if file_object is None:
            if io_api:
                file_object = data.get_file()
            else:
                raise TypeError("argument 'file_object' should be given when 'data' does not have the __file_object__ attribute")

        if io_api:
            data.set_file(file_object)
            file_object = data.get_file()
        elif not isinstance(file_object, _FileObject):
            file_object = _FileObject(file_object)
            
        if Data.has_store_API(data): to_store = data.__store__()
        else:                        to_store = data
        
        file_object.save(to_store)
        
        if io_api and hasattr(data,'loader'): 
            return data.loader()
        else:
            return Data(file_object).loader()  ## if lookup in 'Data'base => equiv to return self
        
    @static_or_instance_method
    def load(url_or_data, update_file=True):
        """ 
        Load the data serialized at `url_or_data` 

        This method can be used as
          1. a static method with url:         Data.load(filename)
          2. a static method with data object: Data.load(data_obj)
          3. an instance method:               data_obj.load()
        
        If the loaded content has the "store" api (such as Data objects) then 
        this function return the output of the `__restore__` method.
        
        If `update_file` is True and the loaded object has the IO api, then its
        __file_object__ attribute is set to the loaded url. 
        """
        data = url_or_data  # for readibility
        
        if Data.has_IO_API(data):
            file_object = data.get_file()
        else:
            file_object = _FileObject(data)
            
        data = file_object.load(serializer=Data.get_serializer(data))
        
        if Data.has_store_API(data):
            data = data.__restore__()
            
        if update_file and Data.has_IO_API(data):
            data.set_file(file_object)
           
        if Data.is_loader(data):
            io_mode = getattr(data,'__io_mode__','').split(':')[-1]
            if len(io_mode): 
                data.__io_mode__ = io_mode
            else:
                del data.__io_mode__
        
        return data
       
    def __store__(self):
        """ 
        Prior data processing before saving. By default return it-self
        
        :Subclassing:
          Overriding this method can serves several purpose:
            1. this is where useful pre-saving cleaning can be done, 
               such as deleting dynamic data.
            2. If the data object cannot by pickled. Overriding this 
               method as well as __restore__ enable to make a workaround.

          The return value should be picklable: (typically) its class definition 
          and all its methods and attributes type must be hard coded.
          
          Note that the return object should have the suitable `__restore__` 
          method callable when reloaded.
          
        :See also: __restore__
        """
        return self
        
    def __restore__(self):
        """
        Postprocessing of loaded data. By default return it-self.

        :Subclassing:
          Overridind this methods allows typically to reverse changes made by 
          the `__store__` method.
            
        :See also: __store__
        """
        return self
        
    def __parent_store__(self):
        """
        Return what should be stored by (parent) container:
          - self.__store__() if it has no __file_object__ set
          - its loader if it does
        """
        if self.get_file() is None:
            return self.__store__()
        else:
            return self.loader()
        
    def loader(self):
        """
        Return an empty object which can be used to load the stored object
        
        If this object has a `__loader_attributes__`, it should contain a list 
        of attribute names that are added the the loader attributes
        
        *** if the object has no __file_object__ set, the loader won't load ***
        """
        loader = Data(file_url=self.get_file(),serializer=self.get_serializer())
        loader.__io_mode__ = 'loader:' + getattr(self,'__io_mode__','')
        loader.__load_class__ = self.__class__
        for attr in getattr(self,'__loader_attributes__',[]):
            setattr(loader,attr,getattr(self,attr,None))
        return loader
    @staticmethod
    def is_loader(obj):
        return getattr(obj,'__io_mode__',"").startswith('loader:')
        
    def __str__(self):
        if Data.is_loader(self):
            cls = 'Data loader'
        else:
            cls = self.__class__
            cls = cls.__module__ +'.'+ cls.__name__
        return  cls + ' with file: ' + str(self.get_file())
            

            
@_node('data_loader')
def save_data(data, filename):
    """
    Use Data class to save input 'data'. Return a Data instance.
    """
    return Data.dump(data=data,filename=filename)

        

# Data subclass with dictionary interface
# ---------------------------------------
class Mapping(Data):
    """
    Class Mapping provide a Data class with a dictionary-like interface
    
    Example:
    --------
        m1 = Mapping()
        m1.q1 = 6
        m1.q2 = 9
        m1.ans = m1.q1 * m1.q2
        print repr(m1)            # q1:6  q2:9  ans:54
        
        m2 = Mapping(the_question='6x9', ans=42)
        m1.update(m2,False)
        print m1.ans              # 54
        m1.update(m2)
        print m1.ans              # 42

    Mapping object can be treated as dictionaries:
    ----------------------------------------------
        * the double-star operator can be used: 
            some_function(**myMappingObject)
            
        * implement iterator. However, iteration return tuple (key,value):
            m = Mapping(a=42,z=0,t=None)
            for key,value in m: print key, 'has value', value
         
    Storage:
    --------
     1. Mapping are Data objects and have the attribut `__file_object__` 
        reserved. Overwriting it will induce failure of IO functionalities.
     2. Mapping can be use as a container through the `set_container` method,
        then using the function `set` and  `get` (see docs for details)
    """
    ##TODO Mapping:
    ##  - dump doc (for now it's the Data.dump doc)
    ##  - saving using pyyaml/json (by default, keeping pickle saving in option)
    ##  - what about saving to intermediate file when in a container ?
    #      > useful ?
    #      > then needs some 'update' flag ?
    def __init__(self, *args, **kwds):
        """
        Create a Mapping object 
        
        Possible arguments are:
         - keywords:                                  Mapping(a=1,b=2)
         - list of (key,value) pairs:                 Mapping([('a',1),('b',2)])
         - a mapping object (i.e. with iteritems()):  Mapping({'a':1,'b':2})
         - any combinasion of the above
        """
        for arg in args: self.__dict__.update(arg)
        self.__dict__.update(kwds)
        
    # accessors
    # ---------
    def __getattribute__(self, attribute):
        value = Data.__getattribute__(self,attribute)
        if Data.is_loader(value):
            value = value.load()
            setattr(self,attribute,value)
        return value
    __getitem__ = __getattribute__
    __setitem__ = Data.__setattr__
   
    def __len__(self):
        return self.__dict__.__len__()
        
    def has_key(self, key):
        """ return True if `key` exists, False otherwise """
        return self.__dict__.has_key(key)
    def keys(self):
        """ Return the list of this Mapping keys (same as dict) """
        return self.__dict__.keys()
    def values(self):
        """ return the list of this Mapping values (same as dict) """
        return self.__dict__.values()

    def set(self,key, value=None, store=False):
        """ 
        Set the `key`,`value` pairs (use `value=None` if not provided)
        
        If this object has a `storage` and store is True: store value to it and 
        set/update the `value` Data file 
        
        Note: this method simply calls `__setitem__`   
        """
        self.__dict__[key] = value
        if store:
            Data.set_file(value,None)                               ## wrong way of doing it?!
            if hasattr(value,'__parent_store__'):                   ##
                val_to_store = value.__parent_store__()             ##  should just call value.__store__()?
            else:                                                   ##
                val_to_store = value                                ##
            self.__map_storage__.set_data(key, val_to_store)        ##
            self.__map_keys__.add(key)                              ##
            Data.set_file(value,self.__map_storage__.get_file(key)) ##
        return self
    def setdefault(self,key, value=None):
        """ 
        M.setdefault(key,value) = M.get(key,value), with M[key]=value if key not in M
        """
        return self.__dict__.setdefault(key,value)
        
    def get(self,key,default=None):
        """ return the value related to `key`, or `default` if `key` doesn't exist
        
        Note: this method simply calls `__getitem__`
        """
        if not self.__dict__.has_key(key):
            return default
        else:
            return self.__getattribute__(key)
    def pop(self,key,default=None):
        """ remove `key` and return its value or default if it doesn't exist' """ 
        return self.__dict__.pop(key,default)

    def update(self, other, overwrite=True):
        """ add `other` content into this Mapping object
        ## todo:
            - other a dict-like object (use update(**other) ?)
            - add **kargs arguments
            - remove the overwrite argument, and add a setdefault/union method
        """
        if isinstance(other,Mapping):   ## if hasattr(other,'__dict__') ?
            other = other.__dict__
        if overwrite:
            self.__dict__.update(other)
        else:
            for k,v in other.items():
                if k not in self.__dict__:
                    self.__dict__[k] = v

    # iterators
    # ---------
    def iteritems(self):
        """ return an iterator over the (key, value) items of Mapping object """
        return self.__dict__.iteritems()
    def itervalues(self):
        """ return an iterator over the values of Mapping object """
        return self.__dict__.itervalues()
    def iterkeys(self):
        """ return an iterator over the keys of this Mapping object """
        return self.__dict__.iterkeys()
        
    # make Mapping a valid iterator
    def __iter__(self):
        return self.iteritems()#__dict__.__iter__() ##?
    def __contains__(self, key):
        return key in self.__dict__


    # temporary attributes
    # --------------------
    @_property
    def temporary_attribute(self):
        """ set of temporary attribute which are not to be saved """
        if not hasattr(self,'_tmp_attr'):
            self._tmp_attr = set()
        return self._tmp_attr
    def clear_temporary_attribute(self, name='all', raise_missing=None):
        """ clear the temporary attributs list, and optinally delete associated data """ 
        if name=='all': name = list(self.temporary_attribute)
        if isinstance(name,basestring): name = [name]
        for attr in name:
            if hasattr(self,attr):
                delattr(self,attr)
            elif raise_missing:
                raise KeyError('Warning: missing temporary attribute %s' % attr)
            self.temporary_attribute.discard(attr)
    
    # manage storage for (de)serializer
    # ---------------------------------
    def __store__(self):
        """
        Return a copy of it-self and call recursively __store__ on all 
        contained objects that have the __store__ method, such as Data objects. 
        
        Note: This is what is really save by the 'save' method.
        """
        s = self.__copy__()
        s.clear_temporary_attribute()
        
        d = s.__dict__
        for key,value in d.iteritems():
            if hasattr(value,'__parent_store__'):
                d[key] = value.__parent_store__()
            
        return s

    def __restore__(self):
        """
        Return it-self after calling `__restore__` on all contained items that
        have this method, such as Data objects.
        """
        for key,value in self.iteritems():
            if hasattr(value,'__restore__') and hasattr(value.__restore__,'__call__'):
                try:
                    self[key] = value.__restore__()
                except: pass##print 'no data file'
            
        return self
    
    def set_map_storage(self, storage, keys=[]):
        """
        Attach MapStorage for automatized I/O of contained attributs objects.
        
        :Inputs:
          - `storage`: 
                Either a MapStorage object, a valid input of its constructor
                (ie. format-type string), or `None` to remove current MapStorage
          - `keys`:
                Either a list of keys which are to be stored in this storage
                or 'all', to use this storage for all keys
        
        :See also: storage.MapStorage
        """
        if not isinstance(storage, _MapStorage) and storage is not None:
            storage = _MapStorage(storage)
        self.__map_storage__ = storage
        self.__map_keys__ = set(keys)
        
    def load(self, filename=None, overwrite=True):
        """
        Load data found in file 'filename' and merge it into the structure

        :Inputs:
          - filename
              If None, load file given by this object Data file attribute.
              The loaded file should be a Mapping object saved using the save 
              method, or anything that can be loaded by the Data save method 
              (i.e. pickle files) and can be used as the `update` method input
              
          - overwrite: 
              If True, loaded item overwrite existing one with same key. 
              Otherwise, it don't. (same as for the `update` method)
              
          ##not implemented
          ##  todo: should be some "container" stuff
          ##        ex: container have a __base_dir_num__ that says how many 
          ##            dir backward is the start of movable content....
                      
        :Output:
            return it-self, updated with loaded content
        """
        if filename is None:
            filename = self.get_file().url
        loaded = Data.load(filename, update_file=True)
        
        self.update(loaded, overwrite=overwrite)
            
        return self
        
    def attempt_load(self):
        """  If this object has a data file, load its content """
        fileobj = self.get_file()
        if fileobj:
            self.load()
            
    # copy Mapping objects
    # --------------------
    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new._tmp_attr = self.temporary_attribute.copy()
        return new
    copy = __copy__
    
    def __change_dir__(self, old_dir, new_dir, load=False, verbose=False, _base=''):
        """
        Change url of all content stored in `old_dir` to `new_dir`

        This is used to update file url of **already moved** content 
        It does not actually move the files.
        
        :Inputs:
          - `old_dir`: the (start of) the path to be replaced   (*)
          - `new_dir`: the (start of) the path to replace it by (*)
          - `load`:
              attempt first to load the content of this object data file
          - `verbose`:
              If True, print a line for each changed made
              
        (*) always include the directory separator '\' or '/'
        if `load`, attempt to load this object data file content first
        if `verbose`, print a line for each applied correction
        """
        import os
        old_len = len(old_dir)
        
        def change_if_required(url):
            if url[:old_len]==old_dir:
                url_end = url[old_len:]
                if url_end[0]==os.sep:
                    url_end = url_end[1:]
                new_url = os.path.join(new_dir,url_end)
                return new_url
            else:
                return None
                
        # update self file
        self_file = self.get_file()
        if self_file:
            new_file = change_if_required(self_file.url)
            if new_file:
                self.set_file(new_file)
                if verbose: print _base,'file url:',new_file
            
        if load:
            self.attempt_load()
            
        # update __map_storage__
        if hasattr(self,'__map_storage__'):
            new_gen = change_if_required(self.__map_storage__.url_generator)
            if new_gen:
                self.__map_storage__.url_generator = new_gen
                if verbose: print _base,'url gen',new_gen
                
        # update attributes (including strings)
        for k,v in self.__dict__.iteritems():
            base = _base+'.'+k if len(_base) else k
            
            if hasattr(v,'__change_dir__'):
                v.__change_dir__(old_dir=old_dir, new_dir=new_dir, verbose=verbose, _base=base)
                
            elif Data.has_IO_API(v): 
                f = v.get_file()
                if f is None: continue
                
                new_url = change_if_required(f.url)
                if new_url:
                    v.set_file(new_url)
                    if verbose: print base,':',new_url
                    
            elif isinstance(v,basestring):
                new_v = change_if_required(v)
                if new_v:
                    self[k] = new_v
                    if verbose: print base,':',new_v
        
            
    # string representation of Mapping content
    # ----------------------------------------
    def __repr__(self):
        return self.multilines_str()[:-1]##self.__dict__.__repr__()
    # nice printing
    def __str__(self):
        #from pprint import pformat
        #return pformat(self.__dict__)
        #return "length %d %s object with associated file: %s" % (len(self), self.__class__.__name__, self.get_file()) # self.multilines_str()
        cls_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return cls_name+":"+str(self.__dict__)
    def display(self, tab=0, max_width=80, avoid_obj_id=None):
        """ same as print, but give access to arguments  
        see multilines_str for details """
        print self.multilines_str(tab=tab, max_width=max_width, avoid_obj_id=avoid_obj_id)
    def multilines_str(self, tab=0, max_width=80, avoid_obj_id=None, self_name=''):
        """ multilines string representation, potentially hierarchical 
        
        :Inputs: 
          - tab:
              number of tab to start all lines with                                       
          - max_width:
              maximum number of character per lines
              
          - avoid_object_id: (optional) 
              None or a dictionary of python object id as key and value to print
          - self_name: (optional) 
              name to be printed for current object, if referenced by sub-item
        """
        if avoid_obj_id is None:
            avoid_obj_id = dict()
        avoid_obj_id[id(self)] = '&' + self_name
        
        string = ''
        keys = sorted([f for f in self.keys() if not f.startswith('_')])
        for key,value in self.iteritems():
            key = str(key)
            if key.startswith('_'): continue
            
            name  = '    '*tab + key + ': '
            shift = ' '*len(name)
            
            if id(value) in avoid_obj_id:
                string += name + '<%s: %s>\n' % (value.__class__.__name__,avoid_obj_id[id(value)])  
            elif hasattr(value, 'multilines_str'):
                string += name + '%s\n' % value.__class__.__name__
                if len(self_name): subname = self_name + '.' + key
                else:              subname = key   
                string += value.multilines_str(tab=tab+1, max_width=max_width, avoid_obj_id=avoid_obj_id, self_name=subname)
            else:
                value = str(value).splitlines()
                value = [shift + v for v in value]
                value[0] = name + value[0][len(name):]
                value = [v if len(v)<=max_width else v[:max_width-4]+' ...' for v in value]
                string += '\n'.join(value) + '\n'
                
        if len(string):
            return string
        else:
            return "Empty " + Data.__str__(self)

    # numerical comparison
    # --------------------
    def __cmp__(self,other):
        """ x.__cmp__(y) <==> cmp(x,y) """ 
        return self.__dict__.__cmp__(getattr(other,'__dict__',other))
    def __eq__(self,other):
        """ x.__eq__(y) <==> x==y """
        return self.__dict__.__eq__(getattr(other,'__dict__',other))
    def __lt__(self,other):
        """ x.__lt__(y) <==> x<y """
        return self.__dict__.__lt__(getattr(other,'__dict__',other))
    def __le__(self,other):
        """ x.__le__(y) <==> x<=y """
        return self.__dict__.__le__(getattr(other,'__dict__',other))
    def __gt__(self,other):
        """ x.__gt__(y) <==> x>y """
        return self.__dict__.__gt__(getattr(other,'__dict__',other))
    def __ge__(self,other):
        """ x.__ge__(y) <==> x>=y """
        return self.__dict__.__ge__(getattr(other,'__dict__',other))
        
@_node("key-value", auto_caption=1)
def get_key(data={}, key='metadata', default=None):
    """
    Return `data[key]`, or `default` if it does not exist.

    Either use the `data.get` method, if it exist, or `getattr` on data 
    """
    if hasattr(data,'get'): return data.get(key, default)
    else:                   return getattr(data,key,default)
        
# Data that manages sequence
class Sequence(Data):
    """
    Simple read-write interface for a sequence of Data objects
    
    :TODO: 
      - make it a Data Container
      - doc
          > setting might save. 
          > setting Data obj change its Data file attribute (?)
      - manage Data_IO_arg: to be send to _data_to_load/save_ ?  
          > then make suitable change in Data doc, and ImageSequence
      - copy: normal copy followed by a clear buffer (how?)
    """                                            
    def __init__(self, files=None, output=None, buffer_size=2, auto_save=True):
        """
        either files with a glob or file list, or output should be given.
        If both files is used.
        output : make an output sequence, see set_output doc.
        buffer_size: -1 means all (see property doc)
        """
        self.__seq_file  = []          
        self.__output    = None
        self.__index     = 0            # current iterator position
        self.buffer_size = buffer_size  # create buffer of suitable length
        self.auto_save   = auto_save    # automatic saving when setting data
        
        if files is not None:
            if isinstance(files,basestring):    # Load file list using pattern
                from glob import glob
                self.__seq_file = glob(files)
            else:                               # copy (ref) given file list
                self.__seq_file = files
        elif output is not None:                # make an output sequence
            self.output = output
        else:
            raise TypeError("either files (input) or output (output) argument must be given")
        
    @_property
    def output(self):
        return self.__output
    @output.setter
    def output(self, output):
        """
        Set output naming of sequence files.
        output should be a "printf" string containing 1 and only 1 '%d' element:
        
            E.g. /tmp/image%03d.data
            
        Note that if this files already has a list of files, this naming will 
        take over. To remove output capabilities, use set_input() 
        """
        if output == '_REMOVE_OUTPUT_':
            self.__output = None
        elif not isinstance(output,basestring):
            raise TypeError("output should be a string")
        else:
            try:
                output % 0
            except TypeError as e:
                raise TypeError("invalid output (" + e.strerror + ")")
            
            self.__output = output
    
    def set_input(self):
        """ remove output capability """
        self.output = '_REMOVE_OUTPUT_'
    
    # buffering:
    # ----------
    @_property
    def buffer_size(self):
        """ buffer size of the Sequence. changing its value clears the buffer content """
        return len(self.__buffer)
        
    @buffer_size.setter
    def buffer_size(self, size):
        if size<0: size = len(self)
        self.__buffer = [(None,None)]*size
        
    def clear_buffer(self):
        self.buffer_size = self.buffer_size

    def _set_buffer_item_(self,item, index):
        """ 
        Stores 'item' at position 'index' in the buffer
            
        Note for subclassing:
        ---------------------
        This method is called by __setitem__ (i.e. the [] operator). It can be 
        overriden by subclasses to change buffering method. In this case, the 
        subclass should also create its own (private) buffer storage
        """
        self.__buffer[index%len(self.__buffer)] = (item,index)
        
    def _get_buffer_item_(self,index):
        """ 
        return the buffer item at position index
        
        Output:
        -------
            item:  what is stored where element at 'index' should
            index: the index of the return item (can be different from input)
            
        Warning:
        --------
        if the returned index is different from the input, it means that the 
        buffer does not contain the item.
        
        Note for subclassing: see _set_buffer_item_ documentation
        """
        return self.__buffer[index%len(self.__buffer)]


    # file stuff:
    # -----------    
    def get_file(self,index=None):
        """
        Return the file of sequence element specified by 'index'.
        
        If index is None, return the list of all sequence files. 
        """
        if index is None:         return self.__seq_file
        if self.__output is None: return self.__seq_file[index]
        else:                     return self.__output % index
        
    def insert_file(self,filename,index):
        if index>=self.__seq_file.__len__():
            self.__seq_file.extend( ('_UNSET_SEQUENCE_FILE_',)*(index-self.__seq_file.__len__()+1))
        self.__seq_file[index] = filename
        
    def __len__(self):
        return self.__seq_file.__len__()


    # loading and saving:
    # -------------------
    def _load_item_(self, filename, index):
        """ 
        Load item from file 'filename'. By default use the Data.load method.
        
        This is called by the __getitem__ method it can be overrided by subclasses """
        return Data.load(filename)
                
    def _save_item_(self, filename, item):
        """ 
        Save 'item' to file 'filename'. By default use the Data.dump method
        
        This is called by the __setitem__ method and can be overrided by subclasses
        """
        Data.dump(item,filename)
                
    def __store__(self):
        s = _copy(self)
        s.clear_buffer()
        return s

    # accessor:
    # ---------
    @_property
    def auto_save(self):
        """ Property for automatic saving behaviors
        
        Its value indicates weither setting elements induce saving (True), or 
        only add them to the Sequence list (False).
        
        Saving means that the __setitem__ [] operator calls _save_item_ method
        """
        return self.__auto_save
    @auto_save.setter
    def auto_save(self,save_it):
        self.__auto_save = save_it
        
    def __getitem__(self, index):
        """
        implement the [] operator for reading.
        If possible, retrieve the data from buffer, otherwise load it and buffer it 
        """
        item,ind = self._get_buffer_item_(index)
        
        # load data if it is not buffered
        if item is None or ind!=index:
            item = self._load_item_(self.get_file(index), index)
            self._set_buffer_item_(item,index)

        return item
        
    def __setitem__(self, index, data):
        """
        implement the [] operator for writing.
        
        if item is a Data object, set its Data file attribute to the suitable
        Sequence filename
        
        Warning: Sequence can only write if it has a valid output attribute.
        See set_output and constructor documentation
        """
        if self.__output is None:
            raise AttributeError("This sequence is read only.")
            
        filename = self.get_file(index)
            
        # save data and buffer it
        if self.auto_save: self._save_item_(filename,data)
        if isinstance(data,Data): data.set_file(filename)
        self._set_buffer_item_(data,index)
        
        # add this file to the file list
        self.insert_file(filename,index)
        
    def __getslice__(self, i,j):
        """ Return a slice of this sequence. Output functionality is lost """
        sub = _copy(self)
        sub.__seq_file = self.__seq_file[i:j]
        sub.clear_buffer()
        return sub ##Sequence(files=self.__seq_file[i:j], buffer_size=self.buffer_size)
        

    # iterator:
    # ---------
    def __iter__(self):
        """ initiate iterator of sequence. Use next() to process iteration"""
        self.__index = 0
        return self
        
    def next(self):
        """ return current iterator data, and increment iterator position """
        if self.__index>=len(self):
            raise StopIteration
            
        self.__index = self.__index +1
        return self[self.__index-1]
        
