""" Data structures that relates run-time objects to storage files

Data classes
------------
 - Data:     base class of all other Data classes. 
             also provide simple IO interface using pickle.
 - Mapping:  a Data subclass that provide a dictionary-like functionality.
 - Sequence: a Data subclass that provide a list-like functionality.
 
TODO:
-----
    - automatically set Data.__entry attribute to loaded Data 
        * same for subclasses 
        * same for contained data
        * what about contained data with already set data file ?
            + look into the 'Data mode' idea ('r','w','u') ?
    - replace Data.__entry by a storage_entry property, replacing the get&set methods
    - replace the _(un)serialize_ by the set/get_state protocol
    - look into the __del__ deconstructor and the ability to automate saving
        * call to del is not garantied by carbage collector
"""
__icon__    = 'datastructure.png'   # icon of openalea package    
                                             
#from openalea import * ## should be optional
from pprint import pprint
from copy   import copy as _copy

from .tool     import static_or_instance_method, _property 
from .workflow import node as _node # to declare workflow nodes
from .storage  import create_entry as _create_entry
from .storage  import MapStorage  as _MapStorage

class Data(object):
    """ 
    Basic Data class: provide automated storage IO operation
    
    The Data class has two aims:
      1) It can be used to relate some data to a storage entry (eg. file), and 
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
    
    def __init__(self, storage_entry=None, serializer=None):
        """                      
        Create an empty Data object that can be related to `storage_entry`
        
        `storage_entry` can be a filename, a valid url string or a 
        `datastructure.StorageEntry` object.
        
        See `datastructure.storage` documentation. 
        """
        self.set_storage_entry(storage_entry)
        self.set_serializer(serializer)
        
    @static_or_instance_method
    def set_storage_entry(self, storage_entry):
        """
        set the storage entry of thei object
        
        `storage_entry` can be either:
          - a filename, url string or a StorageEntry
          - None, to remove storage link
          - -1 for (attempted) removale of the storage entry (i.e file)
          
        See `datastructure.storage` documentation. 
        """
        ## check if entry already set or taken: what to do?  
        if storage_entry is None:
            pass
        elif storage_entry==-1:
            old_entry = self.get_storage_entry()
            if old_entry:
                old_entry.remove()
            storage_entry = None
        if storage_entry is not None:
            storage_entry = _create_entry(storage_entry) 
            
        self.__storage_entry__ = storage_entry
        
    @static_or_instance_method
    def get_storage_entry(self):
        """
        return the file of this data for saving and loading
        """
        return getattr(self,'__storage_entry__',None)
    
    @staticmethod
    def has_IO_API(obj):
        """
        Test if given `obj` has the Data I/O API:
        if it has the attributes `set_storage_entry`, `get_storage_entry`,
        `dump` and `load`.
        """
        return all(map(hasattr,[obj]*4, ['set_storage_entry','get_storage_entry','dump','load']))
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
    def dump(data, entry=None):
        """ 
        Save input `data` to `entry` (which can be a file name) 

        This method can either be called as a:
          1. static   method:  Data.dump(non_Data, entry,      protocol=None)
          2. static   method:  Data.dump(Data_Obj, entry,      protocol=None)
          3. instance method:  someDataObject.dump(entry=None, protocol=None)
        
        If the 1st argument has the "store" API, this function saves the value
        returned by its `__store__` method.
       
        :Inputs:
          - `entry`
            Can be either a `StorageEntry` object, a `url` or a path to a file.
            If it is not a StorageEntry, the value of `entry` is passed to 
            `storage.create_entry()` (see create_entry doc)
            
            In the case (1), `entry` argument is mandatory. 
            In the case (2) and (3),  if `entry` is not given (None) the 
            instance Data `storage_entry` attribute is used. In this case, 
            `entry` becomes the `storage_entry` attribute of `data`.
        
        :Outputs:
            Return an empty Data object that can be use to load the stored data
        """
        if 'w' not in getattr(data,'_mode','w'):
            raise IOError("This Data object is not writable")
        
        io_api = Data.has_IO_API(data)
        if entry is None:
            if io_api:
                entry = data.get_storage_entry()
            else:
                raise TypeError("argument 'entry' should be given when 'data' does not have an associated storage entry")

        if io_api:
            data.set_storage_entry(entry)
            entry = data.get_storage_entry()
        else:
            entry = _create_entry(entry)
            
        if Data.has_store_API(data): to_store = data.__store__()
        else:                        to_store = data
        
        serializer = Data.get_serializer(data)
        entry.save(to_store, serializer=serializer)
        
        if io_api and hasattr(data,'loader'): 
            return data.loader()
        else:
            return Data(entry).loader()  ## if lookup in 'Data'base => equiv to return self
        
    @static_or_instance_method
    def load(url_or_data, merge=False):
        """ 
        Load the data serialized at `url_or_data` 

        This method can be used as
          1. a static method with url:       Data.load(filename)
          2. a static method with IO object: Data.load(IO_obj,merge=True) (*)
          3. an instance method:             data_obj.load(merge=True)
        
        (*) `IO_object`must have the "IO" API - see Data.has_IO_API()
        
        If the loaded object has the "store" api (such as Data objects) then 
        this function return the output of the `__restore__` method.
        
        In cases 2 & 3, `merge` can be False, 'dict' or True. If merge is not 
        False, the loaded data is merged into the Data object:
        
          - it copies all its attributes (found in its `__dict__`) overwriting
            existing attributes with same name 
          - if the caller and loaded data are Data objects and if `merge` = True
            it changes  the instance `__class__` attribute with the loaded one.
          
        If the output is a Data object, then its storage entry attribute is set
        to the loaded url
          
        :Subclassing:
        
            If the merging behavior is not suitable, it might be necessary to 
            override this methods. However the overriding method can call the 
            static `Data.load` method (case 2) with merge=False.
        """
        data = url_or_data  # for readibility
        
        if Data.has_IO_API(data):
            entry = data.get_storage_entry()
        else:
            entry = _create_entry(data)
            
        ##serializer = Data.get_serializer(data)
        
        d = entry.load()##serializer=serializer)
        
        if Data.has_store_API(d):
            d = d.__restore__()
            
        if merge is not False and hasattr(data,'__dict__'):
            data.__dict__.update(d.__dict__)
            if merge==True and isinstance(d,Data) and isinstance(data,Data): 
                data.__class__ = d.__class__
        else:
            data = d
            
        if Data.has_IO_API(data):  ## if hasattr(data,'__dict__'): ??
            data.set_storage_entry(entry)
        
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
          - self.__restore__ if it has no storage entry set
          - it's loader if it does
        """
        if self.get_storage_entry() is None:
            return self.__store__()
        else:
            return self.loader()
        
    def loader(self, attribute=None):
        """
        Return an empty object which can be used to load the stored object
        
        `attribute` can be a name (string) or a list of names of attributs that
        the loader will keep.
        
        *** if the object has no associated entry, it won't be able to load ***
        """
        loader = Data(storage_entry=self.get_storage_entry(),serializer=self.get_serializer())
        loader._mode = 'r:loader'
        loader.__class__ = self.__class__
        if attribute:
            if isinstance(attribute,basestring):
                attribute = [attribute]
            for attr in attribute:
                setattr(loader,attr,getattr(self,attr))
        return loader
        
    def __str__(self):
        cls = self.__class__
        return cls.__module__ +'.'+ cls.__name__ + ' with file: ' + str(self.get_storage_entry())
            

            
@_node('data_loader')
def save_data(data, filename):
    """
    Use Data class to save input 'data'. Return a Data instance.
    """
    return Data.dump(data=data,filename=filename)

##class DataWrapper(Data):
##    """ Class that wrap anything inside a Data """
##    def __init__(self, data, filename):
##        """
##        Create a DataWrapper object that bind file filename to the input data
##        Use save and (later) load to save and reload the data
##        """
##        Data.__init__(self,filename)
##        self.__data = data
##        
##    def __restore__(self):
##        """ Postprocessing of load. return the wrapped data """
##        return self.__data

        

# a simple structure data
# -----------------------
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
     1. Mapping are Data objects and have the attribut `_Data__entry` 
        reserved. Overwriting it will induce failure of IO functionalities.
     2. Mapping can be use as a container through the `set_container` method,
        then using the function `set` and  `get` (see docs for details)
    """
    ##TODO Mapping:
    ##  - dump doc (for now it's the Data.dump doc)
    ##  - saving using pyyaml (by default, keeping pickle saving in option)
    ##  - what about saving to intermediate file when in a container ?
    #      > useful ?
    #      > then needs some 'update' flag ?
    def __init__(self, load_file=None, **kwds):
        """
        Create a Mapping object containing all keyword arguments as keys
        
        If `load_file` is given, it loads the file and stores it as the Data 
        file attribute for later io operation (see load and save documentation).
        """
        if load_file is not None: self.load(load_file)
        self.__dict__.update(kwds)
        
    def keys(self):
        """ Return the list of this Mapping keys (same as dict) """
        return self.__dict__.keys()
    def values(self):
        """ return the list of this Mapping values (same as dict) """
        return self.__dict__.values()


    def set_storage(self, storage):
        """
        Attach a storage for automatized I/O of contained attributs objects.
        
        :Inputs:
          - `storage`: 
                Either a MapStorage object, a valid input of its constructor
                (e.g. format-type string), or `None` to not use a storage
        
        :See also: storage.MapStorage
        """
        if not isinstance(storage, _MapStorage) and storage is not None:
            storage = _MapStorage(storage)
        self.__storage__ = storage
        
    def set(self,key, value=None, store=False):
        """ 
        Set the `key`,`value` pairs (use `value=None` if not provided)
        If this object has a `storage` set and store is True: store value to it 
        
        Note: this method simply calls `__setitem__`
        """
        return self.__setitem__(key,value,store=store)
    def setdefault(self,key, value=None):
        """ 
        M.setdefault(key,value) = M.get(key,value), with M[key]=value if key not in M
        """
        return self.__dict__.setdefault(key,value)
        
    def get(self,key,default=None):
        """ return the value related to `key`, or `default` if `key` doesn't exist
        
        Note: this method simply calls `__getitem__`
        """
        return self.__getitem__(key,default)
    def pop(self,key,default=None):
        """ remove `key` and return its value or default if it doesn't exist' """ 
        return self.__dict__.pop(key,default)


    def iteritems(self):
        """ return an iterator over the (key, value) items of Mapping object """
        return self.__dict__.iteritems()
    def itervalues(self):
        """ return an iterator over the values of Mapping object """
        return self.__dict__.itervalues()
    def iterkeys(self):
        """ return an iterator over the keys of this Mapping object """
        return self.__dict__.iterkeys()

    def has_key(self, key):
        """ return True if `key` exists, False otherwise """
        return self.__dict__.has_key(key)


    def update(self, other, overwrite=True):
        """ add `other` content into this Mapping object
        ## todo:
            - other a dict-like object (use update(**other) ?)
            - add **kargs arguments
            - remove the overwrite argument, and add a setdefault/union method
        """
        if isinstance(other,Mapping):   ## if other hasattr '__dict__' ??
            other = other.__dict__
        if overwrite:
            self.__dict__.update(other)
        else:
            for k,v in other.items():
                if k not in self.__dict__:
                    self.__dict__[k] = v

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
    
    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new._tmp_attr = self.temporary_attribute.copy()
        return new
        
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
                      
        :Output:
            return it-self
        """
        loaded = Data.load(self if filename is None else filename)
        self.update(loaded, overwrite=overwrite)
            
        return self
            
            
    # accessors
    # ---------
    def __getitem__(self,item, *args):
        """ allow access using [] as for python dictionaries """
        if len(args): return self.__dict__.get(item,*args)
        else:         return self.__dict__[item]
    def __setitem__(self,item,value, store=False):
        """ allow setting item using [] as for python dictionaries """
        self.__dict__[item] = value
        if store:
            self.__storage__.set_data(item, value)
        return self
    def __len__(self):
        return self.__dict__.__len__()
        
    # gives string representation of structure content (used by the print function)
    # ------------------------------------------------
    def __repr__(self):
        return self.multilines_str()[:-1]##self.__dict__.__repr__()
    # nice printing
    def __str__(self):
        #from pprint import pformat
        #return pformat(self.__dict__)
        #return "length %d %s object with associated file: %s" % (len(self), self.__class__.__name__, self.get_storage_entry()) # self.multilines_str()
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
        return string

    # allow numerical comparison
    # --------------------------
    def __cmp__(s1,s2):
        """ x.__cmp__(y) <==> cmp(x,y) """ 
        return s1.__dict__.__cmp__(getattr(s2,'__dict__',s2))
    def __eq__(s1,s2):
        """ x.__eq__(y) <==> x==y """
        return s1.__dict__.__eq__(getattr(s2,'__dict__',s2))
    def __lt__(s1,s2):
        """ x.__lt__(y) <==> x<y """
        return s1.__dict__.__lt__(getattr(s2,'__dict__',s2))
    def __le__(s1,s2):
        """ x.__le__(y) <==> x<=y """
        return s1.__dict__.__le__(getattr(s2,'__dict__',s2))
    def __gt__(s1,s2):
        """ x.__gt__(y) <==> x>y """
        return s1.__dict__.__gt__(getattr(s2,'__dict__',s2))
    def __ge__(s1,s2):
        """ x.__ge__(y) <==> x>=y """
        return s1.__dict__.__ge__(getattr(s2,'__dict__',s2))
        
    # make Mapping a valid iterator
    # -----------------------------
    def __iter__(self):
        return self.iteritems()#__dict__.__iter__() ##?
        
    def __contains__(self, key):
        return key in self.__dict__

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
        if isinstance(data,Data): data.set_storage_entry(filename)
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
        
