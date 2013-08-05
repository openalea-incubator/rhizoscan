""" Data structures that relates run-time objects to storage files

Data classes
------------
 - Data:     base class of all other Data classes. 
             also provide simple IO interface using pickle.
 - Mapping:  a Data subclass that provide a dictionary-like functionality.
 - Sequence: a Data subclass that provide a list-like functionality.
 
TODO:
-----
    - automatically set Data.__file attribute to loaded Data 
        * same for subclasses 
        * same for contained data
        * what about contained data with already set data file ?
            + look into the 'Data mode' idea ('r','w','u') ?
    - replace Data.__file by a Data_file property, replacing the get&set methods
    - replace the _data_to_load/save by the set/get_state protocol
    - look intop the __del__ deconstructor and the ability to automate saving
        * call to del seems not garantied...
"""
#from openalea import * ## should be optional
from rhizoscan.tool import static_or_instance_method, _property
from pprint import pprint
from copy   import copy as _copy

_PICKLE_PROTOCOL_ = -1 # by default all pickle saving (dump) are done with the latest protocol

from rhizoscan.workflow.openalea import aleanode as _aleanode


class Data(object):
    """ 
    Basic Data class: provide simple file IO interface using pickle
    
    The Data class has two aims:
      1) It can be used directly to relate some data to a file, and use the
         Data saving and loading interface (pickle based).
         See the constructor documentation
      2) It can be used as a superclass in order to get the saving and loading
         ability of `Data` (see below), and easily provide special behavior for 
         objects being saved or loaded by container Data (`Mapping`, `Sequence`)
         by overriding the `_data_to_save_` and `_data_to_load_` method.
         
    **Note for subclassing**:
    
        As stated, one of the main goal of the Data class is to be subclassed. 
        Subclasses can either keep or override the save, load, _data_to_save_ 
        and _data_to_load methods.
        
        - The Data save and load methods are simple interface with the pickle 
          modules. The save method save the objects (more precisely what is 
          returned by the _data_to_save_ method). And the load method loads it 
          (more precisely what the loaded object _data_to_load_ method returns).
          
        - Save and load are primarily supposed to be called statically. Thus
          when used as instance methods, it should be asserted that their
          behavior are suitable (especialy load) or require overriding. 
          * See save and load documentation for details *
          
        - The container Data class do not call the Data's save and load method. 
          So they can be overriden with no restriction.
          
        - However they call the _data_to_save_ and _data_to_load_ method, which
          are also called by the default implementation of save and load.
          
        - By default, the _data_to_save_ and _data_to_load_ methods simply 
          return the calling instance (i.e. self).
          
        - However, if it is wanted to do some processing prior to saving and/or
          post loading, then it can be done by overriding the _data_to_save_ and 
          _data_to_load_ methods.
          
        - The only constraint is to keep the same signature working: they should 
          be callable with only self as input, and return the object to be saved
          and the loaded object, respectively
          In both cases the returned object can be different from the calling
          one. however the saved object (return by _data_to_save_) should have a
          _data_to_load_ method if you want it to be called at loading.
          
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
    
    def __init__(self, filename=None):
        """                      
        Create an empty Data object that can be used to load the file filename.
        """
        self.__file = filename
        
    def set_data_file(self, filename):
        """
        Set the file of this data for saving and loading
        """
        self.__file = filename
    def get_data_file(self):
        """
        return the file of this data for saving and loading
        """
        return getattr(self,'_Data__file',None)
    
    @static_or_instance_method                           
    def save(data, filename=None, protocol=None):
        """ 
        Save input `data` to file `filename` using pickle 
                                                     
        This method can either be call as a:
          1. static   method:  Data.save(non_Data, filename,      protocol=None)
          2. static   method:  Data.save(Data_Obj, filename,      protocol=None)
          3. instance method:  someDataObject.save(filename=None, protocol=None)
        
        In case 2 & 3, save the value returned by the `_data_to_save_` method.
       
        :Inputs:
          - filename
            - In the 1st case, `filename` argument is mandatory     
            - In the 2nd and 3rd case, the instance Data file attribute is used
              if `filename` is not given. In this case, `filename` overwrite the
              Data file attribute of the Data object.
        
          - protocol
              the pickle protocol to use for saving. 
              If not given (None) use this module `_PICKLE_PROTOCOL_` value.
        
        :Outputs:
            Return an empty Data object that can be use to load the saved file
        """
        if getattr(data,'_mode','w')=='r':
            raise IOError("This Data object is read only")
        
        from cPickle import dump
        import os, shutil
        
        if filename is None:# and hasattr(data,'get_data_file'):
            filename = data.get_data_file()

        if hasattr(data,'_data_to_save_'):  ##? bug with Data being = to None...
            data = data._data_to_save_()

        if protocol is None: 
            protocol = _PICKLE_PROTOCOL_
        
        filetmp = filename + '~pickle_tmp'
        
        f = None
        try:
            d = os.path.dirname(filetmp)
            if len(d) and not os.path.exists(d):
                os.makedirs(d)
            f = file(filetmp,'wb')
            dump(data,f,protocol)
            f.close()
            shutil.move(filetmp,filename)
        finally:
            if f is not None:
                f.close()
        
        return Data(filename)
        
    @static_or_instance_method
    def load(filename_or_data, merge=True):
        """ 
        Load the pickled data from file 
                                                       
        This method can be used as 
          1. a static method with a file name:   Data.load(filename)
          2. a static method with a Data object: Data.load(some_Data_obj,merge=True)
          3. an instance method:                 some_data_obj.load(merge=True)
        
        If the loaded object has the _data_to_load_ method (such as Data objects)
        then it automatically calls this method and return its output.
        
        If case 1, this method return the loaded data.
        
        In cases 2 & 3, if the loaded data is a Data object and if merge is True
        then it merges the loaded data into the Data instance:
        
          - it copies all its attributes (found in its `__dict__`) overwriting
            existing attributes with same name 
          - it changes the instance `__class__` attribute with the loaded one
          
        If the output is a Data object, then its data file attribute is set to 
        the loaded file
          
        **Note for subclassing**:
        
            If the merging behavior is not suitable, it might be necessary to 
            override this methods. However the overriding method can call the 
            static `Data.load` method (case 2) with merge=False.
        """
        from cPickle import load
        
        data = filename_or_data  # for readibility
        
        if hasattr(data,'get_data_file'):
            fname = data.get_data_file()
        else:
            fname = data
        
        f = file(fname,'rb')
        d = load(f)
        f.close()
        
        if hasattr(d,'_data_to_load_') and hasattr(d._data_to_load_,'__call__'):
            d = d._data_to_load_()
            
        if merge and isinstance(data,Data) and isinstance(d,Data):
            data.__dict__.update(d.__dict__)
            data.__class__ = d.__class__
        else:
            data = d
            
        if isinstance(data,Data):
            data.set_data_file(fname)
        
        return data
       
    def _data_to_save_(self):
        """ 
        Prior processing of saved data. By default return it-self
        
        Note for subclassing:
        ---------------------
        Overriding this method can serves several purpose:
          - First this is where useful pre-saving cleaning can be done, 
            such as deleting dynamic data.
          - Sometime, the data object cannot by pickled. Overriding this method
            as well as _data_to_load_ enable to make a workaround.

        The return value should be savable by pickle, 
        typically its class definition and all its attributes (such as the load
        method) must be hard coded. 
        """
        return self
        
    def _data_to_load_(self):
        """
        Postprocessing of load. By default return it-self.

        Note for subclassing:
        ---------------------
        Overriding this method (and _data_to_save_) can serves several purposes:
          - First data that were deleted when saving can ne reloaded / computed.
          - Sometime, the data object cannot by pickled. Overriding this method
            as well as _data_to_save_ enable to make a workaround.
        """
        return self
        
        
    def loader(self, attribute=None):
        """
        Return an empty Data object which can be used to load the data from file
        
        `attribute` can be a name (string) or a list of names of attributs that
        the loader will keep.
        
        *** if this object has no associated file, it won't be able to load ***
        """
        loader = Data(filename=self.get_data_file())
        loader._mode = 'r'
        if attribute:
            if isinstance(attribute,basestring):
                attribute = [attribute]
            for attr in attribute:
                setattr(loader,attr,getattr(self,attr))
        return loader
        
    def __str__(self):
        cls = self.__class__
        return cls.__module__ +'.'+ cls.__name__ + ' with file: ' + str(self.get_data_file())
            
@_aleanode('data_loader')
def save_data(data, filename):
    """
    Use Data class to save input 'data'. Return a Data instance.
    """
    return Data.save(data=data,filename=filename)

class DataWrapper(Data):
    """ Class that wrap anything inside a Data """
    def __init__(self, data, filename):
        """
        Create a DataWrapper object that bind file filename to the input data
        Use save and (later) load to save and reload the data
        """
        Data.__init__(self,filename)
        self.__data = data
        
    def _data_to_load_(self):
        """ Postprocessing of load. return the wrapped data """
        return self.__data

        

# a simple structure data
# -----------------------
class Mapping(Data):
    """
    Class Mapping provide a structure Data type with dynamique field names
    
    Example:
    --------
        m1 = Mapping()
        m1.q1 = 6
        m1.q2 = 9
        m1.ans = m1.q1 * m1.q2
        print repr(m1)            # q1:6  q2:9  ans:54
        
        m2 = Mapping(the_question='6x9', ans=42)
        m1.merge(m2,False)
        print m1.ans              # 54
        m1.merge(m2)
        print m1.ans              # 42
         
    Mapping object can be treated as dictionaries:
    ----------------------------------------------
        * the double-star operator can be used: 
            some_function(**myMappingObject)
            
        * implement iterator. However, iteration return tuple (field,value):
            m = Mapping(a=42,z=0,t=None)
            for field,value in m: print field, 'has value', value
         
    Warning:
    --------
        Mapping are Data objects and have the fields _Data__file and _Data__data 
        reserved. Overwriting them will induce failure of Data functionalities.
    """
    ##TODO Mapping:
    ##  - save doc (for now it's the Data.save doc)
    ##  - saving using pyyaml (by default, keeping pickle saving in option)
    ##  - what about saving to intermediate file when in a container ?
    #      > useful ?
    #      > then needs some 'update' flag ?
    def __init__(self, load_file=None, **kwds):
        """
        Create a Mapping object containing all keyword arguments as fields
        
        If Data_file is given, it loads the file and save it as the Data file 
        attribute for later saving/loading. 
        See the documentation of the load and save methods.
        """
        if load_file is not None: self.load(load_file)
        self.__dict__.update(kwds)
        
    def fields(self):
        """ return a list of all field names """
        return self.__dict__.keys()
    def keys(self):
        """ Same as fields - implement keys s.t. Mapping can be treated as dict """
        return self.__dict__.keys()
    def values(self):
        """ return a list of all field values """
        return self.__dict__.values()


    def set(self,field, value=None):
        """ set a field, with value, or with None if not provided """
        return self.__dict__.setdefault(field,value)
    def get(self,field,default=None):
        """ return a field value, or default if field doesn't exist """
        return self.__dict__.get(field,default)
    def pop(self,field,default=None):
        """ remove field and return its value or default if it doesn't exist' """ 
        return self.__dict__.pop(field,default)


    def iteritems(self):
        """ return an iterator over the (field, value) items of Mapping object """
        return self.__dict__.iteritems()
    def itervalues(self):
        """ return an iterator over the values of Mapping object """
        return self.__dict__.itervalues()
    def iterfields(self):
        """ return an iterator over the fields of Mapping object """
        return self.__dict__.iterkeys()

    def has_field(self, fieldname):
        """ return True if field exists, False otherwise """
        return self.__dict__.has_key(fieldname)


    def merge(self, other, overwrite=True):
        """ ## todo: doc """
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
        
    def _data_to_save_(self):
        """
        Return a copy of it-self and call recursively _data_to_save_ on all 
        contained objects that have the _data_to_save_ method, such as Data objects. 
        
        Note: This is what is really save by the 'save' method.
        """
        s = self.__copy__()
        s.clear_temporary_attribute()
        
        d = s.__dict__
        for field,value in d.iteritems():
            if hasattr(value,'_data_to_save_') and hasattr(value._data_to_save_,'__call__'):
                #print field,   
                #try: print value.get_data_file()  ## debug
                #except: print 'no data file'  
                d[field] = value._data_to_save_()
            
        return s

    def _data_to_load_(self):
        """
        Return it-self after calling _data_to_call_ on all contained fields that
        has this method, such as Data objects.
        """
        for field,value in self.iteritems():
            if hasattr(value,'_data_to_load_') and hasattr(value._data_to_load_,'__call__'):
                try:
                    self[field] = value._data_to_load_()
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
              (i.e. pickle files) and can be passed as input of the merge method
              
          - overwrite: 
              If True, loaded fields overwrite existing one. 
              Otherwise, it don't. (same as for the merge method)
                      
        :Output:
            return it-self
        """
        loaded = Data.load(self if filename is None else filename)
        self.merge(loaded, overwrite=overwrite)
            
        return self
            
            
    # accessors
    # ---------
    def __getitem__(self,item):
        """ allow access using [] as for python dictionaries """
        return getattr(self,item)##.__dict__.__getitem__(item)
    def __setitem__(self,item,value):
        """ allow setting fields using [] as for python dictionaries """
        return setattr(self,item,value) ##self.__dict__.__setitem__(item,value)
    def __len__(self):
        return self.__dict__.__len__()
        
    # gives string representation of structure content (used by the print function)
    # ------------------------------------------------
    def __repr__(self):
        return self.multilines_str()##self.__dict__.__repr__()
    # nice printing
    def __str__(self):
        #from pprint import pformat
        #return pformat(self.__dict__)
        #return "length %d %s object with associated file: %s" % (len(self), self.__class__.__name__, self.get_data_file()) # self.multilines_str()
        cls_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return cls_name+":"+str(self.__dict__)
    def display(self, tab=0, max_width=80, avoid_obj_id=None):
        """ same as print, but give access to arguments  
        see multilines_str for details """
        print self.multilines_str(tab=tab, max_width=max_width, avoid_obj_id=avoid_obj_id)
    def multilines_str(self, tab=0, max_width=80, avoid_obj_id=None, self_name=''):
        """ multilines string representation, potentially hierarchical 
        
        tab: number of tab to start all lines with                                       
        max_width: maximum number of character per lines
        
        (optional)
        avoid_object_id: either None, or a dictionary of python object id as key and value to print
        self_name: name to be print for current object, if referenced by subfield
        """
        if avoid_obj_id is None:
            avoid_obj_id = dict()
        avoid_obj_id[id(self)] = '&' + self_name
        
        string = ''
        fields = sorted([f for f in self.fields() if not f.startswith('_')])
        for field in fields:#self.iteritems():
            value = self[field]
            name  = '    '*tab + str(field) + ': '
            shift = ' '*len(name)
            
            if id(value) in avoid_obj_id:
                string += name + '<%s: %s>\n' % (value.__class__.__name__,avoid_obj_id[id(value)])  
            elif hasattr(value, 'multilines_str'):
                string += name + '%s\n' % value.__class__.__name__
                if len(self_name): subname = self_name + '.' + field
                else:              subname = field   
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

@_aleanode(auto_caption=1)
def get_field(data={}, field='metadata', default=None):
    """
    Return data[field], or default if it does not exist.
                                                                                     
    Either use the data.get method, if it exist, or getattr on data 
    """
    if hasattr(data,'get'): return data.get(field, default)
    else:                   return getattr(data,field,default)
        
# Data that manages sequence
class Sequence(Data):
    """
    Simple read-write interface for sequence of data
    
    :TODO: 
      - make it a DataConatainer
          > associate it to a folder
      - doc
          > setting might save. 
          > setting Data obj change its Data file attribute
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
        output should be a "printf" string containing 1 and only 1 '%d' element
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
        Save 'item' to file 'filename'. By default use the Data.save method
        
        This is called by the __setitem__ method and can be overrided by subclasses
        """
        Data.save(item,filename)
                
    def _data_to_save_(self):
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
        if isinstance(data,Data): data.set_data_file(filename)
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
        
