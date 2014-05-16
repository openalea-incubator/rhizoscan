__icon__    = 'database.png'   # icon of openalea package


import numpy as _np

from rhizoscan.workflow import node as _node # to declare workflow nodesfrom rhizoscan.workflow  import node as _node # decorator to declare workflow nodes

from rhizoscan.datastructure import Mapping as _Mapping
from rhizoscan.datastructure import Data    as _Data

from . import _print_state, _print_error, _param_eval 

class Dataset(list, _Mapping):
    """ 
    Data subclasses which implement a list of dataset items 
    
    Dataset are traditional `list` with additional methods:
      - from the Data class: set/get_file, etc...
      - specialized to access "dataset" item:
        `kget`, `ksort`, `kindex`, `iteritems`
        
    Dataset items are expected to be object with a '__key__' attribute
    that is unique inside the dataset.
    
    A Dataset can be generated from a suitable '.ini' file using `make_dataset`.
    """
    def __init__(self, item_list=None, key=None):
        """ Create a Dataset object """
        if item_list:
            self.extend(item_list)
        self.__key__ = key
            
    def keys(self):
        """ return the list of item keys """
        return [getattr(item,'__key__',None) for item in self]
        
    def kindex(self, key, default=None):
        """ 
        Return the index of the (1st) item with __key__ attribute equal to `key` 
        Return 'default' if 'key' is not found
        """
        try:
            return self.keys().index(key)
        except ValueError:
            return default
        
    def kget(self, key, default=None):
        """ Return the (1st) item with attribute '__key__' equal to given `key` """
        index = self.index(key)
        if index: return self[index]
        else:     return default
        
    def ksort(self):
        """ **inplace*$ sort by item's '__key__' attribute """
        self.sort(key=lambda item: (item.get('__key__'),item))
        return self
        
    def __getslice__(self, i, j):
        """ Return the sub-dataset from item `i` to `j` """
        return Dataset(list.__getslice__(self,i,j))

    def iteritems(self):
        """ Return an iterator on (item.__key__,item) for all Dataset items """
        return ((item.__key__, item) for item in self)

    def itermembers(self):
        """ 
        Return an iterator on (membder.__key__,member) for all Dataset members
        
        For standard Dataset, this is the same as `iteritems`. For grouped
        Dataset however, it iterates over the members of the groups.
        
        See also: `group_by`, `iteritems`
        """
        from itertools import chain
        ##chain(item if isinstance(item,Dataset) else ((item.__key__,item),) for item in self)
        if self.is_grouped():
            return chain(*(item.iteritems() for item in self))
        else:
            return self.iteritems()
        

    def __repr__(self):
        return list.__repr__(self)
    def __str__(self):
        return list.__str__(self)

    def group_by(self, key, key_base=None):
        """ group item having the same value of their `key` attribute 
        
        :Outputs:
            Return a Dataset of Dataset objects.
            If `keys` is a string: the new Dataset items __key__ attribute are 
            equal the (common) values of its items `key` attributes.
            If `key` is a list, its it equal to the tuple of all the keys values
        
        :Inputs:
          - `key`:
              The item attribute(s) that are used to cluster Dataset items.
              It can contain '.' (ex: 'attr.subattr') meaning to process 
              sub-attribute.
              It can be a list of keys, meaning to cluster by the (unique) set 
              of those keys value. The output depends on given `flat` value.
          - `key_base`:
              Common suffix to append to all `key`
              Eg: (key=['b','c'], key_base='a') is the same as key=['a.b','a.c']
              
        See also: `is_grouped`, `itermembers`
        """
        if isinstance(key, basestring):
            key = [key]
            
        if key_base:
            key = [key_base+'.'+k for k in key]
        
        key = [k.split('.') for k in key]
        
        group = {}
        for item in self:
            if len(key)==1: key_value = _mget(item,key[0])
            else:           key_value = tuple(_mget(item,k) for k in key)
            group.setdefault(key_value,Dataset(key=key_value)).append(item)
            
        return Dataset(group.values())

    def is_grouped(self):
        """ Return True if this dataset contains (only) Datasets """
        return all(isinstance(item,Dataset) for item in self)

    def get_column(self, name, default=None):
        """ return the list of item's `name` attribute, or default """
        return [_mget(item,name.split('.'),default) for item in self]

    def __change_dir__(self, old_dir, new_dir, load=False, verbose=False, _base=''):
        """
        Change url of all content stored in `old_dir` to `new_dir`

        This is used to update file url of **already moved** content 
        It does not actually move the files.
        
        :Inputs:
          - `old_dir`: the (start of) the path to be replaced   (*)
          - `new_dir`: the (start of) the path to replace it by (*)
          - `load`:
              This value is passed recusively to each items `__change_dir__` fct
              If =2, call dump after applying changes
              If =3, do not keep loaded content in memory
          - `verbose`:
              If True, print a line for each changed made
              
        (*) always include the directory separator '\' or '/'
        """
        for k,v in self.iteritems():
            base = _base+'.'+k if len(_base) else k
            if hasattr(v,'__change_dir__'):
                if load>=3 and v.get_file():
                    v = v.loader().load()
                v.__change_dir__(old_dir=old_dir, new_dir=new_dir, load=load, verbose=verbose, _base=base)
                if load>=2 and v.get_file():
                    v.dump()


def _mget(item,key, default=None):
    """ recursive getattr for given list of attributes `key`"""
    value = reduce(lambda x,f: getattr(x,f,'__MISSING_ATTR'),[item]+key)
    if value is '__MISSING_ATTR': 
        return default
    else: 
        return value
        
@_node('image_list', 'invalid_file', 'output_directory', OA_hide=['verbose'])
def make_dataset(ini_file, base_dir=None, data_dir=None, out_dir=None, out_suffix='_', verbose=False):
    """
    Return a list of dataset item following parsing rules found in `ini_file`
    
    :Inputs:
      - `ini_file`: 
          file with ini-formated content indicating the dataset to be loaded
          See the 'ini format' section for details
      - `base_dir`:
          Starting directories for inputs and outputs (see 'directories' below)
          If not given, use the directory of given `ini_file`
      - `data_dir`:
          Directories to look for data inputs         (see 'ini format' below)
          If not given, use the value in the ini file, or `base_dir`
          If it is not an absolute path, preppend it with `base_dir`
      - `out_dir`:
          Directories to set output into              (see 'directories' below)
          If not given, use the value in the ini file, or `base_dir`
          If it is not an absolute path, preppend it with `base_dir`
      - `out_suffix':
          String to append to output files            (see 'directories' below)
      - `verbose`:
          If >0, print some message on loaded dataset
    
    :Outputs:
      - A list of `Mapping` object, one for each file found with suitable output
        files configure (nothing is saved). Each contains the attributes:
          - 'filename': the found file name
          - 'metadata': constructed from the ini file
          - '__key__':  an id key made from 'filename' with `data_dir` removed 
      - The list of files found but which could not be parsed
      - The base output directory to all item (see 'directories' below)
    
    :directories:
        All output files and directories are set following the values of given 
        `base_dir`, `data_dir`, `out_dir` and `out_suffix`.
        
        The associated file of output `Mapping` items are set to:
          "[out_dir]/[item-end].namespace"
           
        The output items have their MapStorage set (see Mapping doc) to:
          "[out_dir]/[item-end][out_suffix]{}"
          
        Where `item-end` is the remaining part of the filename of found items 
        after removing `data_dir` from the start and the file extension.
        
        See datastruture.Data and Mapping documentations for details on `Data` 
        associated file and `Mapping` MapStorage
        
    :ini format:
        ##todo
    """
    import os
    from os.path import join as pjoin
    from os.path import splitext, dirname, exists
    import re
    from time import strptime
    from glob import glob
    
    from rhizoscan.tool.path import abspath
    
    if not exists(ini_file):
        raise TypeError('input "ini_file" does not exist')
    
    # load content of ini file
    ini = _load_ini_file(ini_file)
    
    if verbose>2:
        print 'loaded ini:'
        print ini.multilines_str(tab=1)
        
    # directory variable
    if base_dir is None:
        base_dir = dirname(abspath(ini_file))
        
    if data_dir is None: 
        data_dir = ini['PARSING'].get('data_dir')
        if not data_dir: 
            data_dir = base_dir
    data_dir = abspath(data_dir, base_dir)
        
    if out_dir is None: 
        out_dir = ini['PARSING'].get('out_dir')
        if not out_dir: 
            out_dir = base_dir
    out_dir = abspath(out_dir, base_dir)
    
    # find all files that fit pattern given in ini_file
    # -------------------------------------------------
    # list all suitable files
    file_pattern = ini['PARSING']['pattern']
    file_pattern = file_pattern.replace('\\','/')      # for windows
    file_pattern = re.split('[\[\]]',file_pattern)
    
    glob_pattern = pjoin(data_dir,'*'.join(file_pattern[::2]))
    file_list = sorted(glob(glob_pattern))
    
    if verbose:
        print 'glob:', glob_pattern
        if verbose>1:
            print '   ' + '\n   '.join(file_list)
    
    # prepare metatata parser
    # -----------------------
    # meta data list and regular expression to parse file names
    group_re = dict(int='([0-9]*)', float='([-+]?[0-9]*\.?[0-9]+)')
    ##meta_parser = re.compile('(.*)'.join([fp.replace('*','.*') for fp in file_pattern[::2]]))
    meta_list = [m.split(':') for m in file_pattern[1::2]]
    meta_list = [m if len(m)>1 else m+['str'] for m in meta_list]
    meta_parser = file_pattern[:]
    meta_parser[1::2] = [group_re.get(mtype,'(.*)') for name,mtype in meta_list]
    meta_parser = re.compile(''.join(meta_parser))
    meta_list = [_Mapping(name=name,type=mtype) for name,mtype in meta_list]
    date_pattern = ini['PARSING'].get('date','') ## to remove?
    
    types = dict(int=int,float=float,str=str)
    for m in meta_list:
        if m.type=='date':   ## to remove? 
            m.eval = lambda s: strptime(s, date_pattern)
        elif m.type=="$":
            default = m.name+'_default'
            m.eval = lambda name: ini.get(name, default=default)
        else: 
            try: 
                m.eval = types[m.type]
            except KeyError:
                raise KeyError('unrecognized parsing type %s for field %s' % (m.type,m.name))
            
    default_meta = ini.get('metadata',{})
    for k,v in default_meta.iteritems():
        default_meta[k] = _param_eval(v)
        
    # if grouping
    if ini['PARSING'].has_key('group'):
        g = ini['PARSING']['group'] ## eval... value is a dict
        dlist = [dirname(fi) for fi in file_list]
        fenum = [int(di==dlist[i]) for i,di in enumerate(dlist[1:])]          # diff of dlist
        fenum = _np.array(reduce(lambda L,y: L+[(L[-1]+y)*y], [[0]]+fenum))   # ind of file in resp. dir.
        group = _np.zeros(fenum.max()+1,dtype='|S'+str(max([len(gi) for gi in g.itervalues()])))
        for start in sorted(g.keys()):
            group[start:] = [g[start]]*(len(group)-start)
        group = group[fenum]
        if verbose:
            print 'group:', g
            if verbose>1:
                print '   detected:', group
    else:
        group = None
    
    if verbose:
        print 'metadata:', meta_parser.pattern, 
        print '> ' + ', '.join((m.name+':'+m.type for m in meta_list))
        
    # get global variable
    global_attr = ini.get('global',{})
    
        
    # parse all image files, set metadata and remove invalid
    # ------------------------------------------------------
    img_list = Dataset()
    invalid  = []
    rm_len = len(data_dir)  ## imply images are in base_dir. is there a more general way
    for ind,f in enumerate(file_list):
        try:
            if rm_len>0: subf = f[rm_len+1:]
            else:        subf = f 
            subf = subf.replace('\\','/')   # for windows
            fkey = splitext(subf)[0]
            out_store = pjoin(out_dir, fkey) + out_suffix + '{}'
            out_file  = pjoin(out_dir, fkey) + '.namespace'
            meta_value = meta_parser.match(subf).groups()
            if verbose>1:
                print '   ' + str(meta_value) + ' from ' + subf + str(rm_len)
            meta = _Mapping(**default_meta)
            if group is not None:
                meta.update(ini.get(group[ind], default=[]))
            for i,value in enumerate(meta_value):
                field = meta_list[i].name
                value = meta_list[i].eval(value)
                if field=='$': 
                    meta.update(value) 
                else:
                    _add_multilevel_key_value(meta,field,value)
                
            ds_item = _Mapping(filename=f, metadata=meta, __key__=fkey, **global_attr)
            ds_item.__loader_attributes__ = ['filename','metadata']
            ds_item.set_map_storage(out_store)
            ds_item.set_file(out_file)
            img_list.append(ds_item)
        except Exception as e:
            invalid.append((type(e).__name__,e.message, f))
            
    return img_list, invalid, out_dir
    
def make_dataset_item(filename, metadata=None, base_dir=None, data_dir=None, out_dir=None):
    """ Create an item for dataset 
    
    filename: the item `filename` attribute
    metadata: the item `metadata` attribute - default: empty Mapping object
    base_dir: base directory of others - default: filename dir
    data_dir: directory of filename - default: base_dir 
    out_dir:  directory for output  - default: base_dir
    
    item data file has the following set:
     - attribute 'filename': the given value
     - attribute '__key__':  the filename with data_dir & extension removed
     - item file (used by dump&load):        out_dir/__key__+'.namespace'
     - map storage (for external attribute): out_dir/__key__+'_{}'
     
    returns the dataset item
    """
    import os
    file_dir, file_base = os.path.split(filename)
    file_base, file_ext = os.path.splitext(file_base)
    
    if metadata is None:  metadata = _Mapping()
    if base_dir is None:  base_dir = os.path.dirname(filename)
    if data_dir is None:  data_dir = base_dir
    if out_dir  is None:  out_dir  = base_dir
    
    file_base = filename[len(data_dir):].strip(os.sep)
    key = os.path.splitext(file_base)[0]
    
    item_file = os.path.join(out_dir,key)+'.namespace'
    map_store = os.path.join(out_dir,key)+'_{}'
    
    item = _Mapping(filename=filename, metadata=metadata, __key__=key)
    item.__loader_attributes__ = ['filename','metadata']
    item.set_map_storage(map_store)
    item.set_file(item_file)
    
    return item
    
    
    
def _add_multilevel_key_value(m, key,value): 
    """
    add entry to Mapping `m` from possible multilevel `key`
    return updated `mapping`
    :Example: 
        m = _add_multilevel_key_value(Mapping(),'a.b.c',42)
        print m.a.b.c==42
        # True
    """
    key = key.split('.')
    def assert_mapping(subm,k):
        if not subm.has_key(k) or not hasattr(subm[k],'iteritems'): 
            subm[k] = _Mapping()
        return subm[k]
    last = reduce(assert_mapping, key[:-1],m)
    last[key[-1]] = value
    return m
    
def _load_ini_file(ini_file):
    """
    return the ini file content as a hierarchy of Mapping objects
    
    can manage: 
     - multilevel key name
     - inheritance of *previous* section:
     
    Example::
    
        [section1]
        a=1
        [section2:section1]
        subsection.value=42
        
        returns a Mapping containing:
        section1
          a=1
        section2
          a=1
          subsection
            value=42
    """
    import ConfigParser as cfg
    ini = cfg.ConfigParser()
    ini.read(ini_file)
    m = _Mapping()
    for section in ini.sections():
        s,parent = (section+':').split(':')[:2]
        if len(parent):
            parent = m[parent]
        else:
            parent = {}
        m[s] = _Mapping(**parent)
        for k,v in ini.items(section): 
            _add_multilevel_key_value(m[s],k,_param_eval(v)) 
    return m 
   
    
@_node('filtered_db')
def filter(ds, key='', value='', metadata=True):
    """
    Return the subset of dataset `ds` that has `key` attribute equal to `value`
    
    :Input:
      - key:    (*)
        key to filter `ds` by. Can contains dot, eg. 'attr.subattr'
      - value:  (*)
        The value the attribute `key` must be equal to
      - metadata:
        if True, look for key in the 'metadata' attribute
        Same as filter(ds,'metadata.'+key,value, metadata=False) 
        
      (*) if `key` or `value` is empty, return the full (unfiltered) `ds`
    """
    if not key or not value: return ds
    
    if metadata: key = 'metadata.'+key
    return [d for d in ds if reduce(lambda x,f: getattr(x,f,None),[d]+key.split('.'))==value],
    
@_node('metadata_name')
def get_metadata(db):
    """
    Return the union of all db element metadata
    """
    meta = sorted(reduce(lambda a,b: set(a).union(b),[d.metadata.fields() for d  in db]))
    meta = [m for m in meta if m[0]!='_']
    return meta
    
@_node('sorted_dataset')
def sort(ds,key, metadata=True):
    """
    Sort `ds` by `key`
    
    If `key` is a list of keys, sort items by all keys, in order.
    If `metadata` is True, key(s) are looked for into the 'metadata' attribute
    """
    def mget(d,key):
        return reduce(lambda x,f: getattr(x,f,None),[d]+key)
        
    def format_key(k):
        key = key.split('.')
        if metadata: key = ['metadata']+key
        return k
        
    if isinstance(key,basestring):
        key = [format_key(key)]
    else:
        key = map(format_key,key)
    key_num = len(key)
    
    sort_key = [map(mget,[d]*key_num,key) for d in ds]
    order    = sorted(range(len(a)), key=a.__getitem__)
    
    return [ds[i] for i in order]
    
@_node('clustered_dataset')
def group_by(ds, key, metadata=True, flat=True, sort_key=None):
    """ group `ds` element by the values of given `key` 
    
    Return a dictionary of lists where the keys are the set of possible values
    of the value of attributes `key` of item in `ds`.
    
    `key` can contain '.' (ex: 'attr.subattr') meaning to process sub-attribute

    If `key` is a list of keys, it clusters `ds` recursively either in:
     - a 'flat' dict where the keys are the tuple of possible values, if flat=True
     - a dict of dict (etc...), otherwise
    
    If `metadata` is True, key(s) are looked for into the 'metadata' attribute
      i.e. same as cluster_db(db, 'metadata.'+key, metadata=False)
    
    ## not implemented: If `sort_key` is not None, it should be a suitable argument for the `sort`
    function, which is then applied on each group. 
    """
    if len(key)==1:
        key = key[0]
        
    if isinstance(key, basestring):
        key = key.split('.')
        if metadata: key = ['metadata']+key
        def mget(d,key):
            return reduce(lambda x,f: getattr(x,f,None),[d]+key)
        
        group = {}
        for d in ds:
            group.setdefault(mget(d,key),[]).append(d)
        
        #if sort_key is not None:
        #    for k,g in group.iteritems():
        #        group[k] = sort(g,key=sort_key,metadata=metadata)
    else:
        group = group_by(ds, key[0], metadata=metadata)
        for k,subds in group.iteritems():
            group[k] = group_by(subds, key[1:], metadata=metadata, sort_key=sort_key, flat=False)
        
        if flat==True:
            group = _flatten_hierarchical(group,depth=len(key))
            
    return group

def _flatten_hierarchical(hdist, depth, base_key=None):
    """
    Flatten hierarchical dictionaries (dict of dict ...) `hdist`
    
    Convert a hierarchical dictionaries into a "flat" dictionary (i.e. which 
    does not contain subdictionary) where the keys are keys tuple of all levels. 
    Example::
    
      d1 = dict(a=dict(b=1), c=dict(b=2,c=3)
      d2 = _flatten_hierarchical(d2,depth=2)
      # d2={('a','b'):1,('c','b'):2,('c','c'):3}
    
    `depth` is the (maximum) depth of subdictionary (starting at 1)
    `base_key` is used internally for recursivity
    
    This function is used by the `group_by` function to convert the results 
    obtained with multi='hierarchical' into the result for multi='flat'
    """
    if base_key is None: base_key = ()
    if depth<1: return base_key
    
    flat = {}
    for k,v in hdist.iteritems():
        k = base_key+(k,)
        if not all(map(hasattr,(v,v),('keys','iteritems'))):
            flat[k] = v
        else:
            subdict = _flatten_hierarchical(v, depth=depth-1, base_key=k)
            flat.update(subdict)
    
    return flat
    
@_node('tree')  
def load_tree(db_item, ext='.tree'):     ## still useful ?
    return _Data.load(db_item.output+ext)

