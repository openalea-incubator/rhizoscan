__icon__    = 'database.png'   # icon of openalea package


import numpy as _np

from rhizoscan.workflow import node as _node # to declare workflow nodesfrom rhizoscan.workflow  import node as _node # decorator to declare workflow nodes

from rhizoscan.datastructure import Mapping as _Mapping
from rhizoscan.datastructure import Data    as _Data

from . import _print_state, _print_error, _param_eval 


#@_node('image_list', 'invalid_file', 'output_directory')
@_node('image_list', 'invalid_file', 'output_directory', hidden=['verbose'])
def make_dataset(ini_file, base_dir=None, data_dir=None, out_dir='output', out_suffix='_', verbose=False):
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
          If not given, use the `base_dir`
          If it is not an absolute path, preppend it with `base_dir`
      - `out_dir`:
          Directories to set output into              (see 'directories' below)
          If it is not an absolute path, preppend it with `base_dir`
      - `out_suffix':
          String to append to output files            (see 'directories' below)
      - `verbose`:
          If >0, print some message on loaded dataset
    
    :Outputs:
      - A list of `Mapping` object, one for each file found with suitable output
        files configure (nothing is saved)
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
    if base_dir is None: base_dir = dirname(abspath(ini_file))
    if data_dir is None: data_dir = base_dir
    else:                data_dir = abspath(data_dir, base_dir)
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
    meta_parser = re.compile('(.*)'.join([fp.replace('*','.*') for fp in file_pattern[::2]]))
    meta_list = [_Mapping(name=s[0],type=s[1]) for s in [m.split(':') for m in file_pattern[1::2]]]
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
        
    # parse all image files, set metadata and remove invalid
    # ------------------------------------------------------
    img_list = []
    invalid  = []
    rm_len = len(data_dir)  ## imply images are in base_dir. is there a more general way
    for ind,f in enumerate(file_list):
        try:
            if rm_len>0: subf = f[rm_len+1:]
            else:        subf = f 
            subf = subf.replace('\\','/')   # for windows
            out_store = pjoin(out_dir, splitext(subf)[0]) + out_suffix + '{}'
            out_file  = pjoin(out_dir, splitext(subf)[0]) + '.namespace'
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
                
            ds_item = _Mapping(filename=f, metadata=meta)
            ds_item.__loader_attributes__ = ['filename','metadata']
            ds_item.set_map_storage(out_store)
            ds_item.set_file(out_file)
            img_list.append(ds_item)
        except Exception as e:
            invalid.append((type(e).__name__,e.message, f))
            
    return img_list, invalid, out_dir
    
    
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
   
@_node('filename', 'metadata', 'output')
def split_item(db_item):
    return db_item.filename, db_item.metadata, db_item.output
    
@_node('to_update')
def to_update(db, suffix='.tree'):
    from os.path import exists
    return [d for d in db if not exists(d.output+suffix)],

@_node('db_data')
def retrieve_data_file(db, name='tree', suffix='.tree'):
    """
    Add the data filename to all db element as attribute 'name'
    """
    for d in db: d[name] = d.output+suffix
    return db,
    
@_node('db_column')
def get_column(db, suffix, missing=None):
    """
    Retrieve the dataset column related to 'suffix'
    """
    def load(d):
        try:
            return _Data.load(d.output+suffix)
        except:
            return missing
            
    return [load(d) for d in db]
    
@_node('filtered_db')
def filter(db, key='', value='', metadata=True):
    """
    Return the subset of db that has its key attribute equal to value
    
    :Input:
      - key:    (*)
        key to filter db by. Can contains dot, eg. genotype.base
      - value:  (*)
        The value the key must have
      - metadata:
        if True, look for key in the 'metadata' attribute
        Same as filter(db,'metadata.'+key,value, metadata=False) 
        
      (*)if key or value is empty, return the full (unfiltered) db
    """
    if not key or not value: return db
    
    if metadata: key = 'metadata.'+key
    return [d for d in db if reduce(lambda x,f: getattr(x,f,None),[d]+key.split('.'))==value],
    
@_node('metadata_name')
def get_metadata(db):
    """
    Return the union of all db element metadata
    """
    meta = sorted(reduce(lambda a,b: set(a).union(b),[d.metadata.fields() for d  in db]))
    meta = [m for m in meta if m[0]!='_']
    return meta
    
@_node('clustered_db')
def cluster(db, key, metadata=True):
    """ cluster db by (unique) 'key' 
    
    If key is a string, return a dictionary of lists with dictionary keys the 
    possible db key value.
    If key is a list of key, cluster recursively into dict of dict (...)
    
    if 'metadata' is True, looks for key in the metadata attribute of the db item
    i.e. same as cluster_db(db, 'metadata.'+key, metadata=False)
    
    Note: In case of multiple keys (list), then the sub dict only contains keys
          that exist in its cluster. Thus, all sub dict might not have the same
          list of keys.
    """
    if len(key)==1:
        key = key[0]
        
    if isinstance(key, basestring):
        key = key.split('.')
        if metadata: key = ['metadata']+key
        def mget(d,key):
            return reduce(lambda x,f: getattr(x,f,None),[d]+key)
        
        cluster = {}
        for d in db:
            cluster.setdefault(mget(d,key),[]).append(d)
        return cluster
        
    else:
        cluster = cluster_db(db, key[0], metadata=metadata)
        for k,subdb in cluster.iteritems():
            cluster[k] = cluster_db(subdb, key[1:], metadata=metadata)
        
        return cluster

@_node('tree')  
def load_tree(db_item, ext='.tree'):     ## still useful ?
    return _Data.load(db_item.output+ext)

