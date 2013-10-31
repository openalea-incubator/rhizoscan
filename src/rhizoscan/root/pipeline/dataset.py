__icon__    = 'database.png'   # icon of openalea package


import numpy as _np

from rhizoscan.workflow import node as _node # to declare workflow nodesfrom rhizoscan.workflow  import node as _node # decorator to declare workflow nodes

from rhizoscan.datastructure import Mapping as _Mapping
from rhizoscan.datastructure import Data    as _Data

from . import _print_state, _print_error, _param_eval 


#@_node('image_list', 'invalid_file', 'output_directory')
@_node('image_list', 'invalid_file', 'output_directory', hidden=['verbose'])
def make_dataset(ini_file, output='output', out_suffix='_', verbose=False):
    """
    Return a list of dataset entry following parsing rules found in `ini_file`
    
    :Inputs:
      - `ini_file`: 
          file with ini-formated content indicating the dataset to be loaded ##some doc?
      - `output`:
          String used to define the storage associated to each loaded entry.
          If None, does not set storage for dataset entry
      - `out_suffix':
          String to append to the storage string associated to loaded entry
      - `verbose`:
          If >0, print some message on loaded dataset
    
    :Outputs:
      - a list of Mapping containing the following attributes:
          - `filename`: the file name of input image
          - `metadata`: metadata related to this image
          - if `output` is given, each has a storage configured (details below)
            
      - the list of **invalid files**: files found but that could not be parsed
        
      - The base directory to all entry storage. If `storage` is an absolute 
        path, then this is the returned value. Otherwise, it is the `output` 
        string preceded by directory in which `ini_file` is.
    
    
    :Output:
        A `Sequence` of `Mapping` object, one for each found entry.
         - Its StorageEntry is set to:    
              [base_output_dir]
         - Loaded entries StorageEntry are:
              [base_output_dir]/[entry-base].entry
         - Loaded entries have their storage configure to store content in:
              [base_output_dir]/[entry-base][out-suffix][content-key]
          
        where:
          base_output_dir is the 3rd value returned by this function
          entry-base is the filename of the loaded entry with the directory of
            ini_file and the file extension removed
          out_suffix is the value given as argument
          content-key is the key of the content to be stored
          
    :todo:
        - finish doc on ini_file structure
    """
    import ConfigParser as cfg
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
    ini = cfg.ConfigParser()
    ini.read(ini_file)       
    ini = _Mapping(**dict([(s,_Mapping(**dict((k,_param_eval(v)) for k,v in ini.items(s)))) for s in ini.sections()])) 
    
    if verbose>2:
        print 'loaded ini:'
        print ini.multilines_str(tab=1)
        
    # find all image files that fit pattern given in ini_file
    # -------------------------------------------------------
    base_dir = dirname(abspath(ini_file))
    base_out = abspath(output, base_dir)
    
    # list all suitable files
    file_pattern = ini['PARSING']['pattern']
    file_pattern = file_pattern.replace('\\','/')      # for windows
    file_pattern = re.split('[\[\]]',file_pattern)
    
    glob_pattern = pjoin(base_dir,'*'.join(file_pattern[::2]))
    file_list = sorted(glob(glob_pattern))
    
    if verbose:
        print 'glob:', glob_pattern
        if verbose>1:
            print '   ' + '\n   '.join(file_list)
    
    # prepare metatdata parsing
    # -------------------------
    # function that load a referenced fields from the ini file, used for '$' type
    def get_from_ini(field, default):
        value   = ini[field]
        return value
    
    # meta data list and regular expression to parse file names
    meta_parser = re.compile('(.*)'.join([fp.replace('*','.*') for fp in file_pattern[::2]]))
    meta_list = [_Mapping(name=s[0],type=s[1]) for s in [m.split(':') for m in file_pattern[1::2]]]
    date_pattern = ini['PARSING'].get('date','')
    for m in meta_list:
        if m.type=='date': 
            m.eval = lambda s: strptime(s, date_pattern)
        elif m.type=="$":
            default = m.name+'_default'
            m.eval = lambda name: get_from_ini(name, default=default)
        else: 
            m.eval = eval(m.type)  ##Security issue ! (what it should do eval('int'))
            
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
    rm_len = len(base_dir)  ## imply images are in base_dir. is there a more general way
    for ind,f in enumerate(file_list):
        try:
            if rm_len>0: subf = f[rm_len+1:]
            else:        subf = f 
            subf = subf.replace('\\','/')   # for windows
            out_store = pjoin(base_out, splitext(subf)[0]) + out_suffix
            out_entry = pjoin(base_out, splitext(subf)[0]) + '.namespace'
            meta_value = meta_parser.match(subf).groups()
            if verbose>1:
                print '   ' + str(meta_value) + ' from ' + subf + str(rm_len)
            meta = _Mapping(**default_meta)
            if group is not None:
                meta.update(get_from_ini(group[ind], []))
            for i,value in enumerate(meta_value):
                field = meta_list[i].name
                value = meta_list[i].eval(value)
                if field=='$': meta.update(value)
                else:          meta[field] = value
                
            ds_entry = _Mapping(filename=f, metadata=meta)
            ds_entry.__loader_attributes__ = ['filename','metadata']
            ds_entry.set_map_storage(out_store)
            ds_entry.set_storage_entry(out_entry)
            img_list.append(ds_entry)
        except Exception as e:
            invalid.append((type(e).__name__,e.message, f))
            
    return img_list, invalid, base_out
    
    
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

