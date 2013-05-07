__icon__    = 'database.png'   # icon of openalea package


import numpy as _np

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes

from rhizoscan.workflow import Struct as _Struct
from rhizoscan.workflow import Data   as _Data

from . import _print_state, _print_error, _param_eval 


@_aleanode('image_list', 'invalid_file', 'output_directory')
def parse_image_db(ini_file, output='output', verbose=False):
    """
    Return an iterator over all images following parsing format
    
    :Output:
    
    image_list: 
        a list of objects containing the following attributes
            . filename: the file name of input image
            . output:   base name for output related to this image
            . metadata: metadata related to this image
            
    invalid_file: 
        the list of files that were found, but for which metadata
        could not be parsed
    
    output_directory:
        the global output directory = ini_file dir/output
    
    :todo: 
        finish doc - input
        check missing images (1 of each type for all time steps)
    """
    import ConfigParser as cfg
    from os.path import join as pjoin
    from os.path import splitext, dirname
    import re
    from time import strptime
    from glob import glob
    
    from rhizoscan.tool import abspath
    
    # load content of ini file
    ini = cfg.ConfigParser()
    ini.read(ini_file)
    ini = _Struct(**dict([(s,_Struct(**dict(ini.items(s)))) for s in ini.sections()])) 
        
    # find all image files that fit pattern given in ini_file
    # -------------------------------------------------------
    base_dir = dirname(ini_file)
    base_out = abspath(output, base_dir)
    
    # list all suitable files
    file_pattern = ini['PARSING']['pattern']
    file_pattern = re.split('[\[\]]',file_pattern)
    
    glob_pattern = pjoin(base_dir,'*'.join(file_pattern[::2]))
    file_list = sorted(glob(glob_pattern))
    
    if verbose:
        print 'glob:', glob_pattern
        #print '   ' + '\n   '.join(file_list)
    
    # prepare metatdata parsing
    # -------------------------
    # function that load a referenced fields from the ini file, used for '$' type
    def get_from_ini(field, default):
        value   = ini[field]
        return value
    print file_pattern
    
    # meta data list and regular expression to parse file names
    meta_parser = re.compile('(.*)'.join([fp.replace('*','.*') for fp in file_pattern[::2]]))
    meta_list = [_Struct(name=s[0],type=s[1]) for s in [m.split(':') for m in file_pattern[1::2]]]
    date_pattern = ini['PARSING']['date']
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
    if ini['PARSING'].has_field('group'):
        g = eval(ini['PARSING']['group']) ## eval... value is a dict
        dlist = [dirname(fi) for fi in file_list]
        fenum = [int(di==dlist[i]) for i,di in enumerate(dlist[1:])]          # diff of dlist
        fenum = _np.array(reduce(lambda L,y: L+[(L[-1]+y)*y], [[0]]+fenum))   # ind of file in resp. dir.
        group = _np.zeros(fenum.max()+1,dtype='|S'+str(max([len(gi) for gi in g.itervalues()])))
        for start in sorted(g.keys()):
            group[start:] = [g[start]]*(len(group)-start)
        group = group[fenum]
        if verbose:
            print 'group:', group
    else:
        group = None
    
    if verbose:
        print 'metadata:'
        print '   pattern:', meta_parser.pattern
        print '   type:', ', '.join((m.name+':'+m.type for m in meta_list))
    # parse all image files, set metadata and remove invalid
    # ------------------------------------------------------
    img_list = []
    invalid  = []
    rm_len = len(base_dir)  ## imply images are in base_dir. is there a more general way
    for ind,f in enumerate(file_list):
        try:
            subf = f[rm_len+1:]
            out  = pjoin(base_out, splitext(subf)[0])
            meta_value = meta_parser.match(subf).groups()
            if verbose>1:
                print '   ' + str(meta_value) + ' ' + f
            meta = _Struct(**default_meta)
            if group is not None:
                meta.merge(get_from_ini(group[ind], []))
            for i,value in enumerate(meta_value):
                field = meta_list[i].name
                value = meta_list[i].eval(value)
                if field=='$': meta.merge(value)
                else:          meta[field] = value
                
            img_list.append(_Struct(filename=f, metadata=meta, output=out))
        except Exception as e:
            invalid.append((type(e).__name__,e.message, f))
            
    return img_list, invalid, base_out
    
    
@_aleanode('to_update')
def to_update(db, suffix='.tree'):
    from os.path import exists
    return [d for d in db if not exists(d.output+suffix)],

@_aleanode('db_data')
def retrieve_data_file(db, name='tree', suffix='.tree'):
    """
    Add the data filename to all db element as attribut 'name'
    """
    for d in db: d[name] = d.output+suffix
    return db,
    

@_aleanode('filtered_db')
def filter(db, key=None, value=None, metadata=True):
    """
    Return the subset of db that has its key attribut equal to value
    
    :Input:
      - key:    (*)
        key to filter db by. Can contains dot, eg. genotype.base
      - value:  (*)
        The value the key must have
      - metadata:
        if True, look for key in the 'metadata' attribut
        Same as filter(db,'metadata.'+key,value, metadata=False) 
        
      (*)if key or value is None, return the full (unfiltered) db
    """
    if key is None or value is None or key=='': return db
    
    if metadata: key = 'metadata.'+key
    return [d for d in db if reduce(lambda x,f: getattr(x,f,None),[d]+key.split('.'))==value],
    
@_aleanode('metadata_name')
def get_metadata(db):
    """
    Return the union of all db element metadata
    """
    meta = sorted(reduce(lambda a,b: set(a).union(b),[d.metadata.fields() for d  in db]))
    meta = [m for m in meta if m[0]!='_']
    return meta
    
@_aleanode('clustered_db')
def cluster(db, key, metadata=True):
    """ cluster db by (unique) 'key' 
    
    If key is a string, return a dictionary of lists with dictionary keys the 
    possible db key value.
    If key is a list of key, cluster recursively into dict of dict (...)
    
    if 'metadata' is True, looks for key in the metadata attribut of the db item
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

@_aleanode('tree')  
def load_tree(db_item, ext='.tree'):     ## still useful ?
    return _Data.load(db_item.output+ext)

