"""
Tools to process snoopim extracted words
"""
import numpy as np

from . import results as res
from .database import image_to_id_map as img2id

def cluster_wordset(dataset_name,word_group,inflation,save=False):
    """
    Cluster specified wordset
    
    `inflation` should be a value in (0.,30.) or a list of such values
    
    use `load_cluster` to load the saved cluster
    """
    print "processing %s-%s with inflation=%s" %(dataset_name,word_group,inflation)
    print " ***** compute similarity matrix *****"
    words = res.load_words(dataset_name,word_group)
    prox  = compute_word_proximity(words,dataset_name)

    inflation = np.atleast_1d(inflation)
    for inf in inflation:
        print " ***** clustering (inflation=%g) *****" % inf
        c = mcl(prox,float(inf))
        print " ========>  %d clusters" % c.max()
        if save:
            filename = res.get_dataset_path(dataset_name,'Results',word_group,'cluster_inf%g'%inf)
            with open(filename,'w') as f:
                f.write(' '.join(map(str,c)))
                
def load_wordset_cluster(dataset_name, word_group, inflation):
    """
    Load wordset cluster saved by `cluster_wordset`
    
    inflation should be the scalar
    """
    filename = res.get_dataset_path(dataset_name,'Results',word_group,'cluster_inf%g'%inflation)
    return np.loadtxt(filename,dtype=int)    
        
def list_wordset_cluster(dataset_name, word_group):
    """
    List all inflation value for which cluster has been computed
    
    See also:
     - `cluster_word` (computing & saving)
     - `load_wordset_cluster` (loading)
    """
    from glob import glob
    base_name = res.get_dataset_path(dataset_name,'Results',word_group,'cluster_inf')
    flist = glob(base_name + '*')
    inf_start = len(base_name)
    inf = [float(f[inf_start:]) for f in flist]
    
    return inf

def mcl(proximity, inflation=None):
    """
    Call `mcl` to cluster ...
    
    :Inputs:
      - `proximity`:
            The proximity value: none zeros are graph edges
      - `inflation`:
            The mcl 'inflation' parameter. If None, let mcl use default
    
    :Outputs:
      ???
    
    :Warning!!!:
      - mcl should be installed
      - it is runned outside of python (system call)
    """
    from subprocess import Popen, PIPE
    import re
    
    # construct mcl input string
    i,j = proximity.nonzero()
    mcl_in = np.recarray(i.size,dtype=np.dtype([('i',int),('j',int),('v',proximity.dtype)]))
    mcl_in['i'] = i
    mcl_in['j'] = j
    mcl_in['v'] = proximity[i,j]
    mcl_in = '\n'.join(map(str,mcl_in))
    mcl_in = re.sub('[()\[\],]','',mcl_in)
        
    # make inflation parameter
    if inflation is None: inflation = []
    else:                 inflation = ['-I',str(inflation)]
    
    # call mcl and retrieve output
    #cmd = ['mcl', '-', '--abc', '-V','all'] + inflation + ['-o','-']
    cmd = ['mcl', '-', '--abc'] + inflation + ['-o','-']
    p = Popen(cmd, stdout=PIPE, stdin=PIPE)
    out = p.communicate(input=mcl_in)[0]
    
    # convert (i.e. read) mcl output
    group = []
    for c in out.split('\n'):
        if len(c):
            group.append(map(int,c.split('\t')))
    cluster = np.zeros(proximity.shape[0],dtype=int)
    for i,g in enumerate(group):
        cluster[g] = i+1
    
    return cluster
    

def compute_word_proximity(words, img2id_map, verbose=False):
    """
    Compute a matrix of the proximity of all words
    
    :Inputs:
     - `words`:
          a word dictionary, such as returned by `snoopim.results.load_words`
     - `img2id_map`: either
          - a dictionary of (filename,id) of all images
          - the name of the dataset this map should be contructed for
     - `verbose`:
          If True, print computation states
    
    :Outputs:
        The returned matrix has shape len(words)^2. The value of entry (w1,w2) 
        is the sum of overlapping area of the bbox of word w1 and w2, for all 
        images.                                    
    """
    from scipy.sparse import lil_matrix, coo_matrix
    from rhizoscan.ndarray import diagonal
    
    if isinstance(img2id_map,basestring):
        img2id_map = img2id(img2id_map)
    
    if verbose: print 'contruct list of all (word,image,bbox) tuple'
    # construct array of (word-id, image-id, xmin, ymin, xmax, ymax)
    # for all words in all image
    W = [[wid,img2id_map[m['img']]]+m['bbox'] for wid,word in words.iteritems() for m in word]
    dt = np.dtype(zip(('wid','iid','xmin','ymin','xmax','ymax'), (int,)*6))
    W = np.array(W).view(dtype=dt)
    
    # sort by image id and find indices of image changes 
    #   the indices are the last before the change
    W.sort(axis=0,order='iid')
    ind = np.diff(W['iid'],axis=0).nonzero()[0]
    ind = [0] + (ind+1).tolist() + [W.shape[0]]
    
    # for each image, compute all bbox intersection, and add it to proximity
    if verbose: print 'compute bbox intersection area'
    I = []
    #prox = lil_matrix((len(words),)*2)
    prox = np.zeros((len(words),)*2)
    for k in range(len(ind)-1):
        i,j = ind[k],ind[k+1]
        ##print k,i,j,np.unique(W['iid'][i:j])
        w = W[i:j]
        xmin = np.maximum(w['xmin'],w['xmin'].T)
        xmax = np.minimum(w['xmax'],w['xmax'].T)
        dx = xmax-xmin
        dx[dx<0] = 0
        ymin = np.maximum(w['ymin'],w['ymin'].T)
        ymax = np.minimum(w['ymax'],w['ymax'].T)
        dy = ymax-ymin    
        dy[dy<0] = 0      
        area = dx*dy
        
        diagonal(area)[:] = 0
        w1,w2 = area.nonzero()
        w = w.ravel()
        prox[w['wid'][w1],w['wid'][w2]] = area[w1,w2]
        ##loc_prox = coo_matrix((area[w1,w2],(w['wid'][w1],w['wid'][w2])),shape=prox.shape)
        ##prox += loc_prox
        int_num, w_num = w1.size/2,area.shape[0]
        if verbose: print '  > img %d: %d intersections found on %d words (%.1f avg)' % (k,int_num,w_num,int_num/float(w_num))
                     
    return prox
    # sum intersections
    if verbose: print 'sum detected intersection'
    prox = {}
    def add_prox(w1w2area):
        w1,w2,area = w1w2area
        prox.setdefault(w1,{}).setdefault(w2,0)
        prox[w1][w2] += area
    
    return W


def bbox_arrays(words, img_map_id):
    """
    return the `words` bbox as sparse arrays
    
    :Inputs:
     - `words`:
          a word dictionary, such as return by `load_words`
     - `img_map_id`: either
          - a dictionary of (filename,id) of all images
          - the name of the dataset this should be contructed for
    
    :Outputs:
      The four xmin,ymin,xmax,ymax sparse array each of shape [word-num,img-num]
    """
    from scipy.sparse import lil_matrix
    #import numpy as np
    
    if isinstance(img_map_id,basestring):
        img_map_id = img2id(img_map_id)
     
    shape = (max(words.keys())+1,max(img_map_id.values())+1)
    xmin = lil_matrix(shape,dtype=int)
    ymin = lil_matrix(shape,dtype=int)
    xmax = lil_matrix(shape,dtype=int)
    ymax = lil_matrix(shape,dtype=int)
    #bbox = -np.ones(shape+(4,),dtype=int)
    
    for wid,word in words.iteritems():
        img,bbox = zip(*[(img_map_id[match['img']],match['bbox']) for match in word])
        xmin[wid,img] = [bb[0] for bb in bbox]
        ymin[wid,img] = [bb[1] for bb in bbox]
        xmax[wid,img] = [bb[2] for bb in bbox]
        ymax[wid,img] = [bb[3] for bb in bbox]
        
    return xmin,ymin,xmax,ymax
    
def _invert_dict_of_dict(dod, is_lol=False):
    """
    Invert a dictionary-of-dictionary
    
    if is_lol, input `dod` is expected to be a list-of-lists 
    
    return a dict-of-dict dod_inv, such that dod_inv[a][b] = dod[b][a]
    """
    # convert dod into a list-of-lists, then flatten it
    if not is_lol:
        lol  = [[(a,b,val) for b,val in bdict.iteritems()] for a,bdict in dod.items()]
    else:
        lol = dod
    flat = []
    map(flat.extend,lol)
    
    # make the inverted dict-of-dict
    dod_inv = dict()
    def dod_inv_set(abval): 
        a,b,val=abval; 
        dod_inv.setdefault(b,{})[a]=val
    map(dod_inv_set, flat)
    
    return dod_inv

