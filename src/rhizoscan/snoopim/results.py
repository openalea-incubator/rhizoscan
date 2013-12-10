"""
Some tool to read the computed "Results" data, i.e. visual word found

"""
import numpy as np

from . import get_dataset_path
from .database import image_to_id_map as img2id
from .images   import get_thumbnails


def image_with_words(dataset_name, word_group, wordset, plot=False):
    """
    Return the list of images containing the words in `wordset` list, and their bbox
    
    output is a list of [img_file, list-of-bbox]:
     [[img1,[bb1,bb2,...], [img2,[bb1,bb2,...], ...]
     
    the length of output list if the number of images that contains the given
    words. The length of bbox lists (ie. len(out[.][1])) is the number of words.
    """
    word_files = get_word_files(dataset_name,word_group=word_group)
    
    # retrieve given words (word id base on filename - not useful?)
    import re
    find_num = re.compile('\d+')
    words = dict()
    for f in word_files:
        wid = int(find_num.findall(f)[-1])
        if wid in wordset:
            words[wid] = read_visual_word(f)

    # find images that contains all given words
    words = words.values()
    img = set([w['img'] for w in words[0][-1]])
    for word in words[1:]:
        img.intersection_update([w['img'] for w in word[-1]])
        
    # make (image,[bbox]) list
    res = []
    for im in img:
        bbox = [[w['bbox'] for w in word[-1] if w['img']==im][0] for word in words]
        res.append((im,bbox))

    if plot:
        from math import ceil
        from matplotlib import pyplot as plt
        col = int(plot)
        row = ceil(len(res)/float(col))
        for i,(im,bbox) in enumerate(res):
            plt.subplot(row,col,i)
            display_image(dataset_name=dataset_name,image=im,words_bbox=bbox)

    return res

def display_words(dataset_name, words, images=None, column=1):
    """
    Display given `words` in given `images`
    
    if `images` is None, select all images that contains all the given words
    """
    # find images that contains all given words
    if images is None:
        images = set([match['img'] for match in words.values()[0]])
        for word in words.values()[1:]:
            images.intersection_update([match['img'] for match in word])
        
    # make (image,[bbox]) list for display_image arguments
    to_display = []
    for im in images:
        bbox = [[w['bbox'] for w in word if w['img']==im] for word in words.values()]
        bbox = [bb[0] if len(bb) else None for bb in bbox]
        to_display.append((im,bbox))

    # display images
    from math import ceil
    from matplotlib import pyplot as plt
    row = ceil(len(to_display)/float(column))
    for i,(im,bbox) in enumerate(to_display):
        plt.subplot(row,column,i)
        display_image(dataset_name=dataset_name,image=im,words_bbox=bbox)

    return len(to_display)

def display_image(dataset_name, image, words_bbox):
    """
    display the given bounding box `bbox` on top of `image`
    
    `dataset_name` is the name of the dataset the image is taken from
    `image` is the filename of the image in `dataset`
    `words_bbox` is a list of lists [x_min, y_min, x_max, y_max]
    """
    from matplotlib import pyplot as plt
    from os.path import split
    image, ratio = get_thumbnails(dataset_name=dataset_name, image_name=split(image)[-1])
    plt.imshow(image)
    axis = plt.axis()
    for bbox in words_bbox:
        if bbox is None:
            plt.plot(0,0)
        else:
            x = ratio*np.array(bbox)[[0,2,2,0,0]]
            y = ratio*np.array(bbox)[[1,1,3,3,1]]
            plt.plot(x,y)
    plt.axis(axis)
    
def word_to_itemset(dataset_name, word_group=-1, item='word', filename=None, verbose=False):
    """
    Create an itemset dict from the visual words extracted in `word_group`
    
    :Inputs:
     - `dataset_name`:
          The name of the dataset 
     - `word_group` 
          Either the name (string) of the word folder, or an integer of the word 
          folder position relative to the alphabetically sorted folder set
     - `item`:
          What are defined as item. Returns either:
           - 'word':  lists of word  (item) found in  each image (transaction)
           - 'image': lists of image (item) found for each word  (transaction)
     - `filename`:
          If not None, save itemset into the given file:
            - the file is stored in the "Results/word-dir" folder
            - it contains one line per item
            - each line is a list item id (see `item_dim`)
     - `verbose`:
          If True, print some execution status

    :Outputs:
       a dictionary with (key,value) = (tid, item_list), where:
         - tid is the id of image for item='word', and of word for 'image'
         - item_list is a list of word if item='word' and of image otherwise
    """
    words  = load_words(dataset_name=dataset_name,word_group=word_group)
    
    id_map = img2id(dataset_name)    # map img name to img id
    imgset = dict((wid,[id_map[match['img']] for match in word]) for wid,word in words.iteritems())

    if item=='word':
        wordset = dict()
        for wid,img_list in imgset.iteritems():
            for img in img_list:
                wordset.setdefault(img,[]).append(wid)
        itemset = wordset
    else:
        itemset = imgset
        
    if filename:
        from os.path import join, dirname
        word_files = get_word_files(dataset_name=dataset_name,word_group=word_group)
        word_dir   = dirname(word_files[0])
        with open(join(word_dir,filename), 'w') as f:
            for tid in range(max(itemset.keys())+1):
                item_list = itemset.get(tid,[])
                print item_list
                f.write(' '.join(map(str,item_list)) + '\n')
    
    return itemset
    
def words_with_images(words, image_filter, support=1):
    """
    Return a subset of the `words` that contains suitable images
    
    `support` indicate how selective the word filter is with respect to the 
    images set repecting `image_filter`:
     - if >=1,   words have to be found in at least `support` images
     - if <1,    words have to be found in at least `support` percent of images
     - if 'all', words have to be found in all images
    """
    # find all suitable images
    import re
    ifilter = re.compile(image_filter).search
    ##images = [[match['img'] for match in word if ifilter(match['img']) is not None] for word in words.values()] 
    ##images = reduce(lambda x,y: x.union(y), images, set())
    
    # contruct a dict of the list of word for each suitable image
    word_in_image = dict()
    word_support  = dict()
    image_list = []
    for wid,word in words.iteritems():
        img = [match['img'] for match in word if ifilter(match['img']) is not None]
        word_support[wid] = len(img)
        image_list.append(img)
        for im in img:
            word_in_image.setdefault(im,[]).append(wid)
    # list of unique image filename 
    image_list = reduce(lambda x,y: x.union(y), image_list, set())
    
    # select the word ids
    if support=='all': min_support = len(image_list)
    elif support<1:    min_support = int(support*len(image_list))
    else:              min_support = int(support)
    selected = [wid for wid,support in word_support.iteritems() if support>=min_support]
        
    return dict((wid,w) for wid,w in words.iteritems() if wid in selected), image_list

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

def compute_word_proximity(words, img2id_map, verbose=False):
    """
    Compute a matrix of the proximity of all words
    
    :Inputs:
     - `words`:
          a word dictionary, such as return by `load_words`
     - `img2id_map`: either
          - a dictionary of (filename,id) of all images
          - the name of the dataset this should be contructed for
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
    # for all word in all image
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

def load_words(dataset_name, word_group, word_filter=None):
    """
    Load word file content
    
    :Inputs:
      - `dataset_name`:
          the name of the dataset
      - `word_group`:
          Either
            - the folder name (str)
            - its position (integer) in the list of results folder sorted alphbetically
       - `word_filter`:
            Either a string that is a regular expression to filter word filename
            or a list of integer that are the word id to keep
            
    :Outputs:
      A dictionary with (key,value) = (word-id, word-dict)
    """
    from os.path import join, dirname
    import re
    
    # make dictionary of (word-id:filename)
    find_num   = re.compile('\d+')
    word_files = get_word_files(dataset_name=dataset_name,word_group=word_group)
    word_dir   = dirname(word_files[0])
    word_files = dict((int(find_num.findall(f)[-1]),f) for f in word_files)

    # filter words based on their filename, or id
    if isinstance(word_filter,basestring):
        re_filter = re.compile(word_filter)
        word_files = dict((wid,fname) for wid,fname in word_files.iteritems() if re_filter.search(fnname) is not None) 
    elif word_filter is not None:
        word_files = dict((wid,fname) for wid,fname in word_files.iteritems() if wid in word_filter)
    
    # load word filename content
    words = dict((wid,read_visual_word(filename)[1]) for wid,filename in word_files.iteritems()) 
        
    return words
        
def read_visual_word(word_file):
    from ast import literal_eval
    
    # eval a number or return txt
    def eval_num(txt):
        try:
            return literal_eval(txt)
        except:
            return txt
    
    # read word file content
    with open(word_file, 'r') as word_file:
        # read header
        line = word_file.readline()
        header  = dict()
        while len(line) and line[0] not in '0123456789':
            key,value = line.split('=')
            if key=='nb_query_points':
                header[key] = map(float,value.split())
            else:
                header[key] = eval_num(value)
            line = word_file.readline()
            
        # read word
        word = []
        while len(line.strip()):
            line = line.split()
            w = dict(fid=int(line[0]), img=line[1], match_num=int(line[2]), bbox=map(int,line[-4:]))
            word.append(w)
            line = word_file.readline()
            
    return header, word

def get_word_files(dataset_name, word_group):
    """
    Return the list of word files for the given dataset and word set
    
    :Inputs:
      - `dataset_name`:
          the name of the dataset
      - `word_group`:
          Either
            - the folder name (str)
            - its position (integer) in the list of results folder sorted alphbetically
            
     :Outputs:
       the (sorted) list of word filenames 
     """
    from glob import glob
    from os.path import join
        
    # retrieve list of computed words
    if isinstance(word_group, basestring):
        word_dir = get_dataset_path(dataset_name, 'Results', word_group)
    else:
        word_dir = get_dataset_path(dataset_name, 'Results')
        word_dir = sorted(glob(join(word_dir,'*')))[word_group]
    
    return sorted(glob(join(word_dir,'*.res')))
    

