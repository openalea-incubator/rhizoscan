"""
Reading and general display the snoopim "Results" data, i.e. found visual word
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
    """
    Read the content of `word_file`
    
    :Outputs:
      - header: dictionary of key=value found at thebeginning of the files
      - words: list of words (body of the file), each word is a dictionary with
               the keys: 'fid', 'img', 'match_num', 'bbox'
    """
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
    

