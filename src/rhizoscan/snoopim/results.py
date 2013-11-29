"""
Some tool to read the computed "Results" data, i.e. visual word found

"""
import numpy as np

from . import get_dataset_path
from .database import image_to_id_map as img2id
from .images   import get_thumbnails


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
          The name of the dataset 
     - `word_group` 
          Either the name (string) of the word folder, or an integer of the word 
          folder position in the alphabetically sorted folder list
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

def words_in_images(dataset_name, word_group, image_list, plot=False):
    """
    Return the list of images with bbox the words bbox in `wordset` list, and their bbox
    
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
        from matplotlib import pyplot as plt
        for i,(im,bbox) in enumerate(res):
            plt.subplot(len(res),1,i)
            display_image(dataset_name=dataset_name,image=im,words_bbox=bbox)

    return res


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
        x = ratio*np.array(bbox)[[0,2,2,0,0]]
        y = ratio*np.array(bbox)[[1,1,3,3,1]]
        plt.plot(x,y)
    plt.axis(axis)
    
def word_to_itemset(dataset_name, word_group=-1, item='word', filename=None, verbose=False):
    """
    Create an itemset matrix from the visual words extracted in `word_group`
    
    :Inputs:
     - `dataset_name`:
          The name of the dataset 
     - `word_group` 
          Either the name (string) of the word folder, or an integer of the word 
          folder position relative to the alphabetically sorted folder set
          *** if it is a list, it is returned as is. see note below ***
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
          
    :*** Note: ***:
       If given word_group is a list, it is considered to be the itemset, and 
       this function returns it directly. This provide a simple system to 
       compute itemset only once and avoid transparently to its recomputation by
       functions using needing it. 
    """
    if isinstance(word_group,list):
        return word_group
        
    from os.path import join, dirname
    
    word_files = get_word_files(dataset_name=dataset_name,word_group=word_group)
    word_dir   = dirname(word_files[0])
    if verbose:
        print "processing %d .res files in %s" % (len(word_files),word_dir)
    
    # create dictionay of word files with key the number of the file
    import re
    find_num = re.compile('\d+')
    imgset = dict()
    id_map = img2id(dataset_name)    # map img name to img id
    for f in word_files:
        wid  = int(find_num.findall(f)[-1])
        word = read_visual_word(f)[1]
        imgset[wid] = [id_map.get(w.get('img',None),None) for w in word]
    
    if item=='word':
        wordset = dict()
        for wid,img_list in imgset.iteritems():
            for img in img_list:
                wordset[img].append(wid)
        itemset = wordset
    else:
        itemset = imgset
        
    if filename:
        with open(join(word_dir,filename), 'w') as f:
            for tid, item_list in itemset.iteritems():
                f.write(' '.join(map(str,item_list)) + '\n')
    
    return itemset


