"""
a few image conversion tool
"""

##could also contain io tools ? but then needs to be rename

##add toFloat(img,dtype='float32')  and  toInt(img, dtype='uint8')
##add convert(img, dtype=None, color=None)

import numpy as _np
from rhizoscan.misc import printWarning
from rhizoscan.ndarray import add_dim

def gray(image):
    """ 
    Convert the given image to gray, or die trying
    If the image has 2 dimensions: return it
    If it has 3 dimensions:
        if last dimension has size 3, does a rgb to gray conversion
        otherwise, print a warning and return only the first image channel (the 0 of last dimension)
    if the image has not 2 or 3 dimension, raise an error
    """
    if _np.ndim(image) not in (2,3):
            raise TypeError("dimension of input image should be 2 (gray) or 3 (rgb). image.shape=" + str(image.shape))
            
    elif _np.ndim(image)==3:
        if image.shape[2]==3:  # rgb to gray conversion
            image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
            
        else: 
            if image.shape[2]!=1:
                printWarning('image has wrong 3rd dimension, returning the 1st channel');
            image = image[:,:,0]
    
    return image
    
def channel_stack(red,green,blue):
    """
    Stack the separated (2D) channels as one (3D) color image
    
    Input:
        - red:   the red   channel
        - green: the green channel
        - blue:  the blue  channel
        
        All input must 2d and of same shape. Otherwise, at least one should be 2d
        and the other broadcastable to this shape (i.e. you can sum them)
        
    Output:
        An NxMx3 color array, where NxM is the 2D shape of inputs
    """
    addD = lambda x:add_dim(x, axis=-1, size=1)
    red,green,blue = map(addD,_np.broadcast_arrays(red,green,blue))
    
    return _np.concatenate((red,green,blue),axis=-1)
    


