import numpy as np
import scipy.ndimage as nd
import os

from glob import glob as _ls

from rhizoscan.image import ImageSequence as _Seq
from rhizoscan.image import Image         as _Img
from rhizoscan.workflow import node as _node # to declare workflow nodes
image_path = '/Users/diener/root_data/test_set/'
seq_path   = '/Users/diener/root_data/nacry/sequence_21-05-2012/'
dart2_path  = '/Users/diener/root_data/nacry/scan_serie_1/DART/scan_serie_1_J2'
dart3_path  = '/Users/diener/root_data/nacry/scan_serie_1/DART/scan_serie_1_J3'

# icon of openalea package
__icon__ = 'database.png'



#@_node({'name':'files'},inputs=({'name':'reload','interface':'IBool'},))
def _test_image_list(update=False, files=[]):
    if update or len(files)==0:
        for i in range(len(files)): files.pop()
        files.extend(_ls(os.path.join(image_path,'*')))

        # test file type
        #for f in files:
        #    if os.path.isdir(f) or f[0] == '.':
        #        files.remove(f)
    
    return files        

def _seq_name_(update=False, names=[]):
    if update or len(names)==0:
        for i in range(len(names)): names.pop()
        
        d = _ls(os.path.join(seq_path,'*'))
        files = _ls(os.path.join(d[0],'*'))
        
        for f in files: names.append(os.path.split(f)[-1])
            
    return names
def _dart2_name_(update=False, names=[]):
    if update or len(names)==0:
        for i in range(len(names)): names.pop()
        files = _ls(os.path.join(dart2_path,'*.tif'))
        for f in files: names.append(f)
    return names
def _dart3_name_(update=False, names=[]):
    if update or len(names)==0:
        for i in range(len(names)): names.pop()
        files = _ls(os.path.join(dart3_path,'*.tif'))
        for f in files: names.append(f)
    return names
    
#@_node('dart_J2')
def dart_J2(number, color='gray', dtype='f'):
    filename = _dart2_name_()[number]
    return _Img(filename,color=color,dtype=dtype)
#@_node('dart_J3')
def dart_J3(number, color='gray', dtype='f'):
    filename = _dart3_name_()[number]
    return _Img(filename,color=color,dtype=dtype)
    
#@_node('image')
def test_image(filename):
    if not isinstance(filename,basestring):
        filename = _test_image_list()[filename]
        
    if not os.path.exists(filename):
        filename = os.path.join(image_path,filename)
        
    return normalize(_Img(filename,color='gray',dtype='f'))

def normalize(img):
    img -= img.min()
    img /= img.max()
    
    if img.mean() > 0.5: img[:] = 1-img
    
    return img

#@_node('seq')
def seq(filename, color='gray', dtype='f'):
    if not isinstance(filename,basestring):
        filename = _seq_name_()[filename]
    elif filename[-4:]!='.jpg':
        filename += '.jpg'
        
    path = os.path.join(seq_path,'*',filename)
    seq  = _Seq(path,color=color,dtype=dtype, filter=normalize)
    seq.path = path

    return seq
    
