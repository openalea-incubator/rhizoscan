import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow import node as _node         # declare workflow nodes
from rhizoscan.workflow import pipeline as _pipeline # declare workflow pipeline

from rhizoscan.ndarray.measurements import clean_label as _clean_label
from rhizoscan.image                import Image       as _Image

##from .dataset import make_dataset as _make_dataset
from . import load_image
from . import detect_petri_plate
from . import compute_graph, compute_tree
from . import _print_state, _print_error

from rhizoscan.root.image import segment_root_image  as _segment_root
from rhizoscan.root.image import remove_background   as _remove_background
from rhizoscan.root.image.seed import detect_leaves  as _detect_leaves


##@_node('failed_files')
##def process(ini_file, indices=None, **kargs):
##    if isinstance(ini_file, basestring):
##        flist, invalid, outdir = _make_dataset(ini_file=ini_file, output='output')
##    else:
##        flist = ini_file
##    
##    if indices is None: indices = slice(None)
##    imgNum = len(flist[indices])
##    failed = []
##    for i,f in enumerate(flist[indices]):
##        print 'processing (img %d/%d):' %(i+1,imgNum), f.filename
##        try:
##            image_pipeline(f, **kargs)
##        except Exception as e:
##            _print_error(e)
##            failed.append((f,e))
##            
##    return failed

       
# image segmentation
# ------------------
##def _save_segment_image(filename, rmask, bbox):
##    # save the mask, and bbox
##    bb = [(bbox[0].start,bbox[0].stop),(bbox[1].start,bbox[1].stop)]
##    _Image(rmask).save(filename, dtype='uint8', scale=255, pnginfo=dict(bbox=bb))
##    
##def _load_segment_image(filename):
##    rmask = _Image(filename, dtype=bool)
##    bbox  = rmask.info.pop('bbox')
##    bbox = literal_eval(bbox)
##    bbox = map(lambda x: slice(*x),bbox)
##    return rmask, bbox
##    
##@_savable_node('rmask', 'bbox',
##    save_fct=_save_segment_image,
##    load_fct=_load_segment_image,
##    suffix='_mask.png',
##    hidden=['min_dimension','smooth', 'verbose'])
@_node('rmask','bbox', hidden=['min_dimension','smooth', 'verbose'])
def segment_image(image, pmask, root_max_radius=15, min_dimension=50, smooth=1, verbose=False):
    #pmask = _nd.binary_erosion(pmask==pmask.max(), iterations=
    pmask = pmask==pmask.max()
    
    # find the bounding box, and crop image and pmask
    bbox  = _nd.find_objects(pmask)[0]
    pmask = pmask[bbox]
    img   = image[bbox]
    
    if smooth:
        smooth_img  = _nd.gaussian_filter(img*pmask, sigma=smooth)
        smooth_img /= _np.maximum(_nd.gaussian_filter(pmask.astype('f'),sigma=smooth),2**-10)
        img[pmask]  = smooth_img[pmask]
        
    # background removal
    _print_state(verbose,'remove background')
    img = _remove_background(img, distance=root_max_radius, smooth=1)
    img *= _nd.binary_erosion(pmask,iterations=root_max_radius)
    
    # image binary segmentation
    _print_state(verbose,'segment binary mask')
    rmask = _segment_root(img)
    rmask[-pmask] = 0
    if min_dimension>0:
        cluster = _nd.label(rmask)[0]
        cluster = _clean_label(cluster, min_dim=min_dimension)
        rmask = cluster>0
    
    # config root mask serialization
    rmask = rmask.view(_Image)
    rmask.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=255)
    
    return rmask, bbox
                                         
    
# detect leaves:
# --------------
##def _save_detect_leaves(filename, seed_map):
##    _Image(seed_map).save(filename, dtype='uint8', scale=25)  ## max 10 plants!
##def _load_detect_leaves(filename):
##    return _Image(filename, dtype='uint8', scale = 1./25), 
##    
##@_savable_node('rmask', 'bbox',
##    save_fct=_save_detect_leaves,          
##    load_fct=_load_detect_leaves,
##    suffix='_seed.png',
##    hidden=['sort'])
@_node('seed_map', hidden=['sort'])
def detect_leaves(rmask, image, bbox, plant_number=1, root_min_radius=3, leaf_height=[0,.2], sort=True):
    seed_map = _detect_leaves(mask=rmask, image=image[bbox], leaf_number=plant_number, root_radius=root_min_radius, leaf_height=leaf_height, sort=sort) ##
    seed_map = seed_map.view(_Image)
    seed_map.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=255/plant_number)
    return seed_map
        

@_pipeline([load_image,    _node.copy(detect_petri_plate,name='detect_frame'), 
            segment_image, detect_leaves,
            compute_graph, compute_tree])
def pipeline(): pass
# pipeline of all root image analysis modules
#@_pipeline_node([frame_detection, image_segmentation, leaves_detection, root_graph, root_tree])
#def pipeline(): pass                             


#@_pipeline_node([frame_detection2, image_segmentation, leaves_detection, root_graph, root_tree])
#def pipeline2(): pass

