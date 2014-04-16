import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow import node as _node         # declare workflow nodes
from rhizoscan.workflow import pipeline as _pipeline # declare workflow pipeline

from rhizoscan.ndarray.measurements import clean_label as _clean_label
from rhizoscan.image                import Image       as _Image

from . import load_image
from . import detect_petri_plate
from . import detect_marked_plate
from . import no_plate_to_detect
from . import compute_graph, compute_tree
from . import _print_state, _print_error

from rhizoscan.root.image import segment_root_image  as _segment_root
from rhizoscan.root.image import remove_background   as _remove_background
from rhizoscan.root.image.seed import detect_leaves  as _detect_leaves
from rhizoscan.root.graph.mtg  import tree_to_mtg


# image segmentation
# ------------------
@_node('rmask','bbox', OA_hide=['min_dimension','smooth', 'verbose'])
def segment_image(image, pmask=None, root_max_radius=15, min_dimension=50, smooth=1, verbose=False):
    if pmask is not None:
        pmask = pmask==pmask.max()
    
        # find the bounding box, and crop image and pmask
        bbox  = _nd.find_objects(pmask)[0]
        img   = image[bbox]
        pmask = pmask[bbox]
    else:
        img  = image
        bbox = map(slice,[0,0],image.shape)
    
    if smooth:
        if pmask is None:
            img  = _nd.gaussian_filter(img, sigma=smooth)
        else:
            smooth_img  = _nd.gaussian_filter(img*pmask, sigma=smooth)
            smooth_img /= _np.maximum(_nd.gaussian_filter(pmask.astype('f'),sigma=smooth),2**-10)
            img[pmask]  = smooth_img[pmask]
        
    # background removal
    _print_state(verbose,'remove background')
    img = _remove_background(img, distance=root_max_radius, smooth=1)
    if pmask is not None:
        img *= _nd.binary_erosion(pmask,iterations=root_max_radius)
    
    # image binary segmentation
    _print_state(verbose,'segment binary mask')
    rmask = _segment_root(img)
    if pmask is not None:
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
@_node('seed_map', OA_hide=['sort'])
def detect_leaves(rmask, image, bbox, plant_number=1, root_min_radius=3, leaf_height=[0,.2], sort=True):
    seed_map = _detect_leaves(mask=rmask, image=image[bbox], leaf_number=plant_number, root_radius=root_min_radius, leaf_height=leaf_height, sort=sort) ##
    seed_map = seed_map.view(_Image)
    seed_map.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=255/plant_number)
    return seed_map
        

@_pipeline([load_image,    _node.copy(detect_petri_plate,name='detect_frame'), 
            segment_image, detect_leaves,
            compute_graph, compute_tree, tree_to_mtg])
def pipeline(): pass

@_pipeline([load_image,    _node.copy(detect_marked_plate,name='detect_frame'), 
            segment_image, detect_leaves,
            compute_graph, compute_tree, tree_to_mtg])
def pipeline_marked_frame(): pass

@_pipeline([load_image,    _node.copy(no_plate_to_detect,name='detect_frame'), 
            segment_image, detect_leaves,
            compute_graph, compute_tree, tree_to_mtg])
def pipeline_no_frame(): pass

