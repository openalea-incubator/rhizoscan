from cv2 import erode as _erode
from sklearn.cluster import KMeans as _Kmeans

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


from cv2 import erode as _erode
from sklearn.cluster import KMeans as _Kmeans

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


def load_image(filename, image_roi=None, *args, **kargs):
    image,op = _normalize_image(_Image(filename,dtype='f',color='gray'))
    image.__serializer__.post_op = op
    if image_roi:
        if isinstance(image_roi,slice):
            roi = [roi]*image.ndim
        image = image[roi]
    return image
    
    
    
# Petri plate detection, function and node

def detect_petri_plate(image, border_width=.05, plate_size=140, plate_shape='square', fg_smooth=7, gradient_classes=(2,1)):
    # binaray segmentation of image
    fg_mask = _plate.detect_foreground(image=image, smooth=fg_smooth, gradient_classes=gradient_classes)
    
    # Find petri plate in foreground mask
    pmask, px_scale, hull = _plate.detect_petri_plate(fg_mask=fg_mask, border_width=border_width,
                                                     plate_size=plate_size, plate_shape=plate_shape)
    
    # set serialization parameter of output petri mask
    pmask = pmask.view(_Image)
    pmask.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=85)
    
    return pmask, px_scale, hull



def detect_marked_plate(image, border_width=0.03, plate_size=140, marker_threshold=0.6, marker_min_size=100):
    pmask, px_scale, hull = _detect_marked_plate(image=image, border_width=border_width, plate_size=plate_size, marker_threshold=marker_threshold, marker_min_size=marker_min_size)
    
    # set serialization parameter of output petri mask
    pmask = pmask.view(_Image)
    pmask.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=85)
    
    return pmask, px_scale, hull
    
@_node('pmask', 'px_scale', 'hull')
def no_plate_to_detect(image, px_scale=1):
    """to replace detect_*_plate when image contain no plate to detect """
    #pmask = _np.zeros(image.shape,dtype='uint8').view(_Image)
    #pmask.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=85)
    
    h,w = image.shape
    hull = [[0,0],[0,w],[w,h],[h,0]]
    
    return None,px_scale,hull
    




# image segmentation
# ------------------

def segment_image(image, pmask=None, root_max_radius=15, min_dimension=50, smooth=1, verbose=False):
    
    
    
    
    return rmask, bbox


def detect_leaves_with_kmeans(image,
                              bounding_box=None,
                              erode_iteration=0,
                              plant_number=5):

    image = image.astype(_np.int8)

    if bounding_box is not None:
        wbound = [bounding_box[0], bounding_box[2]]
        hbound = [bounding_box[1], bounding_box[3]]
        wbound = [int(w * image.shape[1]) for w in wbound]
        hbound = [int(h * image.shape[0]) for h in hbound]
        image[:, :wbound[0]] = 0
        image[:, wbound[1]:] = 0
        image[:hbound[0], :] = 0
        image[hbound[1]:, :] = 0

    if erode_iteration > 0:
        kernel = _np.ones((3, 3), _np.uint8)
        image = _erode(image, kernel, iterations=erode_iteration)

    x, y = _np.where(image > 0)

    pts = list()
    for xx, yy in zip(x, y):
        pts.append((xx, yy))

    pts = _np.array(pts, dtype=float)

    kmeans = _Kmeans(n_clusters=plant_number).fit(pts)
    label = kmeans.labels_

    pts = pts.astype(int)

    for (x, y), label in zip(list(pts), label):
        image[x, y] = int(label) + 1

    return image

# detect leaves:
# --------------
@_node('seed_map', OA_hide=['sort'])
def detect_leaves(rmask, image, bbox, plant_number=1, root_min_radius=3, leaf_bbox=[.05,.05,.95,.2], sort=True):
    seed_map = _detect_leaves(mask=rmask, image=image[bbox], leaf_number=plant_number, root_radius=root_min_radius, leaf_bbox=leaf_bbox, sort=sort) ##
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

