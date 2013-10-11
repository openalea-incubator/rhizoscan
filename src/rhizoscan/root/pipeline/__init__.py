import numpy as _np
from ast import literal_eval as _literal_eval

from rhizoscan.workflow import node as _node # to declare workflow nodes

from rhizoscan.image       import Image as _Image
from rhizoscan.root.image  import normalize_image as _normalize_image 
from rhizoscan.root.image  import plate           as _plate
from rhizoscan.root.graph  import RootAxialTree   as _RootAxialTree 

from rhizoscan.root.dev_graph2axial import make_axial_tree as _make_axial_tree 


def _print_state(verbose, msg):
    if verbose: print '  ', msg
def _print_error(msg):
    print '  \033[31m*** %s *** \033[30m' % repr(msg)

def _param_eval(string):
    """ safe literal eval """
    try:     return _literal_eval(string)
    except:  return string


# load image function and node
# ----------------------------
@_node('image')
def load_image(filename, image_roi=None, *args, **kargs):
    image = _normalize_image(_Image(filename,dtype='f',color='gray'))
    if image_roi:
        if isinstance(image_roi,slice):
            roi = [roi]*image.ndim
        image = image[roi]
    return image


# petri plate detection, function and node
# ----------------------------------------
##def _save_detect_petri_plate(filename, pmask, px_scale, hull):
##    pnginfo = dict(px_scale=px_scale, hull=repr(hull.tolist()))
##    _Image(pmask).save(filename, dtype='uint8', scale=85, pnginfo=pnginfo) # 85 = 255/pmask.max()
##    
##def _load_detect_petri_plate(filename):
##    pmask = _Image(filename,dtype='uint8')/85
##    px_scale = pmask.info.pop('px_scale')
##    hull = _np.array(pmask.info.pop('hull'))
##    return pmask, px_scale, hull
##    
##@_node('pmask','px_scale', 'hull', 
##    save_fct=_save_detect_petri_plate,
##    load_fct=_load_detect_petri_plate,
##    suffix='_frame.png',
##    hidden=['plate_shape','smooth', 'gradient_classes'])
@_node('pmask','px_scale', 'hull', hidden=['plate_shape','smooth', 'gradient_classes'])
def detect_petri_plate(image, border_width=.05, plate_size=120, plate_shape='square', smooth=5, gradient_classes=(2,1)):
    # binaray segmentation of image
    fg_mask = _plate.detect_foreground(image=image, smooth=smooth, gradient_classes=gradient_classes)
    
    # Find petri plate in foreground mask
    pmask, px_scale, hull = _plate.detect_petri_plate(fg_mask=fg_mask, border_width=border_width,
                                                     plate_size=plate_size, plate_shape=plate_shape)
    
    # set serialization parameter of output petri mask
    pmask = pmask.view(_Image)
    pmask.set_serializer(pil_format='PNG', ser_dtype='uint8', ser_scale=85)
    
    return pmask, px_scale, hull


# compute graph:
# --------------
from rhizoscan.root.image.to_graph import linear_label as _linear_label
from rhizoscan.root.image.to_graph import image_graph  as _image_graph
from rhizoscan.root.image.to_graph import line_graph   as _line_graph

##def load_graph(filename, tree=False):
##    graph = _Data.load(filename)
##    if tree and not hasattr(graph, 'axe'):
##        return None # induce computing but no error message
##    return graph
##    
##@_savable_node('graph',
##    load_fct=load_graph,  load_kargs=dict(tree=False),
##    suffix='.tree',
##    hidden=['verbose'])
@_node('graph',hidden=['verbose'])
def compute_graph(rmask, seed_map, bbox=None, verbose=False):
    _print_state(verbose,'compute mask linear decomposition')
    sskl, nmap, smap, seed = _linear_label(mask=rmask, seed_map=seed_map, compute_segment_map=True)
    
    # make "image-graph"
    _print_state(verbose,'compute graph representation of mask decomposition')
    im_graph = _image_graph(segment_skeleton=sskl, node_map=nmap, segment_map=smap, seed=seed)
    
    # make polyline graph
    _print_state(verbose,'compute graph of roots')
    graph = _line_graph(image_graph=im_graph, segment_skeleton=sskl)    
    
    # shift graph node position by cropped box left corner
    if bbox:
        graph.node.x[:] += bbox[1].start
        graph.node.y[:] += bbox[0].start
        graph.node.position[:,0] = 0
    
    return graph
    
    
# axial tree extraction from root graph
# -------------------------------------
##@_savable_node('tree',
##    load_fct=load_graph,  load_kargs=dict(tree=True),
##    suffix='.tree',
##    hidden=['to_tree','to_axe','metadata','px_scale', 'verbose'])
@_node('tree', hidden=['axe_selection','metadata','px_scale', 'verbose'])
#@_node('tree', hidden=['to_tree','to_axe','metadata','px_scale', 'verbose'])
def compute_tree(graph, px_scale=1, axe_selection=[('length',1),('min_tip_length',10)], metadata={}):
    #def compute_tree(graph, px_scale=1, to_tree=2, to_axe=2, metadata={}):
    #tree = _RootAxialTree(node=graph.node, segment=graph.segment, to_tree=to_tree, to_axe=to_axe)
    tree = _make_axial_tree(graph=graph, axe_selection=[('length',1),('min_tip_length',10)])
    metadata['px_scale'] = px_scale
    tree.metadata = metadata
    return tree 

