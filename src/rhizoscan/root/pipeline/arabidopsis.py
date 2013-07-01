import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes

from rhizoscan.ndarray.measurements import clean_label as _clean_label
from rhizoscan.ndarray.filter       import otsu as _otsu
from rhizoscan.image    import Image as _Image
from rhizoscan.workflow import Data  as _Data

from .database import parse_image_db as _parse_image_db
from . import PipelineModule   as _PModule
from . import pipeline_node    as _pipeline_node
from . import load_root_image
from . import compute_tree   as _compute_tree
from . import _print_state, _print_error
from . import _normalize_image

from ..image import segment_root_image           as _segment_root
from ..image import remove_background            as _remove_background
from ..image.seed import detect_leaves           as _detect_leaves
from ..image.to_graph import linear_label        as _linear_label
from ..image.to_graph import image_graph         as _image_graph
from ..image.to_graph import line_graph          as _line_graph


##@_aleanode('failed_files')
##def process(ini_file, indices=None, **kargs):
##    if isinstance(ini_file, basestring):
##        flist, invalid, outdir = _parse_image_db(ini_file=ini_file, output='output')
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

    
# detect plate: and pre-process input image
# -------------
def mask_fillhull(mask):
    """ find mask hull, and return image with filled hull image and hull array"""
    from scipy import spatial, sparse
    px = _np.transpose((mask<>_nd.uniform_filter(mask,size=(3,3))).nonzero())
    
    # compute hull (sort indices using csgraph stuff)
    hull = spatial.Delaunay(px).convex_hull
    graf = sparse.csr_matrix((_np.ones(hull.shape[0]),hull.T), shape=(hull.max()+1,)*2)
    hull = sparse.csgraph.depth_first_order(graf,hull[0,0],directed=False)[0]    
    hull = px[hull]

    ##todo: fit quad plate and transform
    
    # compute (out of) petri plate mask
    from PIL import Image, ImageDraw
    plate = Image.new('L', mask.shape[::-1], 0)
    ImageDraw.Draw(plate).polygon([tuple(p) for p in hull[:,::-1]], fill=1)
    plate = _np.array(plate)
    
    return plate, hull
    
def find_plate(filename, image, plate_width=120, threshold=0.6, marker_min_size=100, codebar=True):
    mask    = image>threshold
    cluster = _clean_label(_nd.label(mask)[0], min_dim=marker_min_size)

    # find plate mask and hull
    pmask,phull = mask_fillhull(cluster>0)
    pwidth = pmask.sum()**.5                                 # estimated plate width in pixels
    border = pwidth * .03                                    ## constant 3% of plate width
    pmask  = pmask + 2*_nd.binary_erosion(pmask>0, iterations=int(border))
    px_ratio = plate_width/pwidth
    
    # detect codebar box as the biggest connex cluster
    if codebar:
        cbmask = _nd.label(_nd.binary_closing((cluster>0) & (pmask==3), iterations=5))[0]
        obj = _nd.find_objects(cbmask)
        cbbox = _np.argmax([max([o.stop-o.start for o in ob]) for ob in obj])+1
    
        # find codebar mask and hull
        cbmask,cbhull = mask_fillhull(cbmask==cbbox)

        # stack masks such that 
        #   pixels to process = 3,
        #   codebar pixels = 2
        #   plate border = 1
        pmask[cbmask>0] = 2

    # save plate mask
    _Image(pmask).save(filename, dtype='uint8', scale=85, pnginfo=dict(px_ratio=px_ratio)) # 85 = 255/pmask.max()
    
    return pmask, px_ratio

def find_plate_2(filename, image, plate_width=120, plate_border=100, codebar=False):
    # compute laplacian with radius relative to width
    #   then threshold it
    lp = -_nd.laplace(_nd.gaussian_filter(image,sigma=plate_border/3))
    mask = lp>_otsu(lp,step=100)
    
    # find plate mask and hull
    pmask,phull = mask_fillhull(mask)
    pwidth = pmask.sum()**.5                                 # estimated plate width in pixels
    #border = pwidth * .03                                    ## constant 3% of plate width
    pmask  = pmask + 2*_nd.binary_erosion(pmask>0, iterations=int(plate_border))
    px_ratio = plate_width/pwidth
    
    # detect codebar box as the biggest connex cluster
    if codebar:
        cbmask = _nd.label(_nd.binary_closing((cluster>0) & (pmask==3), iterations=5))[0]
        obj = _nd.find_objects(cbmask)
        cbbox = _np.argmax([max([o.stop-o.start for o in ob]) for ob in obj])+1
    
        # find codebar mask and hull
        cbmask,cbhull = mask_fillhull(cbmask==cbbox)
        pmask[cbmask>0] = 2

    # save plate mask
    _Image(pmask).save(filename, dtype='uint8', scale=85, pnginfo=dict(px_ratio=px_ratio)) # 85 = 255/pmask.max()
    
    return pmask, px_ratio
    
def load_plate_mask(filename):
    from ast import literal_eval 
    import PIL
    info = PIL.Image.open(filename).info
    px_ratio = literal_eval(info['px_ratio'])
    pmask = _Image(filename,dtype='uint8')/85
    #px_ratio = literal_eval(pmask.info['px_ratio'])
    return pmask, px_ratio

frame_detection = _PModule(name='frame', \
                           load=load_plate_mask,  compute=find_plate, \
                           suffix='_frame.png',   outputs=['pmask','px_ratio'])    
    
frame_detection2 = _PModule(name='frame', \
                           load=load_plate_mask,  compute=find_plate_2, \
                           suffix='_frame.png',   outputs=['pmask','px_ratio'])    
    
# image segmentation
# ------------------
def segment_image(filename, image, pmask, root_max_radius=15, min_dimension=50, smooth=1, verbose=False):
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
    
    # save the mask, and bbox
    bb = [(bbox[0].start,bbox[0].stop),(bbox[1].start,bbox[1].stop)]
    _Image(rmask).save(filename, dtype='uint8', scale=255, pnginfo=dict(bbox=bb))
    
    return rmask, bbox
    
def load_root_mask(filename):
    from ast import literal_eval 
    import PIL
    info = PIL.Image.open(filename).info
    bbox = literal_eval(info['bbox'])
    bbox = map(lambda x: slice(*x),bbox)
    return _Image(filename, dtype=bool), bbox

image_segmentation = _PModule(name='segmentation', \
                              load=load_root_mask, compute=segment_image, \
                              suffix='_mask.png',   outputs=['rmask','bbox'],\
                              hidden=['min_dimension','smooth'])
    
    
# detect leaves:
# --------------
def detect_leaves(filename, rmask, image, bbox, plant_number=1, root_min_radius=3, leaf_height=[0,.2], sort=True):
    seed_map = _detect_leaves(mask=rmask, image=image[bbox], leaf_number=plant_number, root_radius=root_min_radius, leaf_height=leaf_height, sort=sort) ##
    _Image(seed_map).save(filename, dtype='uint8', scale=25)  ## max 10 plants
    
    return seed_map, 

def load_leaves(filename):
    return _Image(filename, dtype='uint8', scale = 1./25), 
    
leaves_detection = _PModule(name='leaves', \
                            load=load_leaves,   compute=detect_leaves, \
                            suffix='_seed.png', outputs=['seed_map'], hidden=['sort'])
    
# compute graph:
# --------------
def compute_graph(filename, rmask, seed_map, bbox, verbose=False):
    _print_state(verbose,'compute mask linear decomposition')
    sskl, nmap, smap, seed = _linear_label(mask=rmask, seed_map=seed_map, compute_segment_map=True)
    
    # make "image-graph"
    _print_state(verbose,'compute graph representation of mask decomposition')
    im_graph = _image_graph(segment_skeleton=sskl, node_map=nmap, segment_map=smap, seed=seed)
    
    # make polyline graph
    _print_state(verbose,'compute graph of roots')
    pl_graph = _line_graph(image_graph=im_graph, segment_skeleton=sskl)    
    
    # shift graph node position by cropped box left corner
    pl_graph.node.x[:] += bbox[1].start
    pl_graph.node.y[:] += bbox[0].start
    pl_graph.node.position[:,0] = 0
    
    pl_graph.save(filename)
    
    return pl_graph, 
    
def load_graph(filename, tree=False):
    graph = _Data.load(filename)
    if tree and not hasattr(graph, 'axe'):
        return None # induce computing but no error message
        #raise IOError('loaded graph is not a tree')
        
    return graph, 
        
root_graph = _PModule(name='graph', \
                      load=load_graph,  compute=compute_graph,  \
                      suffix='.tree',   outputs=['graph'],      \
                      load_kargs=dict(tree=False))##, hidden=['tree'])

# compute axial tree:
# -------------------
def compute_tree(filename, graph, px_ratio, to_tree=2, to_axe=2, metadata={}, verbose=False):
    metadata['px_ratio'] = px_ratio
    tree = _compute_tree(graph, to_tree=to_tree, to_axe=to_axe, metadata=metadata, output_file=filename, verbose=verbose) 
    return tree, 
    
root_tree = _PModule(name='tree', \
                     load=load_graph,   compute=compute_tree, \
                     suffix='.tree',    outputs=['tree'],      \
                     load_kargs=dict(tree=True), hidden=['tree','to_tree','to_axe','metadata','px_ratio'])

# pipeline of all root image analysis modules
@_pipeline_node([frame_detection, image_segmentation, leaves_detection, root_graph, root_tree])
def pipeline(): pass


@_pipeline_node([frame_detection2, image_segmentation, leaves_detection, root_graph, root_tree])
def pipeline2(): pass

