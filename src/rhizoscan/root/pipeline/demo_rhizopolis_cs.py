import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow import node as _node # to declare workflow nodes
from rhizoscan.tool  import static_or_instance_method as _static_or_instance_method

from rhizoscan.ndarray.measurements import clean_label as _clean_label
from rhizoscan.image                import Image as _Image
from rhizoscan.datastructure        import Data  as _Data

from .dataset import make_dataset as _make_dataset
from . import compute_tree   as _compute_tree
from . import _print_state, _print_error
from . import _normalize_image

from ..image import segment_root_image           as _segment_root
from ..image import remove_background            as _remove_background
from ..image.seed import detect_leaves           as _detect_leaves
from ..image.to_graph import linear_label        as _linear_label
from ..image.to_graph import image_graph         as _image_graph
from ..image.to_graph import line_graph          as _line_graph
from ..graph          import RootAxialTree       as _RootAxialTree 


##class Test(object):
##    @_static_or_instance_method
##    def run(self, value):
##        print value
##        return 42
##
##_aleanode('42', name='Test.run', nodeclass='Test.run')(Test.run)

@_node('failed_files')
def process(ini_file, indices=None, **kargs):
    if isinstance(ini_file, basestring):
        flist, invalid, outdir = _make_dataset(ini_file=ini_file, output='output')
    else:
        flist = ini_file
    
    if indices is None: indices = slice(None)
    imgNum = len(flist[indices])
    failed = []
    for i,f in enumerate(flist[indices]):
        print 'processing (img %d/%d):' %(i+1,imgNum), f.filename
        try:
            image_pipeline(f, **kargs)
        except Exception as e:
            _print_error(e)
            failed.append((f,e))
            
    return failed
    
@_node('tree')
def image_pipeline(image, plate_width=120, min_dimension=50, smooth=1, to_tree=2, to_axe=2, seed_height=[0,.25], metadata=None, output=None, update=[], verbose=True):
    import os
    
    if hasattr(image, 'metadata'):  metadata = image.metadata
    if hasattr(image, 'output'):    output   = image.output
    if hasattr(image, 'filename'):  image    = image.filename
    
    plant_number  = int(metadata.plant_number)
    #circle_number = metadata.get('circle_number', circle_number)
    
    # function recompute the data only if necessary
    def get_data(name, out, load, load_args, compute, compute_args={}):
        data = None
        if name not in update and 'all' not in update and os.path.exists(out):
            try:
                data = load(out, **load_args)
            except:
                _print_error('Error while loading %s: unreadable file or missing/invalid metadata' % name)  ## print error line ?
                
        if data is None:
            #update = ['all']  # update all following data ## problem of var scope, maybe not necessary: do it by hand if necessary
            _print_state(verbose, '  computing '+ name)
            data = compute(out, **compute_args)
        else:
            _print_state(verbose, '  > '+ name + ' loaded')
            
        return data
        
    # name of output files
    if output is None:
        update = ['all']
        
    # assert output directory exist
    outdir = os.path.dirname(output)
    if len(outdir) and not os.path.exists(outdir):
        os.makedirs(outdir)
    
    out_frame  = output+'_frame.png'
    out_circle = output+'_circle.png'
    out_mask   = output+'_mask.png'
    out_seed   = output+'_seed.png'
    out_tree   = output+'.tree'
    
    ## load image, woulod be nice to avoid if possible    
    image = _normalize_image(_Image(image,dtype='f',color='gray'))#[::2,::2]) ## sliced !!
        
    # segment image:
    # --------------
    # find circles
    def compute_circle(filename, image, circle_number):
        from ..image.circle import detect_circles
        cmask   = _nd.binary_opening(image>.45,iterations=5)
        cluster = _nd.label(cmask)[0]
        cluster = _clean_label(cluster, min_dim=30)
        
        ind,qual,c,dmap = detect_circles(n=circle_number,cluster=cluster)
        radius = qual**.5/_np.pi
        
        #cind = _np.zeros(c.max()+1)
        #cind[ind] = _np.arange(1,len(ind)+1)
        #cmask = cind[c]
        
        from PIL.PngImagePlugin import PngInfo
        meta = PngInfo()
        meta.add_text('circles', repr(ind.tolist())) 
        meta.add_text('radius',  repr(radius.tolist())) 
        _Image(cluster).save(filename, dtype='uint8', scale=1, pnginfo=meta)
        #_Image(cmask).save(filename, dtype='uint8', scale=1)
        
        return cluster, ind, radius
        
    def load_circle(filename):
        from ast import literal_eval 
        import PIL
        info = PIL.Image.open(filename).info
        cind = literal_eval(info['circles'])
        rad  = literal_eval(info['radius'])
        cmask = _Image(filename,dtype='uint8',scale=1)
        
        return cmask, cind, rad
        
    
    #cmask, ind, radius = get_data('circle', out_circle, load_circle, {}, \
    #                              compute_circle, dict(image=image,circle_number=circle_number))
    
    def mask_fillhull(mask):
        """ find mask hull, and return image with filled hull image and hull array"""
        from scipy import sparse, spatial
        px = _np.transpose(mask.nonzero())
        
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
        
    def find_plate(filename, image, plate_width):
        from ..image.circle import detect_circles
        mask    = image>.6                                       ## constant
        cluster = _clean_label(_nd.label(mask)[0], min_dim=100)  ## cosntant

        # find plate mask and hull
        pmask,phull = mask_fillhull(cluster>0)
        pwidth = pmask.sum()**.5                                 # estimated plate width in pixels
        border = pwidth * .03                                    ## constant 3% of plate width
        pmask  = pmask + 2*_nd.binary_erosion(pmask>0, iterations=int(border))
        px_scale = plate_width/pwidth
        
        # detect codebar box as the biggest connex cluster
        cbmask = _nd.label(_nd.binary_closing((cluster>0) & (pmask==3), iterations=5))[0]
        obj = _nd.find_objects(cbmask)
        cbbox = _np.argmax([max([o.stop-o.start for o in ob]) for ob in obj])+1
        
        # find codebar mask and hull
        cbmask,cbhull = mask_fillhull(cbmask==cbbox)

        # stack masks such that 
        #   pixels to process = 3,
        #   codebar pixels = 2
        #   plate border = 1
        ##mask = pmask + 1*cbmask + 1*_nd.binary_erosion(pmask, iterations=int(border))
        pmask[cbmask>0] = 2

        # save plate mask
        from PIL.PngImagePlugin import PngInfo
        meta = PngInfo()
        meta.add_text('px_scale', repr(px_scale)) 
        _Image(pmask).save(filename, dtype='uint8', scale=85, pnginfo=meta) # 85 = 255/pmask.max()
        
        return pmask, px_scale
        
    def load_plate_mask(filename):
        from ast import literal_eval 
        import PIL
        info = PIL.Image.open(filename).info
        px_scale = literal_eval(info['px_scale'])
        return _Image(filename,dtype='uint8')/85, px_scale
    
    pmask, px_scale = get_data('frame', out_frame, load_plate_mask, {}, \
                                        find_plate, dict(image=image, plate_width=plate_width))
    
    # do the actual image segmentation
    def segment_image(filename, image, mask, root_max_radius=15, min_dimension=50, smooth=1):
        #mask = _nd.binary_erosion(mask==mask.max(), iterations=
        mask = mask==mask.max()
        
        # find the bounding box, and crop image and mask
        bbox = _nd.find_objects(mask)[0]
        mask = mask[bbox]
        img  = image[bbox]
        
        if smooth:
            smooth_img  = _nd.gaussian_filter(img*mask, sigma=smooth)
            smooth_img /= _np.maximum(_nd.gaussian_filter(mask.astype('f'),sigma=smooth),2**-10)
            img[mask] = smooth_img[mask]
            
        # background removal
        _print_state(verbose,'remove background')
        img = _remove_background(img, distance=root_max_radius, smooth=1)
        img *= _nd.binary_erosion(mask,iterations=root_max_radius)
        
        # image binary segmentation
        _print_state(verbose,'segment binary mask')
        rmask = _segment_root(img)
        rmask[-mask] = 0
        if min_dimension>0:
            cluster = _nd.label(rmask)[0]
            cluster = _clean_label(cluster, min_dim=min_dimension)
            rmask = cluster>0
        
        # save the mask, and bbox
        from PIL.PngImagePlugin import PngInfo
        meta = PngInfo()
        meta.add_text('bbox', repr([(bbox[0].start,bbox[0].stop),(bbox[1].start,bbox[1].stop)])) 
        _Image(rmask).save(filename, dtype='uint8', scale=255, pnginfo=meta)
        
        return rmask, bbox
        
    def load_root_mask(filename):
        from ast import literal_eval 
        import PIL
        info = PIL.Image.open(out_mask).info
        bbox = literal_eval(info['bbox'])
        bbox = map(lambda x: slice(*x),bbox)
        return _Image(filename, dtype=bool), bbox
        
    mask, bbox = get_data('mask', out_mask, load_root_mask, {}, \
                    segment_image, dict(image=image,mask=pmask, smooth=smooth))
                    # root_max_radius=15, min_dimension=50):
        
    # detect leaves:
    # --------------
    def find_leaves(filename, mask, image, plant_number, root_radius=3, leaf_height=[0,.2], sort=True):
        seed_map = _detect_leaves(mask=mask, image=image, leaf_number=plant_number, root_radius=root_radius, leaf_height=leaf_height, sort=sort) ##
        _Image(seed_map).save(out_seed, dtype='uint8', scale=25)  ## max 10 plants
        
        return seed_map
    
    def load_leaves(filename):
        return _Image(filename, dtype='uint8', scale = 1./25)
        
    seed_map = get_data('seed', out_seed, load_leaves, {}, \
        find_leaves, dict(mask=mask,image=image[bbox],plant_number=plant_number, root_radius=3, leaf_height=[0,.2]))
    
    # compute graph:
    # --------------
    def compute_graph(filename, mask, seed_map):
        _print_state(verbose,'compute mask linear decomposition')
        sskl, nmap, smap, seed = _linear_label(mask=mask, seed_map=seed_map, compute_segment_map=True)
        
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
        
        pl_graph.dump(filename)
        
        return pl_graph
        
    def load_graph(filename, tree):
        graph = _Data.load(filename)
        if tree and not hasattr(graph, 'axe'):
            return None # induce computing but no error message
            #raise IOError('loaded graph is not a tree')
            
        return graph
            
    graph = get_data('graph', out_tree, load_graph, dict(tree=False), \
                     compute_graph, dict(mask=mask, seed_map=seed_map))
    
    # compute axial tree:
    # -------------------
    def compute_tree(filename, graph, to_tree, to_axe, metadata, px_scale, verbose):
        metadata.px_scale = px_scale
        tree = _compute_tree(graph, to_tree=to_tree, to_axe=to_axe, metadata=metadata, output_file=filename, verbose=verbose) 
        return tree
        
    tree = get_data('tree', out_tree, load_graph, dict(tree=True), \
                     compute_tree, dict(graph=graph, to_tree=to_tree, to_axe=to_axe, metadata=metadata, px_scale=px_scale, verbose=verbose))

    return tree
