import numpy as _np
from scipy import ndimage as _nd

from rhizoscan.workflow import node as _node # to declare workflow nodes
from rhizoscan.image         import Image as _Image
from rhizoscan.datastructure import Data  as _Data

from .dataset import make_dataset as _make_dataset
from . import _print_state, _print_error
from . import _normalize_image

from ..image import segment_root_in_circle_frame as _segment_root
from ..image.seed import detect_seeds            as _detect_seeds
from ..image.to_graph import linear_label        as _linear_label
from ..image.to_graph import image_graph                 as _image_graph
from ..image.to_graph import line_graph                  as _line_graph
from ..graph          import RootAxialTree               as _RootAxialTree 

@_node('failed_files')
def process(ini_file, indices=None, **kargs):
    if isinstance(ini_file, basestring):
        flist, invalid, outdir = _make_dataset(ini_file=ini_file, output='tree')
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
    

@_node('mask', 'tree')
def image_pipeline(image, seed_min_radius, circle_number, pixel_size, min_dimension, plant_number, to_tree, to_axe, smooth=1, seed_height=[.35,.65], metadata=None, output=None, update=[], verbose=True):
    import os
    
    if all(map(hasattr,(image,)*3, ('filename', 'metadata', 'output'))):
        metadata = image.metadata
        output   = image.output
        image    = image.filename
    
    if output is not None:
        out_mask = output+'_mask.png'
        out_seed = output+'_seed.png'
        out_tree = output+'_b.tree'
    else:
        update = ['all']
        
    # segment image:
    # --------------
    mask = None
    if 'mask' not in update and 'all' not in update and os.path.exists(out_mask):
        # try to load mask
        from ast import literal_eval 
        import PIL
        try:
            info = PIL.Image.open(out_mask).info
            bbox = literal_eval(info['bbox'])
            bbox = map(lambda x: slice(*x),bbox)
            T    = literal_eval(info['T'])
            T    = _np.array(T)
            mask = _Image(out_mask,dtype=bool)      
        except:
            _print_error('Error while loading mask: unreadable file or missing/invalid metadata')  ## print error line ?
        
    if mask is None:
        update = ['all']  # from here on, update everything
        
        # load image
        if isinstance(image,basestring):
            _print_state(verbose, 'load image file')
            image = _normalize_image(_Image(image,dtype='f',color='gray')[::2,::2]) ## sliced !!
            
        if smooth:
            img = _nd.gaussian_filter(image, sigma=smooth)
        else:
            img = image                                                              
        
        ## background removal
        #_print_state(verbose,'remove background')
        #img = _remove_background(img, distance=root_max_radius, smooth=1)
        
        # image binary segmentation
        _print_state(verbose,'segment binary mask')
        # convert to uint8 dtype for segmentation not to fail as a consequence of hardware filtering
        img = ((256/img.max())*img).astype('uint8')
        cluster,T,bbox = _segment_root(img,n=circle_number, pixel_size=pixel_size, min_dimension=min_dimension)
        mask = cluster>0  ## should it be in segmentation function ? not really... 
        
        if output is not None:   
            from PIL.PngImagePlugin import PngInfo
            dir_mask = os.path.dirname(out_mask)
            if len(dir_mask) and not os.path.exists(dir_mask):
                os.makedirs(dir_mask)
            meta = PngInfo()
            meta.add_text('bbox', repr([(bbox[0].start,bbox[0].stop),(bbox[1].start,bbox[1].stop)])) 
            meta.add_text('T',    repr(T.tolist())) 
            _Image(mask).save(out_mask, dtype='uint8', scale=255, pnginfo=meta)
        
    # detect seed:
    # ------------
    seed_map = None
    seed_save_scale = 36  ## jsut for visualisation, auto scaling should be done, with clean_label at loading
    if 'seed' not in update and 'all' not in update and os.path.exists(out_seed):
        try:
            seed_map = _Image(out_seed, dtype='uint8', scale = 1./seed_save_scale)
        except:
            _print_error('Error while loading seed_map file') 
        
    if seed_map is None:
        update = ['all']  # from here on, update everything
        
        _print_state(verbose,'detect seeds')
        seed_map = _detect_seeds(mask=mask, seed_number=plant_number, radius_min=seed_min_radius, seed_height=seed_height, sort=True)
    
        if output is not None:   
            _Image(seed_map).save(out_seed, dtype='uint8', scale=36)
            
    # compute graph:
    # --------------
    pl_graph = None
    if 'graph' not in update and 'all' not in update and os.path.exists(out_tree):  ## only tree is saved
        try:
            pl_graph = _Data.load(out_tree)
        except:
            _print_error('Error while loading graph file (actually, the tree file which stores the graph)') 
            
    if pl_graph is None:
        update = ['all']  # from here on, update everything
        
        # image linear decomposition
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

    # extract axial tree:
    # -------------------
    tree = None
    if 'tree' not in update and 'all' not in update and os.path.exists(out_tree):
        try:
            tree = _Data.load(out_tree)
        except:
            _print_error('Error while loading tree file') 
    
    if tree is None:
        _print_state(verbose,'extract axial tree')
        tree = _RootAxialTree(node=pl_graph.node, segment=pl_graph.segment, to_tree=to_tree, to_axe=to_axe, single_order1_axe=False)

        if metadata is not None: tree.metadata = metadata
        if output is not None:   tree.dump(out_tree)

    return tree
    
    
    
def rice_test0(filename=None):
    import ConfigParser as cfg
    
    if filename is None:
        filename = '/Users/diener/root_data/perin/test0/test0.ini'
        
    ini = cfg.ConfigParser()
    ini.read(filename)
    ini = dict([(s,_Mapping(**dict(ini.items(s)))) for s in ini.sections()])

    out = [] # to store output

    for name in ['test_old']:#ini.keys():##
        data = ini[name]
        
        mfile = data.mask
        tfile = data.tree
        
        param = dict([(k[6:],_param_eval(v)) for k,v in data.iteritems() if k[:6]=='param_'])
        
        image = _Image(data.image)#,dtype='f',color='gray')
        if hasattr(data,'sample'):
            sl = slice(None,None,_param_eval(data.sample))
            image = image[sl,sl]
        #image = _normalize_image(image)

        print "  *** processing image '%s': ***" % data.image
        print param
        mask, tree = image_pipeline_no_frame(image=image, **param)
        
        res = _Mapping()
        res.mask = mask#_Image(mask).save(mfile, dtype='uint8', scale=255)
        res.tree = tree#tree.dump(tfile)
        
        out.append(res)
        
    return out
    
