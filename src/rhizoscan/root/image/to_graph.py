"""
##to put all image-to-graph conversion stuff?
    image.linear_map
    graph.RootGraph.init_from_maps  (and subfunctions)
    graph.RootGraph.compute_polyline_graph
"""
import numpy as _np
from scipy import ndimage as _nd

from rhizoscan.ndarray              import virtual_array   as _virtual_array     
from rhizoscan.ndarray.measurements import clean_label     as _clean_label     
from rhizoscan.ndarray.measurements import label_size      as _label_size     
from rhizoscan.image.measurements   import corner_count    as _corner_count
from rhizoscan.image.measurements   import skeleton_label  as _skeleton_label     
from rhizoscan.image.measurements   import curve_to_spline as _curve_to_spline     

from ..graph import RootGraph   as _RootGraph
from ..graph import NodeList    as _NodeList
from ..graph import SegmentList as _SegmentList

from rhizoscan.workflow.openalea import aleanode as _aleanode

@_aleanode('segment_map', 'segment_skeleton','node_map', 'seeds_structure')
def linear_label(mask, seed_map=None, compute_segment_map=True):
    """
    Partition mask as a set of linear segments
    
    Input:
    ------
        mask: 
            a binary map of linear structure such as it is return by 
            segment_root_* functions
        seed_map:
            optional map of seed map - an integer area of contiguous labels. 
            Seed area are removed from output maps, but additional nodes are 
            added at cossing of skeleton and seed area. Also fill 'seed' output.
        compute_segment_map:
            If True, the returned segment_map is the input mask labeled with the
            computed segment_skeleton id.
        
    Output:
    -------
        segment_skl:  (*)
            The map of segments pixels of the skeleton
        node_map:     (*)
            The map of nodes    pixels of the skeleton
        segment_map:  (*) 
            The label map of all linear segment - cluster divided in segments
        seed: 
            None if seed_map is None
            Otherwise return a dictionary which contains the following items: 
              'position': the y-x coordinates of the seed center
              'nodes':    the list of nodes id (in node_map) at the seed area
                          border (in no particular order)
              'sid':      the seed id of all these nodes
           
    (*) the labels are contiguous integer starting at 1 (background is 0)
    
    *** Require scikits.image ***
    """
    # compute labeled skeleton  
    segment_skl,node_map = _skeleton_label(mask, closing=0, fill_node=True, terminal=True)[:2]
    
    if seed_map is not None:
        # remove segment and node from seed area
        seed_mask = seed_map>0
        segment_skl[seed_mask] = 0                 
        node_map[seed_mask] = 0
            # make it contiguous labels
        tmp = segment_skl>0
        segment_skl[tmp] = _clean_label(segment_skl[tmp])
        tmp = node_map>0
        node_map[tmp]    = _clean_label(node_map[tmp])
        
        # add nodes at seed area border
        dil_smap = _nd.grey_dilation(seed_map,size=(3,3))
        ny,nx = ((dil_smap>0)&(segment_skl>0)).nonzero()
        nid   = node_map.max() + _np.arange(1,ny.size+1)
        node_map[ny,nx] = nid
        segment_skl[ny,nx] = 0
        ## new node don't touch existing nodes?
        #  two connected new node is not a problem
        ## any other unexpected error ?
        
        # create output seed structure
        seed = dict()
            # compute seed  positions
        obj  = _nd.find_objects(seed_map)
        spos = _np.array([[(sl.stop+sl.start)/2. for sl in o] for o in obj])
        seed['position'] = spos
        seed['nodes'] = nid
        seed['sid']   = dil_smap[ny,nx]
        
        if compute_segment_map:
            mask = mask - seed_mask
    else:
        seed = None
    
    if compute_segment_map:
        # compute segment map by dilation of segment_skeleton into mask
        segment_map = _nd.distance_transform_edt(segment_skl==0,return_indices=True,return_distances=False)
        segment_map = segment_skl[tuple(segment_map)] * (mask)
    else:
        segment_map = None

    return segment_skl, node_map, segment_map, seed


@_aleanode('image_graph')
def image_graph(segment_skeleton, node_map, segment_map=None, seed=None):
    """
    Construct a RootGraph representing a "linear map" image
    
    :Input: 
        same as output of linear_map.
        If segment_map is not None, the segmentList of the returned graph
        contains the attribut 'area' and 'radius'
        If seed is not None, the seed 'position' is added to the graph nodes as
        well as the segments from the seed to all its 'node'. The NodeList and 
        SegmentList of the graph also have an attribute 'seed' containing the 
        suitable seed id for each node/segment (0 for non-seed)
    
    :Output: 
        a RootGraph where the node are taken from nmap (node_map) and segment 
        from smap (segment_map).  
    
    :See also:
        root.graph.RootGraph
        root.graph.SegmentList
        root.graph.NodeList
    """
    # To simplify coding 
    smap = segment_map
    nmap = node_map
    sskl = segment_skeleton

    # Contruct a NodeList from nmap with optional seeds
    # -------------------------------------------------
    obj = [o if o is not None else (slice(0,0),)*2 for o in _nd.find_objects(nmap)]
    x = [(s[1].start+s[1].stop-1)/2. for s in obj]
    y = [(s[0].start+s[0].stop-1)/2. for s in obj]
    nodeNum = len(obj)
    
    # first node is a dummy node
    if seed is not None:
        # also add seed nodes
        seedNum  = seed['position'].shape[0]  # number of seeds
        position = _np.zeros((2,1+nodeNum+seedNum))
        position[0,1:nodeNum+1] = x
        position[1,1:nodeNum+1] = y
        position[:,nodeNum+1:] = seed['position'].T[::-1] # seed position
        
        nseed = _np.zeros(1+nodeNum+seed['position'].shape[0],dtype=int)
        ## nseed[seed['nodes']+1] = seed['sid'] # seed id for seed border nodes?
        nseed[1+nodeNum:] = _np.arange(1,seedNum+1)
        
        node = _NodeList(position=position)
        node.seed = nseed
        
        # id of seed node w.r.t seed label 
        seed_nid = _np.zeros(seedNum+1, dtype=int)
        seed_nid[1:] = _np.arange(seedNum) + node.seed.size-seedNum
    else:
        position = _np.zeros((2,1+nodeNum))
        position[0,1:] = x
        position[1,1:] = y
        node = _NodeList(position=position)
        
    
    segNum = smap.max()
    
    # List all segments in neighborhood of all nodes
    # ----------------------------------------------
    # create a set ns_set of all (node,segment) pairs
    
    #   First pass, dilate node map and look for intersection with segment map
    n_dil = _nd.grey_dilation(nmap,size=(3,3))
    mask = (n_dil!=0) & (sskl!=0)
    ns_set = set([(n,s) for n,s in zip(n_dil[mask],sskl[mask])])
    
    #   Second pass, dilation is done with priority to node with lower id
    #     (necessary for cases where a 1 pixel segment touches 2 nodes)
    skn    = nmap.copy()
    skn[skn==0] = node.size+1
    n_dil  = -_nd.grey_dilation(-skn,size=(3,3))
    n_dil[n_dil==n_dil.max()] = 0
    mask = (n_dil!=0) & (sskl!=0)
    ns_set.update([(n,s) for n,s in zip(n_dil[mask],sskl[mask])])
    
    #   Third pass: add seed segments to ns_list
    segNum_no_seed = segNum
    if seed is not None:
        # add a segment (id) for all seed border nodes
        seed_id = seed['sid']
        seg_id  = segNum + _np.arange(1, seed_id.size+1)
        sseed   = _np.zeros(segNum+seg_id.size+1,dtype=seed_id.dtype) # segment.seed
        sseed[-seg_id.size:] = seed_id                              #  attribute
        
        # add (border-node,seed-segments) to ns_set
        ns_set.update([(n,s) for n,s in zip(seed['nodes'],seg_id)])
        # add (seed-node, seed-segments) to ns_set
        ns_set.update([(n,s) for (n,s) in zip(seed_nid[seed['sid']],seg_id)])
        
        segNum = sseed.size-1 # for snlist construction, see below
    

    # add segment to node.segment list & node to segment.node list
    # ------------------------------------------------------------
    nslist = [[] for n in xrange(node.size+1)]     # node.segment
    snlist = [[] for s in xrange(segNum+1)]        # segment.node
    for ni,si in ns_set: 
        nslist[ni].append(si)
        snlist[si].append(ni)
        
    # add segment-list to node array
    node.set_segment(nslist)
    
    # add node pairs to segment array
    snlist[0] = [0,0]  # dummy segment (link dummy node to it-self)
    segment = _SegmentList(node=_np.array([n[:2] + [0]*(2-len(n)) for n in snlist]))
    
    # "remove" invalid segments
    segment.node[(segment.node==0).any(axis=1),:] = 0
    
    if seed is not None:
        segment.add_property('seed',sseed) # seed id for seed segment
 
    # add segment size properties
    # ---------------------------
    if smap is not None:
        # estimated length, area and radius of segments
        n = segNum_no_seed+1
        length = _np.zeros(segment.size+1)
        area   = _np.zeros(segment.size+1)
        radius = _np.zeros(segment.size+1)
        length[:n] = _corner_count(sskl)
        area  [:n] = _label_size(smap)
        radius[:n] = 0.5*area[:n]/_np.maximum(length[:n],1)
        length[0] = 0
        area[0]   = 0
        radius[0] = 0
        
        segment.add_property('length',length)
        segment.add_property('area',  area)
        segment.add_property('radius',radius)
        
    return _RootGraph(node=node, segment=segment)
    
@_aleanode('polyline_graph')
def line_graph(image_graph, segment_skeleton, print_fit_error=True):
    """
    Convert segments of image_graph to polyline
    
    :Input:
        image_graph: a RootGraph as returned by image_graph function
        segment_skeleton: the labeled skeleton related to the graph
        
    :Output:
        A RootGraph where segments are fitted on the skeleton 
    """
    # for concise coding 
    sskl = segment_skeleton
    imgr = image_graph
    
    # local class for efficient list from list of lists
    from itertools import chain
    class ListChain:
        def __init__(self, first_list=[]):
            self.lists  = [first_list]
            self.length = [len(first_list)]
        def extend(self,new_list):
            self.lists.append(new_list)
            self.length.append(self.length[-1]+len(new_list))
        def __len__(self):
            return self.length[-1]
        def __iter__(self):
            return chain.from_iterable(self.lists)
        def to_array(self,dtype=float):
            return _np.fromiter(self.__iter__(), dtype=dtype)
        def id_array(self):
            chain_id = [[i]*(length-(0 if i==0 else self.length[i-1]))
                            for i,length in enumerate(self.length)]
            return _np.fromiter(chain.from_iterable(chain_id), dtype=int)
            
    bg_seg = _np.array([[0,0]],dtype=int)
    pl_seg = ListChain(bg_seg)          # segment of polyline graph (node id pairs)
    pl_nx  = ListChain(imgr.node.x)     # x-coord of polyline graph
    pl_ny  = ListChain(imgr.node.y)     # y-coord of polyline graph
    
    # list of polyline segments per graph segment
    pl_sn  = _np.zeros(imgr.segment.size+1,dtype=object)
    pl_sn[0] = _np.array([])
    
    # parse all segments
    # ==================
    obj = _nd.find_objects(sskl)
    if hasattr(imgr.segment,'seed'):
        seed = imgr.segment.seed
    else:
        seed = _virtual_array(imgr.segment.size+1, 0)  # no segment is a seed
    for s in xrange(1,imgr.segment.size+1):
        if seed[s]==0 and obj[s-1] is not None:
            # fit polyline
            # ------------
            sy,sx = obj[s-1]
            sy = slice(sy.start-1,sy.stop+1)       
            sx = slice(sx.start-1,sx.stop+1) 
            mask = sskl[sy,sx]==s
            
            # get and sort positions of nodes
            sn = imgr.segment.node[s]
            nx = imgr.node.x[sn] - sx.start
            ny = imgr.node.y[sn] - sy.start
            
            # compute the spline
            try:
                x,y = _curve_to_spline(mask,tip_weight=100,smooth=.5,order=1)[1]
                
                # replace end point by segment node positions
                #   first need to check the order of nodes
                dx = (x[[0,-1]][:,None] - nx[None])**2
                dy = (y[[0,-1]][:,None] - ny[None])**2
                d = (dx+dy)**.5
                if (d[0,0]+d[1,1])>(d[1,0]+d[0,1]):
                    nx = nx[::-1]
                    ny = ny[::-1]
                    sn = sn[::-1]
                x[0]  = nx[0]
                x[-1] = nx[1]
                y[0]  = ny[0]
                y[-1] = ny[1]
            except: ## not useful anymore ?
                if print_fit_error:
                    print '\033[31m  failure of spline fitting on a root segment %d\033[30m' % s
                x=nx
                y=ny
                
            x=x+sx.start
            y=y+sy.start
        else:
            sn = imgr.segment.node[s]
            x  = imgr.node.x[sn]
            y  = imgr.node.y[sn]
        
        # append suitable values to polyline graph nodes and segment
        # ----------------------------------------------------------
        prevLen = len(pl_nx)
        pl_nx.extend(x[1:-1])
        pl_ny.extend(y[1:-1])
        seg = _np.fromiter(chain.from_iterable([[sn[0]],xrange(prevLen,len(pl_nx)),[sn[1]]]),dtype=int)
        prevLen = len(pl_seg)
        pl_seg.extend(_np.vstack((seg[:-1],seg[1:])).T)
        pl_sn[s] = _np.arange(prevLen,len(pl_seg))

    pl_node    = _NodeList(x=pl_nx.to_array(), y=pl_ny.to_array())
    pl_segment = _SegmentList(node=_np.concatenate(pl_seg.lists))
    
    pl_node.set_segment(pl_segment)     ## should that be done?? seglist in node?
    pl_segment.compute_length(node=pl_node) 
    pl_segment.compute_direction(pl_node)
    pl_segment.compute_terminal(pl_node)  
    pl_segment.add_property('sid', pl_seg.id_array())
    
    for prop in imgr.segment.properties:
        if prop not in pl_segment.properties:
            pl_segment.add_property(prop, imgr.segment[prop][pl_segment.sid])
    
    return _RootGraph(node=pl_node, segment=pl_segment)

