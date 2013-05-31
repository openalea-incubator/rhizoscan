import numpy as _np
import scipy as _sp
import scipy.ndimage as _nd

from rhizoscan.ndarray   import kernel, add_dim, virtual_array as _virtual_array

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes


@_aleanode('segment_map','node_map')
def skeleton_label(mask, closing=1, fill_node=True, terminal=True):
    """
    Skeletonize (thin) mask, and return map of labeled skeleton segments and nodes
    
    :Inputs:
       - mask: 
           An image containing the objects to be skeletonized. '1' represents 
           foreground, and '0' represents background. It also accepts arrays of 
           boolean values where True is foreground.
       - closing:
           iteration number of binary closing applied on mask. If not previously 
           filtered or otherwise "well constructed", one iterations is 
           recommanded to avoid one-pixel holes.
       - fill_node: 
           If True add node pixels that touches only one segment to this segment.
           This makes the segment map convering most of the mask skeleton (which 
           is better segment length estimation). As consequences:
             - segment can then touch other segments
             - all nodes keep at least one pixels
             - individually segment are still linear (i.e. contains no loop) 
       - terminal:
           If True, add terminal node to node label (i.e. node that touch only 
           one segment). These node are also kept in the segment label. 
           To remove them, simply do::
              segment[node>0] = 0
           
    :Outputs:
        - segment: a label map of the skeleton segments
        - node:    a label map of the skeleton nodes
        - nSeg:    number of segment
        - nNode:   number of nodes (including terminals)
        - nTerm:   number of terminal
        
    *** Require scikits images: skimage.morphology.skeletonize ***
    """
    from skimage.morphology import skeletonize
    
    if closing:
        mask = _nd.binary_closing(mask, iterations=closing)
        
    # skeletonize and extract linear skeleton segment
    sk   = skeletonize(mask).astype('float32')
    sk  *= _np.round(9*_nd.uniform_filter(sk))-1 # neighbor number
    seg  = (sk==2)| (sk==1)                      # segment: 2 neighbors
    node = (sk>2)                                # nodes: 1 or more than 2 neighbors
    
    # label segment
    seg,ns = _nd.label(seg,structure=_np.ones((3,3)))
    
    # add the node pixels that touches only one segment to this segment
    node,nn = _nd.label(node,structure=_np.ones((3,3)))
    while fill_node:
        # distance of node pixels to closest segment pixels
        dist = _nd.convolve((seg!=0).astype('f'),kernel.distance((3,3)))
        
        # distance limit for a node pixel to be merge in its closest segment
        # computed by node cluster
        dmin = _nd.minimum(dist, labels=node, index=xrange(nn+1))
        dmax = _nd.maximum(dist, labels=node, index=xrange(nn+1))
        dmax[dmin==0]    = _np.inf  # no restiction: at least one node pixels will be kept
        dmax[dmin==dmax] = 0        # no (more) node pixel can be removed
        dmax[0]          = 0        # to "protect" background

        # find node pixels that touches only one segment 
        n_seg   = _nd.maximum_filter(seg, size=(3,3))
        uniq    = n_seg == _nd.minimum_filter(seg + (seg==0)*(ns+1), size=(3,3))
                   
        # node pixels to be changed                   
        to_fix  = uniq  &(dist<dmax[node]) #& (node!=0)
        
        fill_node = _np.sum(to_fix)
        if fill_node: 
            seg [to_fix] = n_seg[to_fix]
            node[to_fix] = 0
    
    # add terminal node
    if terminal:
        terminal,nt = _nd.label(sk==1, structure=_np.ones((3,3)))
        term = terminal!=0
        node[term] = terminal[term] + nn
        nn += nt
    else:
        nt = 0
    
    return seg, node, ns, nn, nt
    
    
@_aleanode('size')
def corner_count(curve_map, background=0):
    """
    compute the length of (linear) label using the corner count method
    
    :Inputs:
      - curve_map:
          a labeled map where each label should be a connected linear set 
          of pixels (a curve) surrounded by background pixels. 
          The algorithm works on any type of label map, but is meaningfull 
          only if it correspond to the description above.
      - background:
          value of the background pixels (default is 0). 
      
    :Outputs:
        a vector array where each element is the size of the corresponding 
        labeled curve. The size of the background label is 0 
    
    Corner count reference::
        "Vector code probability and metrication error in the representation of 
         straight lines of finite length"
         Vossepoel and Smeulders, 1982
    """
    d4 = 0.98  # weight of direct   connections (4-connected neighbors)
    d8 = 1.406 # weight of diagonal connections (diagnonal neighbors)
    
    d = _np.array([[d8,d4,d8],[d4,0,d8],[d8,d4,d8]])    # weight kernel 
    c = _np.array([[ 1, 2, 3],[-4,0, 4],[-3,-2,-1]])    # corner detection kernel
    
    mask = (curve_map!=0).astype(float)
    d = _nd.convolve(mask,weights=d,mode='constant')/2  # distance per pixel to its neighbors
    c = _nd.convolve(mask,weights=c,mode='constant')!=0 # True if pixel is a corner
    
    ind  = range(0,curve_map.max()+1)
    size = _np.array(_nd.sum(d,labels=curve_map,index=ind)) \
           -0.091*(_np.array(_nd.sum(c,labels=curve_map,index=ind))-1)
    
    size[background] = 0  # background size = 0
    return size


_colormap = _np.array([[0,0,0],[1,1,1],[1,0.3,0.1],[0,1,0],[0.2,0.7,1],[1,1,0],[1,0,1],[0,1,1]])

def color_label(label, order='shuffle', cmap=6, start=1, negative=0):
    """
    make a color image from label.
    
    :Inputs:
      - order:
          how to choose the color order - either:
            * shuffle: shuffle lavels id (>start)
            * xmin:    order label mapping by the labels minimum x coordinates
            * ymin:    order label mapping by the labels minimum y coordinates
            * an integer: use directly the label id multiplied by this number
      - cmap:
         the color map - either
            * None: use the module default _colormap (8 basic colors) 
            * a colormap (Nx3 array of N colors),
            * or a number (simply apply modulus, and return a grey color)
      - start:
          loop into cmap starting at this label.
          it should be less than the number of colors in the color map
          if order is shuffle, labels below start are not shuffled
      - negative:
          method to treat negative indices - a value to replace <0 labels by
    """
    label = _np.copy(label)
    label[label<0] = negative
        
    if cmap is None:
        cmap  = _colormap
        start = _np.minimum(start, cmap.shape[0])
    else:
        cmap = _np.asarray(cmap)
        if cmap.size==1:
            cmap = _np.arange(cmap)
    
    if order == 'xmin':
        x   = add_dim(_np.arange(0,label.shape[1]),axis=0,size=label.shape[0])
        ind = _np.argsort(_np.argsort(_nd.minimum(x,label,index=range(0,label.max()+1))))
    elif order == 'ymin':
        y   = add_dim(_np.arange(0,label.shape[0]),axis=1,size=label.shape[1])
        ind = _np.argsort(_np.argsort(_nd.minimum(y,label,index=range(0,label.max()+1))))
    else:
        ind = _np.arange(label.max()+1)
        if order=='shuffle':
            _np.random.shuffle(ind[start:])
        else:
            factor = max(order,1)
            ind[start:] = factor*ind[start:]
    
    ind[ind>=start] = start+_np.mod(ind[ind>=start],cmap.shape[0]-start)
    Lind   = ind[label]
    clabel = cmap[Lind,:]

    return clabel
                                                              
def image_to_csgraph(image, edge_value='mean', neighbor=8):##DOC FALSE
    """
    Make a compressed sparse graph from input image. 
    The graph connects all non-zero pixels to its non-zero neighbors.
    
    :Inputs:
      - image:
          an ndarray with 2 dimensions
      - edge_value:
          * 'mean' - the edge value is the mean of src & dst
          * 'min'  - the edge value is the minimum of src & dst
          * 'max'  - the edge value is the minimum of src & dst
          * 'diff' - the edge value is  dst - src
          * 'grad' - the edge value is  abs(dst - src)
          * a function with 2 argument f(src,dst) that return the edge value
      - neighbor:
          4 (direct pixels neighbors) or 8 (default, with diagonal neighbors)
    
    :Outputs:
        - a sparse graph (from scipy.sparse), which represents adjacency matrix 
          of the pixels graph, that can be used by scipy.sparse.csgraph functions.
        - y-coordinates of selected pixels
        - x-coordinates of selected pixels
    
    :Note:
        border pixels should all equal 0
    """
    from scipy import sparse
    
    ##add dtype arguments for output matrix
    idy, idx = _np.nonzero(image)
    mid = _np.arange(idy.size)                 
    
    idmask = -_np.ones(image.shape,dtype=int)
    idmask[idy,idx] = _np.arange(mid.size)
    
    g = sparse.csr_matrix((mid.size,)*2)
    
    src_val = image[idy,idx]
    
    if isinstance(edge_value,basestring):
        if   edge_value=='mean': edge_value=lambda x,y,d: (x+y)/2
        elif edge_value=='diff': edge_value=lambda x,y,d: (y-x)
        elif edge_value=='grad': edge_value=lambda x,y,d: _np.abs(x-y)
        elif edge_value=='min':  edge_value=lambda x,y,d: _np.minimum(x,y)
        elif edge_value=='max':  edge_value=lambda x,y,d: _np.maximum(x,y)
        else:
            raise TypeError('invalid edge_value function name')
        
    from rhizoscan.ndarray import virtual_array
    d = virtual_array(idx.shape)
    
    for nb,(dy,dx) in enumerate([(-1,0),(0,+1),(+1,0),(0,-1),(-1,-1),(-1,+1),(+1,+1),(+1,-1)][:neighbor]):
        y = idy+dy
        x = idx+dx
        
        nb_id = idmask[y,x]
        valid = nb_id>=0
        d[0]  = (abs(dx)+abs(dy))**.5
        value = edge_value(src_val, image[y,x],d)
        
        nbg = sparse.csr_matrix((value[valid],(mid[valid],nb_id[valid])),(mid.size,)*2)
        g   = g+nbg
        
    return g, idy, idx

def sort_curve_pixels(mask):
    """
    Sort the pixels of a curve from a binary ndarray
    
    :Inputs:
      - mask:
          a binary image containing a 1-pixel width curve. All pixels should 
          have maximum 2 neighbors.
      - x0,y0:
          the coordinates of the first pixels
          
    :Outputs:
        The y and x coordinates of the sorted mask pixels.
    
    .. Require::
        scipy>=0.11
    """
    from scipy import sparse
    
    g,Y,X = image_to_csgraph(mask, edge_value=lambda x,*args: _virtual_array(x.shape))
    nbStart = g.sum(axis=1)==1
    if nbStart.any():
        start = _np.ravel(nbStart).nonzero()[0][0]
    else:    # if skeleton is a close loop
        start = 0
    
    # require scipy 0.11
    order = sparse.csgraph.depth_first_order(g,start,return_predecessors=False)
    
    return Y[order], X[order]
     
def curve_to_spline(mask, tip_weight=10, smooth=1, order=3):##, x0=None, y0=None, x1=None, y1=None):
    """
    Fit a spline on curve in binary array 'mask'
    
    :todo: finish doc
    """
    from scipy.interpolate import splprep as fit_spline
    
    y,x = sort_curve_pixels(mask)
    x = _np.asarray(x,dtype='f')
    y = _np.asarray(y,dtype='f')
    
    # if order is too big w.r.t number of pixels
    order = min(order,max(y.size-1,1))  # in case their is not enough point for required order
    
    ## OLD: TO delete, aswell as x&y args... manage x* & y* arguments
    #if x0 is not None:
    #    # replace first point by given x0,y0
    #    x[0] = x0
    #    y[0] = y0
    #if x1 is not None:
    #    # replace last point by given x1,y1
    #    if y.size>1:
    #        x[-1] = x1
    #        y[-1] = y1
    #    else:
    #        # in case the curve has only one point
    #        # return a order 1 spline
    #        arr = _np.array
    #        return [arr([0,0,1,1]), [arr([x[0],x1]),arr([y[0],y1])], 1]
            
    if x.size==1:
        # in case the curve has only one point
        # return a order 1 spline
        return [_np.array([0,0,1,1]), [x[[0,0]],y[[0,0]]], 1]
        
    w = _np.ones(y.size)
    w[0] = w[-1] = tip_weight
    tck,u = fit_spline([x,y],w=w,s=x.size*smooth,k=order)
    return tck

    
