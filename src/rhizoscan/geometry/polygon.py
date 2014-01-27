"""
geometric tools related to polygon
"""
import numpy as _np
import scipy as _sp
          
def convex_hull(vertices):
    """
    compute the convex hull of the given set of `vertices`
    
    vertices: a Nx2 array of the `N` vertices coordinates
    
    return a Kx2 array of the convex hull coordinates, as a polygon.
    """
    from scipy.spatial import Delaunay
    hull = Delaunay(vertices).convex_hull
    graf = _sp.sparse.csr_matrix((_np.ones(hull.shape[0]),hull.T), shape=(hull.max()+1,)*2)
    hull = _sp.sparse.csgraph.depth_first_order(graf,hull[0,0],directed=False)[0]
    
    return vertices[hull]
    


def fit_quad(polygon, corner0='bbox', focus=0.9, method='BFGS', dof_map=None, **min_kargs):
    """
    Find the 4 corners of a quad to `polygon` 
    
    polygon:
        Nx2 array of the polygon vertices
    corner0:
        Initial corner positions. Should be a 4x2 array, or length-8 vector.
        If 'bbox', use the `polygon` bounding box.
    focus:
        Percent of vertices total weight used to subsample vertices:
        The weight of each vertices is computed relative the to length of the 
        polygon part closest to each vertex. If `focus` < 1, then fit the quad 
        on the vertices with main weight that represent `focus` percent of total
        If focus is None, don't use weighting - nor vertices selection. In this 
        case the fit is done on all vertices without consideration on segments.
    method:
        the minimization method used by `scipy.optimise.minimize`. By default,
        i.e. if min_kargs is empty, it is called with suitable minimization 
        function f:X->scalar-cost, and initial corner position
    **min_kargs:
        option key-arguments to give to `scipy.optimize.minimize`
        
    dof_map:
        Use to change the degree of freedom (dof) that is optimized 
        - mostly for internal purpose -
        If not None, use be a function that takes as input the dof variables and 
        return the respective corner array. In this case, the optimization is 
        done for the given dof. And input `corner0` should be given accordingly.
        
    Return the result object from `scipy.optimize.minimize` where x the solution
    """
    polygon = _np.asarray(polygon)
    
    # initiate quad as bound box of `polygon`
    if corner0=='bbox':
        hmin = polygon.min(axis=0)
        hmax = polygon.max(axis=0)
        coord = lambda i,j: [hmax[0] if i else hmin[0], hmax[1] if j else hmin[1]]
        c0 = _np.array([coord(0,0),coord(0,1),coord(1,1),coord(1,0)])
    else:
        c0 = _np.asarray(corner0)
        
    # some functions
    hnorm      = lambda v: (v**2).sum(axis=1)**.5
    hnormalize = lambda v: v/hnorm(v)[:,None]
    projdist   = lambda x,v: _np.diff(x[:,None,:]*v[None,:,::-1],axis=-1)[:,:,0]
    def vertice_weight(h):
        lhtl = hnorm(_np.roll(h,-1,0)-h)    # length of polygon segment
        return lhtl + _np.roll(lhtl,-1)     # weight of polygon vertices
        
    # find most representative polygon vertices, and their weight
    ##   add some vertex removale algorithm that doesn't change the shape ???
    if focus is None:
        h = polygon
        w = 1
    elif focus<1:
        hw = vertice_weight(polygon)
        order =_np.argsort(hw)[::-1]
        percent  = _np.cumsum(hw[order])
        percent /= percent[-1]
        keep = _np.sort(order[:_np.argmax(percent>focus)])
        
        h = polygon[keep]
        w = vertice_weight(polygon[keep])
    else:
        h = polygon
        w = vertice_weight(polygon)
        
    if dof_map is None:
        dof_map = lambda x:x
    
    # cost function (to minimize)
    def dist2quad(c):
        c = dof_map(c).reshape(-1,2)
        t = hnormalize(_np.roll(c,-1,0)-c)  # tangent of quad
        p = _np.diag(projdist(c,t))         # shift of tangent origin w.r.t tangent vector  
        
        return _np.sum(abs(projdist(h,t)-p[None,:]).min(axis=1)*w)
        
                                                                                 
    # do the fitting:
    from scipy.optimize import minimize
    c = minimize(fun=dist2quad, x0=c0, method=method, **min_kargs)
    c.x = c.x.reshape(c0.shape)
    
    return c
    

def rect2corner(C):
    """
    Convert rectangle coordinates - C=[x,y,w,h,r] - to corner array
    """
    x,y,w,h,r = C
    cr, sr = _np.cos(r), _np.sin(r)
    return _np.array([[x,y],[x-sr*h,y+cr*h],[x+cr*w-sr*h,y+sr*w+cr*h],[x+cr*w,y+sr*w]])
    
def fit_rect(polygon, corner0='bbox', focus=0.9, method='BFGS', **min_kargs):
    """
    Same as fit quad but restricted to rectangle (x,y,w,h,rotation)
    """
    if corner0=='bbox':
        hmin = polygon.min(axis=0)
        hdif = polygon.max(axis=0)-hmin
        corner0 = _np.hstack((hmin,hdif,[0])).astype(float)
              
    r = fit_quad(polygon=polygon, corner0=corner0, focus=focus, method=method,
                 dof_map=rect2corner, **min_kargs)
                                               
    return r
    
def rect2corner_b(C):
    """
    Convert rectangle coordinates - C=[x,y,dx1,dy1,d2] - to corner array
    """
    x,y,dx1,dy1,d2 = C
    
    dratio  = d2/(dx1**2+dy1**2)**.5    # normalisation
    dx2 = -dy1*dratio                   # 90 degree rotation
    dy2 =  dx1*dratio                   # ---------"--------
    
    return _np.array([[x,y],[x+dx1,y+dy1],[x+dx1+dx2,y+dy1+dy2],[x+dx2,y+dy2]])
    
def fit_rect_b(polygon, corner0='bbox', focus=0.9, method='BFGS', **min_kargs):
    """
    Same as fit quad but restricted to rectangle (x,y,w,h,rotation)
    """
    if corner0=='bbox':
        hmin = polygon.min(axis=0)
        hdif = polygon.max(axis=0)-hmin
        corner0 = _np.hstack((hmin,[hdif[0],0,hdif[1]])).astype(float)
            
    r = fit_quad(polygon=polygon, corner0=corner0, focus=focus, method=method,
                    dof_map=rect2corner_b, **min_kargs)
    r.x = corner0
    return r
    
    
def distance_to_segment(points,segments):
    """
    Compute the minimum distance from all `points` to all `segments`
    
    `points`: 
       a (k,n) array for the k-dimensional coordinates of n points
    `segments`:
       a (k,s,2) array for the k-dim coordinates of the 2 vertices of s segments
       
    :Outputs: 
       - a (n,s) array of the point-segment distances
       - a (k,n,s) array of the coordinates of the closest point (i.e. the point
       projection) of all points on all segments
       - a (n,s) array of the position in [0,1] of points on the segments
    """
    points   = _np.asfarray(points)
    segments = _np.asfarray(segments)
    
    norm = lambda x: (x**2).sum(axis=0)**.5
    
    v1    = segments[...,0]           # 1st segment vertex, shape (k,s)
    v2    = segments[...,1]           # 2nd segment vertex, shape (k,s)
    sdir  = v2-v1                     # direction vector of segment
    lsl   = norm(sdir)                # distance between v1 and v2
    lsl   = _np.maximum(lsl,2**-5)    
    sdir /= lsl                       # make sdir unit vectors
    
    # distance from v1 to the projection of points on segments
    #    disallow projection out of segment: values are in [0,lsl]
    on_edge = ((points[:,:,None]-v1[:,None,:])*sdir[:,None,:]).sum(axis=0) # (n,s)
    on_edge = _np.minimum(_np.maximum(on_edge,0),lsl[None,:])
    
    # distance from points to "points projection on sdir"
    nproj = v1[:,None,:] + on_edge[None,:,:]*sdir[:,None,:]   # (k,n,s)
    d = norm(nproj - points[:,:,None])                        # (n,s)

    return d, nproj, on_edge/lsl


