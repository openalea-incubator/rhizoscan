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
    hull = _sp.spatial.Delaunay(vertices).convex_hull
    graf = _sp.sparse.csr_matrix((_np.ones(hull.shape[0]),hull.T), shape=(hull.max()+1,)*2)
    hull = _sp.sparse.csgraph.depth_first_order(graf,hull[0,0],directed=False)[0]
    
    return vertices[hull]


def fit_quad(polygon, corner0='bbox', focus=0.9, method='BFGS', **min_kargs):
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
    
    # cost function (to minimize)
    def dist2quad(c):
        c = c.reshape(-1,2)
        t = hnormalize(_np.roll(c,-1,0)-c)  # tangent of quad
        p = _np.diag(projdist(c,t))         # shift of tangent origin w.r.t tangent vector  
        
        return _np.sum(abs(projdist(h,t)-p[None,:]).min(axis=1)*w)
        
        
    # do the fitting:
    from scipy.optimize import minimize
    c = minimize(fun=dist2quad, x0=c0, method=method, **min_kargs)
    c.x = c.x.reshape(c0.shape)
    
    return c
