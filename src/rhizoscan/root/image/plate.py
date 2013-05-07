import numpy as _np
import scipy.ndimage  as _nd
import scipy.optimize as _optim

from rhizoscan.ndarray import pad_array as _pad
from rhizoscan.ndarray import lookup    as _lookup
from rhizoscan.image   import Image     as _Image
from rhizoscan         import geometry  as _geo
    
from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes

    
   
@_aleanode('grad')
def border_filter(img, size, axis):
    # "convolve" by flat kernel
    grad = _nd.uniform_filter1d(img,size=size,axis=axis)
    
    # gives slice in suitable dimension
    def sl(s,e): return [slice(s,e) if i==(axis%2) else slice(None) for i in xrange(2)]
    
    # diff with distance = to size
    grad[sl(size/2,-size/2)] = grad[sl(size,None)] - grad[sl(None,-size)]

    # set side pixels with invalid diff to 0
    grad[sl(None, size/2)] = 0
    grad[sl(-size/2,None)] = 0

    return grad

@_aleanode('west','east','north','south')
def fit_border(img, border=0.1, scan_area=[0.25,0.75], verbose=False, bg=1, Wbg=None,Ebg=None,Nbg=None,Sbg=None, pad=True):
    """
    Find the best 4 borders of a square petry plate in an image. 
    
    The plate borders should be close to align with the image border for the
    detection algorithm to work. But Tests worked up to 20 degree rotation.
    
    img:       the gray-scaled image of a petry plate
    border:    overestimated size of the petry plate borders. This size should 
               either be given as the percentage of the lower image dimension 
               and <1, or the number of pixel as an integer >=1
    scan_area: lower and higher bound in the image to use for fitting
    verbose:   if True, display the minimization results
    bg:        relative color of the background compared to the plate
               Should be either 0 for darker background or 1 for lighter.
    *bg:       different color for the west (Wbg), east (Ebg), north (Nbg) and
               south (Sbg) background if different from "bg"
    pad:       plate should be at "border" pixels from the image border. If not
               it is necessary to pad the image. pad = True does that.
         
    return 4 (length-3) vectors representing the 4 borders in homogeneous 
    coordinates. The vectors values are given in the same order as image
    coordinates: (y,x,1)
    
    See also: geometry.fit_homography
    """
    from scipy.optimize import fmin
    
    if border<1:
        border = border*min(img.shape)
    border = int(border)
    
    Wbg = bg if Wbg is None else Wbg
    Ebg = bg if Ebg is None else Ebg
    Nbg = bg if Nbg is None else Nbg
    Sbg = bg if Sbg is None else Sbg
    Wbg = 1 if Wbg<0.5 else 0
    Ebg = 0 if Ebg<0.5 else 1
    Nbg = 1 if Nbg<0.5 else 0
    Sbg = 0 if Sbg<0.5 else 1
    
    
    argfct = [_np.argmax, _np.argmin]  # evaluation w.r.t bg
    factor = [ -1, 1]                  # fmax (i.e. fmin(-f)) or fmin  
    
    if pad: 
        img = _pad(img,border,border,fill_value='nearest') ##??
        pad = border
    
    # find border along x-axis
    # ------------------------
    
    # parameters
    y0,y1 = _np.asarray(scan_area)*img.shape[0] # region-of-interest over y-axis
    subIm = img[y0:y1]      # image to be processed (roi)
    h,w   = subIm.shape     # shape of roi
    Y     = _np.mgrid[:h]   # uncentered indices vector
    y     = Y -h/2          #   centered indices vector
    
    # get the map to fit line in
    gx = border_filter(subIm,size=border, axis=1)
    
    # x-coordinates of line (centered at y1,x)
    def xline(x,dx):  return _np.clip( x + y*dx, 0, w)
    
    # function to optimize
    def Ex(arg, factor=1):
        return _np.sum((factor*_lookup(gx,(Y,xline(*arg)))))
        
    #import matplotlib.pyplot as plt
    #plt.figure(2)
    #plt.plot(_np.sum(gx[:,0:],axis=0))
    #plt.figure(1)
    #plt.clf()
    #plt.imshow(gx[:,pad:-pad],hold=0) ##
    #print argfct[Wbg](_np.sum(gx[:,0:w/2],axis=0)) - pad
    
    # fit left border 
    if verbose: print " find left border \n ----------------- "
    argW = [argfct[Wbg](_np.sum(gx[:,0:w/2],axis=0)), 0]
    argW = fmin(Ex, argW, args=(factor[Wbg],), disp=verbose)
    if verbose: print "\tx = %4.2f \t dx = %1.4f\n" % tuple(argW)
    
    # fit right border 
    if verbose: print " find right border \n ----------------- "
    argE = [argfct[Ebg](_np.sum(gx[:,w/2:],axis=0)) + w/2, 0]
    argE = fmin(Ex, argE, args=(factor[Ebg],), disp=verbose)
    if verbose: print "\tx = %4.2f \t dx = %1.4f\n" % tuple(argE)
    
    # find border along y-axis
    # ------------------------
    
    # parameters
    x0 = argW[0] + abs(argW[1])*h/2.       # area bounded by detected vertical border
    x1 = argE[0] - abs(argE[1])*h/2. - x0  # start (x0) and distance to end (x1)
    x0,x1 = x0 + _np.asarray(scan_area)*x1     # region-of-interest over x-axis
    subIm = img[:,x0:x1]    # image to be processed (roi)
    h,w   = subIm.shape     # shape of roi
    X     = _np.mgrid[:w]   # uncentered indices vector
    x     = X -w/2          #   centered indices vector
    
    # get the map to fit line in
    gy = border_filter(subIm,size=border, axis=0)
    
    # y-coordinates of line (centered at y1,x)
    def yline(y,dy):  return _np.clip( y + x*dy, 0, h)
    
    # function to optimize
    def Ey(arg, factor):
        return _np.sum((factor*_lookup(gy,(yline(*arg),X))))

    # fit upper border 
    if verbose: print " find upper border \n -----------------"
    argN = [argfct[Nbg](_np.sum(gy[0:h/2],axis=1)), 0]
    argN = fmin(Ey, argN, args=(factor[Nbg],), disp=verbose)
    if verbose: print "\ty = %4.2f \t dy = %1.4f\n" % tuple(argN)
    
    # fit lower border 
    if verbose: print " find lower border \n -----------------"
    argS = [argfct[Sbg](_np.sum(gy[h/2:],axis=1)) + h/2, 0]
    argS = fmin(Ey, argS, args=(factor[Sbg],), disp=verbose)
    if verbose: print "\ty = %4.2f \t dy = %1.4f\n" % tuple(argS)

    #h,w = img.shape  # necessary for xline to work
    #return (Y+y0, xline(*argW), xline(*argE)),(X+x0,yline(*argN), yline(*argS))
    
    
    # Compute homography that represent those 4 lines (in homogeneous coordinates)
    # -----------------------------------------------
    ## something strange with sign... to be checked
    west  = _geo.normalize([argW[1], -1, argW[0]-argW[1]*(y0+y1)/2. - pad])
    east  = _geo.normalize([argE[1], -1, argE[0]-argE[1]*(y0+y1)/2. - pad])
    north = _geo.normalize([ -1, argN[1],argN[0]-argN[1]*(x0+x1)/2. - pad])
    south = _geo.normalize([ -1, argS[1],argS[0]-argS[1]*(x0+x1)/2. - pad])

    return west,east, north, south
    
    
@_aleanode('transform')
def find_plate(img, border=0.1, fit='homography', plate_width=1., verbose=False, white_stand=False):
    w,e,n,s = fit_border(img, border=border, scan_area=[0.25,0.75], verbose=verbose)
    
    # quad corner: crossing of plate edges
    # in order: (0,0)   (0,1)   (1,0)   (1,1)   x plate-width
    src = map(lambda x:_np.cross(*x,axis=0),((w,n),(e,n),(w,s),(e,s)))
    src = _np.hstack(src)
    
    # reference plate cornres 
    w = plate_width
    dst = _np.array( [[0.,0,1],[0,w,1],[w,0,1],[w,w,1]] ).T
    
    if fit=='homography': return _geo.fit_homography(src, dst)
    else:                 return _geo.fit_affine(src, dst)

def plot_plate(plate_width, T, color='r',lw=2):
    pw = plate_width
    pos = _np.array([[0,0,1],[pw,0,1],[pw,pw,1],[0,pw,1],[0,0.2*pw,1]]).T
    pos = _geo.normalize(_geo.dot(_geo.inv(T),pos))
    import matplotlib.pyplot as plt
    plt.plot(pos[1],pos[0],color=color, lw=lw)
    
@_aleanode()
def track_plate(sequence,step=1, stand=False):
    """
    ##Obsolete
    warning: since implementation find_plate changed fitting direction => NOT WORKING
    """
    import matplotlib.pyplot as plt                                          
    
    if isinstance(sequence,basestring):                               
        import glob
        sequence = glob.glob(sequence)
        
    plt.ion()
    plt.gray()
    for f in sequence:
        img = _pad(_Image(f,color='gray',format='f'),200,200,fill_value='nearest')
        H   = find_plate(img,border=0.06, plate_width=1200, fit='homography',white_stand=stand)
        im  = _geo.transform(img,H,((0,1200+1,step),(0,1200+1,step)))
        plt.imshow(im,hold=0)
        
        k = raw_input(" 'q' to quit':")
        if k=='q': break
    
@_aleanode()
def draw_line(l,x=None,y=None, color='r'):
    """
    plot a line given in homogeneous coordinates.
    
    Either x or y should be given as a pairs of values of the 2 line ends
    line homoegeneous coordinates should be in image order: (y,x,w)
    ##to be moved to root.gui
    """
    
    if x is not None:
        x = _np.asarray(x)
        y = -(l[1]*x + l[2])/l[0]
    else:
        y = _np.asarray(y)
        x = -(l[0]*y + l[2])/l[1]

    import matplotlib.pyplot as plt
    plt.plot(x,y,color)

@_aleanode()
def draw_plate(w,e,n,s, imshape, color='r', fig=None):
    """ ##to be moved to root.gui"""
    if fig:
        import matplotlib.pyplot as plt
        plt.figure(fig)
        
    y,x = _np.ogrid[map(slice,imshape)]
    draw_line(w,y=y,color=color)
    draw_line(e,y=y,color=color)
    draw_line(n,x=x.T,color=color)
    draw_line(s,x=x.T,color=color)
    
    
def match_plate(mask1, mask2, H0=None, rigid=True, maxiter=None, disp=False):
    """
    find homography that best fit the object in mask1 to the one in mask2
    
    ##todo: make a general image matching algorithm
    ##      not used anymore ?
    """
    dmap = _nd.distance_transform_edt(mask2==0)
    
    ind  = _geo.homogeneous(_np.nonzero(mask1))
    
    if rigid:
        if H0 is None: 
            H0 = [0,0,0]
        def rigid(u,v,r):
            c = _np.cos(r)
            s = _np.sin(r)
            return _np.array([[ c,s,u],[-s,c,v],[0,0,1]])
        def E(T):
            return _np.sum(_geo.transform(dmap,T=rigid(*T),coordinates=ind,cval=1))
            
        H = rigid(*_optim.fmin(E,H0,maxiter=maxiter,disp=disp))
    else:
        if H0 is None: 
            H0 = [1,0,0,0,1,0,0,0,1]
        def E(T): return _np.sum(_geo.transform(dmap,T=_np.reshape(T,(3,3)),coordinates=ind))
        H = _optim.fmin(E,H0,maxiter=maxiter,disp=disp)
        H.shape = 3,3
        
    return H
    
