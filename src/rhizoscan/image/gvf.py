import numpy as np
from scipy import ndimage as nd
from scipy import signal  as sg


from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes


del2 = nd.filters.laplace
##TODO:
##  what's max dt for 3D
##  is it ok to use 'full' filter in 3D (and keep dt max)
##  add other stoping criterion
##  check if gvf3D is actually working
##  make a level of detail method: pyramid ? bigger filter?



@_aleanode({'name':'u'},{'name':'v'})
def GVF_2d(image, iterations, mu, dt='max',filter='default',mode='reflect',cval=0, verbose=False,callback=None):
    """
    Compute the gradient vector field of a 2D image.
    
    :Input:
        image:      either a 2D array or a tuple of two 2D array of the precomputed image gradient
        iteration:  number of iteration of the GVF algorithm
        mu:         the smoothing coefficient of GVF
        dt:         the 'time' step of each iteration. By default dt is max, i.e. 1/(4*mu)
        filter:     either 'default': [0,1,0 / 1,-4,1 / 0,1,0]
                    or     'full'   : [1,1,1 / 1,-8,1 / 1,1,1]
        mode:       how borders are handled by the laplacian filter
                    valid values are : 'reflect','constant','nearest','mirror', 'wrap'
        cval:       if mode is 'constant', use cval value out of image
        verbose:    if 1, print the iteration number as each iteration
                    if 2, print more informations (non-zero pixel % and energy variation)
        callback:   if not None, it should be a function that is call at the end
                    of each iteration. It should be taking 3 arguments: the 
                    iteration number, the u field and the v field: f(iter,u,v)
                    
    :Output:
        u,v: the diffused u and v gradient field 
    """
    
    if dt == 'max':
        dt = 1.0/(4*mu)

    if isinstance(image,tuple):
        # input is the gradient image
        Fx,Fy = image
    else:
        # Normalize image to be between 0 and 1
        image = (image-image.min()) / (image.max()-image.min())                                                             
     
        # Enforce the mirror conditions on the boundary
        image = EnforceMirrorBoundary2D(image)
     
        # Calculate the gradient of the image f
        Fx, Fy = np.gradient(image)
    
    # constant used during iteration    
    grad2 = Fx*Fx + Fy*Fy
    
    # Set up the initial vector field
    u,v = Fx,Fy
 
    for i in range(iterations):
        
        if verbose==1: 
            print 'GVF iteration ', i
        elif verbose:
            print 'GVF iteration ', i, ': ',
            
            # percent of non-zeros gradient pixel and average value
            N = (np.abs(u)>0) | (np.abs(v)>0)
            print 'non-0 pixels:', int(round(100*N.mean())), '% (avg value:',
            print round((np.abs(u[N]).mean() + np.abs(v[N]).mean())/2.0,3), ')',
            
            # current energy
            gf = np.gradient(u) + np.gradient(v)
            e_smooth = mu*reduce(np.add, map(np.square,gf)).sum()
            e_fit    = (grad2*((u-Fx)**2+(v-Fy)**2)).sum()
            print '=> E=', np.around(e_smooth + e_fit,2), '(smooth:', np.around(e_smooth,2), '+ fit:', np.around(e_fit,2), ')'
        
 
        # Enforce the mirror conditions on the boundary
        u = EnforceMirrorBoundary2D(u)
        v = EnforceMirrorBoundary2D(v)
 
        # Update the vector field
        if filter=='default':
            d2u = del2(u,mode=mode)
            d2v = del2(v,mode=mode)
        else:
            d2u = nd.filters.uniform_filter(u,size=3) * (9.0/2) - 4.5*u
            d2v = nd.filters.uniform_filter(v,size=3) * (9.0/2) - 4.5*v
            ##d2u = nd.convolve(u,delOp) #sg.fftconvolve(u,delOp,mode='same')
            ##d2v = nd.convolve(v,delOp) #sg.fftconvolve(v,delOp,mode='same')
        
        u = u + dt*(mu*d2u - (u-Fx)*grad2)
        v = v + dt*(mu*d2v - (v-Fy)*grad2)
        
        
        if callback is not None:
            callback(i,u,v)
        
    return u,v
    
    
@_aleanode({'name':'u'},{'name':'v'},{'name':'w'})
def GVF_3d(volume, iterations, mu, dt='max',mode='reflect',cval=0, verbose=False,callback=None):
    """
    Compute the gradient vector field of a 3D volume.
    
    :Input:
        volume:     either a 3D array or a tuple of three 3D array of the precomputed volume gradient
        iteration:  number of iteration of the GVF algorithm
        mu:         the smoothing coefficient of GVF
        dt:         the 'time' step of each iteration. By default dt is max, i.e. 1/(4*mu) ## valid for 2D only
        mode:       how borders are handled by the laplacian filter
                    valid values are : 'reflect','constant','nearest','mirror', 'wrap'
        cval:       if mode is 'constant', use cval value out of image
        verbose:    if 1, print the iteration number as each iteration
                    if 2, print more informations (non-zero pixel % and energy variation)
        callback:   if not None, it should be a function that is call at the end
                    of each iteration. It should be taking 3 arguments: the 
                    iteration number, the u field and the v field: f(iter,u,v)
                    
    :Output:
        u,v,w: the diffused gradient field 

    ##WARNING: it was not tested, might not even start
    ##todo: test and debug,  what's max dt in 3D ?
    """
    if dt == 'max':
        dt = 1.0/(4*mu)

    if isinstance(volume,tuple):
        # input is the gradient image
        Fx,Fy,Fz = volume
    else:
        # Normalize image to be between 0 and 1
        volume = (volume-volume.min()) / (volume.max()-volume.min())                                                             
     
        # Enforce the mirror conditions on the boundary
        image = EnforceMirrorBoundary3d(volume)
     
        # Calculate the gradient of the image f
        Fx, Fy, Fz = np.gradient(volume)
    
    # constant used during iteration    
    grad2 = Fx*Fx + Fy*Fy + Fz*Fz
    
    # Set up the initial vector field
    u,v,w = Fx,Fy,Fw
 
 
    for i in range(iterations):
        if verbose==1: 
            print 'GVF iteration ', i
        elif verbose:
            print 'GVF iteration ', i, ': ',
            
            # percent of non-zeros gradient pixel and average value
            N = (np.abs(u)>0) | (np.abs(v)>0) | (np.abs(w)>0)
            print 'non-0 pixels:', int(round(100*N.mean())), '% (avg value:',
            print round((np.abs(u[N]).mean() + np.abs(v[N]).mean() + np.abs(w[N]).mean())/3.0,3), ')',
            
            # current energy
            gf = np.gradient(u) + np.gradient(v) + np.gradient(w)
            e_smooth = mu*reduce(np.add, map(np.square,gf))
            e_fit    = (grad2*((u-Fx)**2+(v-Fy)**2+(w-Fz)**2)).sum()
            print '=> E=', round(e_smooth + e_fit,2), '(smooth:', round(e_smooth,2), '+ fit:', round(e_fit,2), ')',

        # Enforce the mirror conditions on the boundary
        u = EnforceMirrorBoundary3D(u)
        v = EnforceMirrorBoundary3D(v)
        w = EnforceMirrorBoundary3D(w)
 
        # Update the vector field
        d2u = del2(u,mode=mode)
        d2v = del2(v,mode=mode)
        d2w = del2(w,mode=mode)
        
        u = u + dt*(mu*d2u - (u-Fx)*grad2)
        v = v + dt*(mu*d2v - (v-Fy)*grad2)
        w = w + dt*(mu*d2w - (w-Fz)*grad2)
        
        
        if callback is not None:
            callback(i,u,v)
        
    return u,v,w


def EnforceMirrorBoundary2D(f):
    """
    This function enforces the mirror boundary conditions
    on the 2D input image f. The values of all pixels at 
    the boundary is set to the values of the pixels 2 steps 
    inward
    """
    N,M = f.shape  #y,x
 
    # Corners
    f[[0,N-1], [0,M-1]] = f[[2,N-3], [2,M-3]]
                                    
    # Edges                         
    f[  2:N,   [0,M-1]] = f[  2:N,   [2,M-3]]
    f[[0,N-1],   2:M  ] = f[[2,N-3],   2:M  ]

    return f

def EnforceMirrorBoundary3D(f):
    """
    This function enforces the mirror boundary conditions
    on the 3D input image f. The values of all voxels at 
    the boundary is set to the values of the voxels 2 steps 
    inward
    """
    N,M,O = f.shape  #y,x,z
 
    # Corners
    f[[0,N-1], [0,M-1], [0,O-1]] = f[[2,N-3], [2,M-3], [2,O-3]];
                                    
    # Edges                         
    f[[0,N-1], [0,M-1],   2:O]   = f[[2,N-3], [2,M-3],   2:O];
    f[  2:N,   [0,M-1], [0,O-1]] = f[  2:N,   [2,M-3], [2,O-3]];
    f[[0,N-1],   2:M,   [0,O-1]] = f[[2,N-3],   2:M,   [2,O-3]];
                                    
    # Faces                         
    f[[0,N-1],   2:M,     2:O]   = f[[2,N-3],  2:M,     2:O];
    f[  2:N,   [0,M-1],   2:O]   = f[  2:N,  [2,M-3],   2:O];
    f[  2:N,     2:M,   [0,O-1]] = f[  2:N,    2:M,   [2,O-3]];   

    return f

