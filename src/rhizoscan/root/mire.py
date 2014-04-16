import numpy as np
import scipy as sp
import scipy.ndimage as nd
from scipy.linalg import inv

dot = np.dot

def mire_segment(p):    ### this function seems obsolete
    import matplotlib.pyplot as plt
    plt.clf()
    color = 'rgbycm'
    for i,rg in enumerate(p.rgraph):
        sl = rg.segment.length()
        so = np.mod(rg.segment.direction() + 2,np.pi) - 2
        cl = nd.sum(sl,rg.segment.cluster,np.arange(19))
        co = nd.sum(sl*so,rg.segment.cluster,np.arange(19)) / cl
        
        
        bSize  = dot(inv(p.transform[i]),[1,0,1])
        bSize /= bSize[-1]
        px2cm  = 12.5/(bSize[0]**2+bSize[1]**2)**.5

        plt.plot(cl*px2cm,co,'.' + color[i%len(color)])
    
    axis = plt.axis()
    plt.plot([[0.5,1,3,5,8]]*2,[[axis[2]]*5 , [axis[3]]*5], 'k')
    plt.plot([[axis[0]]*3 , [axis[1]]*3], [[-np.pi/2, -np.pi/4, 0]]*2,'k')

def correction_projection(p,i, box_size):
    from rhizoscan import geometry as geo

    Y,X = np.mgrid[map(lambda x: slice(x+1),p.cluster[0].shape)]
    x,y = box_size*geo.transform(T=p.transform[0],coordinates=(X,Y))  # to 125mm square box
    
    du = (np.diff(x[:-1,:],axis=1)**2 + np.diff(y[:-1,:],axis=1)**2)**.5
    dv = (np.diff(x[:,:-1],axis=0)**2 + np.diff(y[:,:-1],axis=0)**2)**.5
    
    return du, dv
    

def detect_circle_frame(n=2, lab=None, d=None, c=None, img=None):
    from rhizoscan.stats   import cluster_1d
    from rhizoscan.ndarray import local_min
    from rhizoscan.ndarray.measurements import label_size
    
    if d is None:
        if c is None:
            c = cluster_1d(img,classes=2,bins=256)
        d = nd.distance_transform_edt(nd.binary_closing(c>0))
    if lab is None:
        lab     = nd.label(d>0)[0]
        
    lradius = nd.maximum(d,lab,index=np.arange(lab.max()+1))
    larea   = label_size(lab)
    #rank   = larea.size - np.argsort(np.argsort(larea-abs((np.pi*lradius**2) - larea)))
    #c_flag = rank<5
    
    #dmax = local_min(-d)*(d>1)  # use position of local max?
    cindex  = np.argsort(larea-abs((np.pi*lradius**2) - larea))[-n:]
    Y,X = np.mgrid[map(slice,d.shape)]
    cy  = nd.mean(Y,labels=lab,index=cindex)
    cx  = nd.mean(X,labels=lab,index=cindex)
    if n>2:
        #keep = np.argsort(cy)[:2]
        #cindex = cindex[keep]
        #cy = cy[keep]
        #cx = cx[keep]
        ## DEBUG test
        keep = np.argsort(cy)
        cx = np.array([cx[keep[0]],np.mean(cx[keep[[1,3]]])])
        cy = np.array([cy[keep[0]],np.mean(cy[keep[[1,3]]])])

        
    # sort left to right
    order = np.argsort(cx)
    cx = cx[order]
    cy = cy[order]
    
    # create affine transorm
    a  = np.arctan2(cy[1]-cy[0], cx[1]-cx[0])
    sa = np.sin(-a)
    ca = np.cos(-a)
    T = np.array([[ca,-sa, cx[0]-cx[1]],[sa,ca, cy[0]-cy[1]],[0,0,1]])
    #T = np.array([[ca,-sa, 0],[sa,ca, 0],[0,0,1]])
    
    return d, lab, cx,cy, T

def process_mire_image(fname=None):
    if fname is None:
        fname = '/Users/diener/root_data/frame_mire1.png '
        
    from rhizoscan.image   import Image
    from rhizoscan.ndarray import local_min
    from rhizoscan.image.measurements import image_to_csgraph as tograph
    
    img = Image(fname,dtype='f')
    img = 1-img[:,:,0]
    
    mask = img>0.5
    d = nd.distance_transform_edt(mask)
    dmax = local_min(-d,strict=True)*(d>1)
    
    # construct csgraph
    cost = np.exp(-d)
    cost[dmax] = 0
    g,x,y = tograph(cost,edge_value=lambda x,y,d:x*y, neighbor=4) ##exp(-(dx+dy))
    
    
    
