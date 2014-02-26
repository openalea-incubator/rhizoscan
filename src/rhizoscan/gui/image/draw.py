"""
Some image drawing functions
"""

import numpy as _np

from rhizoscan.workflow import node as _node # to declare workflow nodes

@_node('updated_mask')
def fill_polygon(mask, polygon, order='xy', color=1):
    """
    Draw given `polygon` in a mask using opencv
    
    `polygon`: a Nx2 array-like of 2d coordinates of N points 
    `order`: either 'xy' or 'yx', the coordinates order of `polygon` 
    `color`: value to draw
    
    Note: 
      This method tries first to use opencv (cv2), then otherwise use PIL.
      One of those is thus required
    """
    try:
        import cv2
        method = 'cv'
    except ImportError:
        try:
            from PIL import Image, ImageDraw
            method = 'PIL'
        except ImportError:
            raise ImportError("fill_polygon requires either opencv (cv2) or PIL")
    
    polygon = _np.asarray(polygon)
    if order=='yx':
        polygon = polygon[:,::-1]
    bmin = _np.floor(polygon.min(axis=0)).astype(int)
    bmax = _np.ceil( polygon.max(axis=0)).astype(int)
    slices = [slice(bmin[1],bmax[1]), slice(bmin[0],bmax[0])]
    
    m = mask[slices]
    polygon = polygon-bmin+0.5
    
    if method=='cv':
        cv2.fillPoly(m,[polygon.astype('int32')],color=color)
    else:
        m2 = Image.new('L', m.shape[::-1], 0)
        ImageDraw.Draw(m2).polygon(map(tuple,polygon), fill=1)
        m2 = _np.asarray(m2)
        m[m2>0] = color

    return mask
    
def _uitest_fill_polygon(shape=(30,40)):
    """
    test fill_polygon_cv through a simple matplotlib interface:
        - show empty array and let user manually input a polygon
        - draw the polygon
        - show it 
    """
    from matplotlib import pyplot as plt
    from rhizoscan.gui.image import linput
    
    mask = _np.zeros(shape,dtype='uint8')
    plt.clf()
    plt.imshow(mask)
    value=1
    
    while 1:
        p = _np.asarray(linput())
        if len(p)==0: break
        
        fill_polygon(mask, p, order='xy', color=value)
        plt.imshow(mask, interpolation='nearest')
        plt.plot(p[:,0],p[:,1],'w')
        value+=1