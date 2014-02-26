"""
Implement a tool to correct/create the `seed_map` containing the root seeds
"""

import numpy as _np
from scipy import ndimage as _nd

from matplotlib.blocking_input import BlockingMouseInput as _BlockMouse
##from matplotlib.blocking_input import BlockingInput as _BlockInput, BlockingMouseInput as _BlockMouse
##import matplotlib.pyplot as _plt  # check if matplotlib exist
##from rhizoscan.workflow import node as _node # to declare workflow nodes

class Edition(object):
    """
    A unit edition for an image
    """
    def __init__(self, action, slices, mask, value):
        self.action = action
        self.slices = slices
        self.mask  = mask
        self.value = value
        
        self.previous = None
        
    def apply(image):
        if self.previous is not None:
            print "*** warning: this edition has been applied before - stored 'previous' mask will be overwritten ***" 
        img = image[self.slices]
        self.previous = img
        img[mask] = self.value
        
        return image
        
    def undo(image):
        if self.previous is None:
            raise RuntimeError("This edition has not been applied yet (its previous mask is not set)")
        
        img = image[self.slices]
        img[mask] = self.previous
        self.previous = None
        
        return image
        
class EditionStack(list):
    """
    Implement a stack of Edition objects for the edition of a seed map array
    """
    def __init__(self, seed_map):
        self.seed_map = seed_map
        
    def extend(self, iterable)
        for i,edition in enumerate(iterable):
            if not isinstance(edition,Edition):
                raise TypeError("EditionStack can only store Edition object, last append: element %d" % i-1)
            self.append(edition)
        
    def append(self, action, seed_number, polygon=None):
        """
        Create (&apply) the suitable action and append it to this stack
        
        `action`:
            The following actions are implemented:
             - 'add': draw seed `seed_number`            - requires `polygon`
             - 'delete': remove the seed `seed_number`
             
        `seed_number`:
            The number of the seed to add/remove
             
        `polygon`: (if `action`=='add') 
            a Nx2 array-like of the seed contour **in XY order**
    
    
        Note: 'add' action requires opencv or PIL
        """
        seed_map =self.seed_map
        if action=='delete':
            # find seed
            slices = _nd.find_objects(seed_map==seed_number)[1]
            smap = seed_map[slices]
            mask = smap==seed_number
            value = 0
            
        elif action=='add':
            from rhizoscan.gui.image.draw import fill_polygon
            
            # construct slices bbox of given polygon
            polygon = _np.asarray(polygon)
            bmin = _np.floor(polygon.min(axis=0)).astype(int)
            bmax = _np.ceil( polygon.max(axis=0)).astype(int)
            slices = [slice(bmin[1],bmax[1]), slice(bmin[0],bmax[0])]
            
            # create mask
            mask = _np.zeros((bmax-bmin)[:,::-1],dtype=bool)
            mask = fill_polygon(mask=mask.astype('uint8'),polygon=polygon-bmin)>0
            
            value = seed_number
            
        # create and apply edition
        edit = Edition(action=action,slices=slices,mask=mask, value=value)
        edit.apply(seed_map=seed_map)
            
        list.append(self,edit)
        

class SeedMapEditor(_BlockMouse):
    """
    Simple user interface to view and edit seed map (in-place)
    """
    def __init__(self, seed_map, fig=1):
        if fig:
            _plt.ion()
            _plt.figure(1)
        _plt.clf()
        _plt.imshow(seed_map)
            
    def mouse_event_add(event):
        print 'mouse add', event.xdata, event.ydata
    def mouse_event_pop(event):
        print 'mouse pop', event.xdata, event.ydata

