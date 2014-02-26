"""
Implement a tool to correct/create the `seed_map` containing the root seeds
"""

import numpy as _np
from scipy import ndimage as _nd

import matplotlib.pyplot as _plt
from matplotlib.blocking_input import BlockingInput as _BlockInput

from rhizoscan.workflow import node as _node # to declare workflow nodes

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
        
    def apply(self, image):
        if self.previous is not None:
            print "*** warning: this edition has been applied before - 'previous' mask will be overwritten ***" 
        img = image[self.slices]
        self.previous = img[self.mask]
        img[self.mask] = self.value
        
        return image
        
    def undo(self, image):
        if self.previous is None:
            print "This edition has not been applied yet or already undone"
            return
        
        img = image[self.slices]
        img[self.mask] = self.previous
        self.previous = None
        
        return image
        
class SeedMapEditor(list):
    """
    Editor of seed map array, implemented as a stack of Edition objects
    """
    def __init__(self, seed_map, root_mask):
        self.seed_map  = seed_map
        self.root_mask = root_mask
        self.update_seed_id_list()
        
    # manage list of used seed id
    # ---------------------------
    def update_seed_id_list(self):
        self.cur_seed = set(_np.unique(self.seed_map))
        self.cur_seed.discard(0)
    def _remove_seed_id(self, sid):
        self.cur_seed.remove(sid)
    def _get_new_seed_id(self):
        """ find first unused seed id, add it to used list and return it """
        slist = sorted(self.cur_seed)
        free_id = [j for i,j in zip(slist,range(1,len(slist)+1)) if i<>j]
        if len(free_id):
            new_id = free_id[0]
        elif len(slist):
            new_id = slist[-1]+1
        else:
            new_id = 1
            
        self.cur_seed.add(new_id)  ## if new_id is not really used, then cur_seed will be wrong
        return new_id
    
    # add/remove edition
    # ------------------
    def append(self, action, point=None):
        """
        Create (&apply) the suitable action and append it to this stack
        
        `action`:
            The following actions are implemented:
             - 'add': draw seed `seed_number`
             - 'delete': remove the seed `seed_number`
             
        `point`: 
            - if `action`=='add': a Nx2 array of the seed contour **in XY order**
            - the x,y coordinate of a point in seed area to be deleted
    
        Note: 'add' action requires opencv or PIL
        """
        seed_map =self.seed_map
        if action=='delete':
            # find seed
            x,y = map(int, point)
            seed_number = seed_map[y,x]
            if seed_number==0: 
                return None
            slices = _nd.find_objects(seed_map==seed_number)[0]
            smap = seed_map[slices]
            mask = smap==seed_number
            value = 0
            self._remove_seed_id(seed_number)
            print "*** remove seed %d ***" % seed_number
            
        elif action=='add':
            from rhizoscan.gui.image.draw import fill_polygon
            
            # construct slices bbox of given polygon
            polygon = _np.asarray(point)
            bmin = _np.floor(polygon.min(axis=0)).astype(int)
            bmax = _np.ceil( polygon.max(axis=0)).astype(int)
            slices = [slice(bmin[1],bmax[1]), slice(bmin[0],bmax[0])]
            
            # create mask
            mask = _np.zeros((bmax-bmin)[::-1],dtype=bool)
            mask = fill_polygon(mask=mask.astype('uint8'),polygon=polygon-bmin)[0]>0
            
            value = self._get_new_seed_id()
            print "*** add seed %d ***" % value
            
        # create and apply edition
        edit = Edition(action=action,slices=slices,mask=mask, value=value)
        edit.apply(image=seed_map)
            
        list.append(self,edit)
        
    def pop(self):
        """ undo last edition """
        if len(self):
            self[-1].undo(self.seed_map)
            list.pop(self)
        
    def extend(self, iterable):
        raise NotImplementedError()
        
        for i,edition in enumerate(iterable):
            if not isinstance(edition,Edition):
                raise TypeError("not Edition object (failed at element %d)" % i-1)
            ##todo: apply edition and append it to self
        

class SeedMapBlocking(_BlockInput):
    """
    Simple user interface to view and edit seed map (in-place)
    """
    def __init__(self, seed_map_editor, fig=1):
        if fig:
            _plt.ion()
            _plt.figure(fig)
        fig = _plt.gcf()
            
        self.editor = seed_map_editor
            
        self.update_display()
            
        _BlockInput.__init__(self, fig=fig, eventslist=('key_press_event',
                                                        'button_press_event', 
                                                        'motion_notify_event',
                                                        'button_release_event'))
            
    def post_event(self):
        """ Dispatch event """
        assert len(self.events) > 0, "No events yet"

        update_display = False
        event = self.events[-1]
        if event.name == 'key_press_event':
            self.key_event(event)
        else:
            button = event.button
            x,y = event.xdata, event.ydata
            if x is None:
                return
                
            if button==1:
                if self.mode=='select':
                    self.mode = 'draw'
                    self.polygon = [[x,y]]
                elif self.mode=='draw':
                    self.polygon.append([x,y])
                    if event.name=='button_release_event':
                        try:
                            self.mode = 'wait'
                            self.editor.append(action='add',point=self.polygon) # add seed
                            update_display = True
                            self.polygon = None
                        finally:
                            self.mode = 'select'
                        
            elif button==3:
                if self.mode=='select':
                    self.mode = 'wait'
                    self.editor.append(action='delete', point=[x,y])  # delete seed
                    update_display = True
                    self.mode = 'select'
                        
        if update_display:
            self.update_display()
            
    def key_event(self, event):
        key = event.key
        if key in ['escape', 'enter']:
            self.stop(event)
        elif key=='ctrl+z' and self.mode<>'wait':
            self.editor.pop()
            self.update_display()
            
            
    def stop(self, event):
        """
        Stop block, ending user interaction
        """
        _BlockInput.pop(self, -1)
        self.fig.canvas.stop_event_loop()
        
    def update_display(self):
        _plt.clf()
        _plt.imshow(self.editor.seed_map + self.editor.root_mask)
        
    def __call__(self):
        """ block execution and start the event handler """
        self.mode = 'select'
        self.seed_select = None
        _BlockInput.__call__(self,n=0,timeout=0)
        
        return None

@_node('seed_map','editor_object')
def seedmap_editor(seed_map, root_mask, fig=1):
    sedit = SeedMapEditor(seed_map, root_mask)
    smb = SeedMapBlocking(seed_map_editor=sedit,fig=fig)
    print "seed map editor, shows  the seed map on top of the root map:"
    print "  - right click on a seed area to remove it"
    print "  - left  click + drag to draw the contour of new seed area"
    print "  - ctrl+z to undo"
    
    smb()
    return seed_map, sedit