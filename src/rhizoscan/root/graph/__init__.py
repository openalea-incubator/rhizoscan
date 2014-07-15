"""
Package that implement graph representation used in RhizoScan

The central representation are `RootGraph` and `RootTree` which are mostly 
storing a `NodeList`, a `SegmentList` and (for RootTree) an AxeList objects.

See submodule `nsa` for the Node/Segment/Axe-List class definition

See also the `conversion` module for other graph representation.
"""

import numpy                        as _np
from   scipy   import ndimage       as _nd

from rhizoscan         import geometry      as _geo
from rhizoscan.ndarray import reshape       as _reshape
from rhizoscan.image.measurements   import color_label as _color_label
from rhizoscan.datastructure        import Mapping as _Mapping

_aleanodes_ = []           # openalea wrapped packages


"""
TODO:                                                        
  - N/S/A List: required used of set_node/segment_list?
  - RootGraph/Tree constructor (re)set node.segment, segment.node, axe.segment?
  
  - make some workflow nodes for RG/RT constructor and plot?
"""

from rhizoscan.root.graph.nsa import NodeList, SegmentList, AxeList


class RootGraph(_Mapping):
    """
    A graph representation of roots system 
    
    It basically stores a pair of NodeList and SegmentList in `node` and 
    `segment` attributes. It also provide a `plot` method.
    """
    def __init__(self, node=None,segment=None):
        if node    is not None: self.node    = node
        if segment is not None: self.segment = segment
        
            
    # printing
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments' % (self.node.number(), self.segment.number())
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...
        
    # a ploting method
    # ----------------
    def plot(self,bg='k', transform=None, sc=None, cmap=None, corder=1, cstart=1, indices='valid', **kargs):
        """ plot the graph, require matplotlib """
        lcol,bbox = self._get_LineCollection(sc=sc, cmap=cmap, corder=corder, cstart=cstart, transform=transform, indices=indices, **kargs)
        self._plot_collection(bg,collection=lcol, bbox=bbox, transformed=transform is not None)
            
    def _plot_collection(self,bg,collection, bbox, transformed=False):
        from matplotlib import pyplot as plt
        if isinstance(bg,basestring):
            plt.cla()
            axis = None
        elif bg is None:
            axis = None
        else:
            plt.cla()
            plt.imshow(bg)
            axis = plt.axis()
        
        plt.gca().add_collection(collection)
        
        if axis is not None:
            plt.axis(axis)
        elif not transformed:
            plt.axis(_np.array(bbox)[[0,1,3,2]])
            plt.gca().set_aspect('equal')
            if bg is not None:
                plt.gca().set_axis_bgcolor(bg)
        else:
            plt.gca().autoscale_view(True,True,True)
            if isinstance(bg,basestring):
                plt.gca().set_aspect('equal')
                plt.ylim(plt.ylim()[::-1])
                plt.gca().set_axis_bgcolor(bg)
            plt.draw()
            
    def _get_LineCollection(self, sc=None, cmap=None, corder=1, cstart=1, transform=None, indices='valid', **kargs):
        from matplotlib import collections as mcol
        if sc is None: 
            sc=_np.arange(self.segment.number())
            color = _color_label(sc,cmap=cmap, order=corder, start=cstart)
        elif isinstance(sc,basestring):
            color = sc
        else:
            sc = _np.asarray(sc)
            if sc.ndim==1:
                if sc.dtype.kind=='f':
                    if sc.max()>1: sc = sc/sc.max()
                    color = _reshape(sc, (None,-3))
                else:
                    if sc.dtype.kind=='b': sc = sc+0
                    color = _color_label(sc,cmap=cmap, order=corder, start=cstart)
            else:
                color = sc

        nxy = self.node.position
        if transform is not None:
            nxy = _geo.normalize(_geo.dot(transform,_geo.homogeneous(nxy)))[:-1]#[::-1])))[1::-1]
        nxy = nxy.T
            
        if indices=='all':
            indices = slice(None)
        elif indices=='valid':
            indices = (self.segment.node!=0).all(axis=1)
            
        line = nxy[self.segment.node[indices]]
        colr = color if len(color)==1 else color[indices]
        lcol = mcol.LineCollection(line, color=colr, **kargs)
        
        line = line.reshape(-1,2)
        minb = line.min(axis=0)
        maxb = line.max(axis=0)
        
        return lcol, [minb[0], maxb[0], minb[1], maxb[1]]


class RootTree(RootGraph):
    def __init__(self,node, segment, axe):
        """
        Construct a RootTree graph from 
          - NodeList `node`
          - SegmentList `segment`
          - AxeList `axe`
        """
        RootGraph.__init__(self, node=node, segment=segment)
        self.axe = axe
    

    def plot(self, bg='k', ac='id', sc=None, max_shift=0, transform=None, cmap=None, corder=1, cstart=1, **kargs):
        """
        Plot tree on top of `bg`
        
        If `sc` is not None, call RootGraph.plot with it and all other arguments
        
        Otherwise, use `ac` to select color. POssible values are
          -  None:   color is selected w.r.t. axe index
          -  id:     color is selected w.r.t. axe id
          - 'order': color is selected w.r.t. axe order
          - 'plant': color is selected w.r.t. axe plant id
          -  an array of shape (axe.number(),)
        """
        if sc is not None:
            RootGraph.plot(self, bg=bg, sc=sc, transform=transform, cmap=cmap, corder=corder, cstart=cstart, **kargs)
            return
            
        from matplotlib import collections as mcol
        
        # manage color arguments
        if ac=='order':
            ac = self.axe.order()
        elif ac=='id':
            ac = self.axe.get_id()
        elif ac=='plant':
            ac = self.axe.plant
        
        if ac is None: 
            ac=_np.arange(self.axe.number())
            color = _color_label(ac,cmap=cmap, order=corder, start=cstart)
        elif isinstance(ac,basestring):
            color = ac
        else:
            ac = _np.asarray(ac)
            if ac.ndim==1:
                if ac.dtype.kind=='f':
                    if ac.max()>1: ac = ac/ac.max()
                    color = _reshape(ac, (None,-3))
                else:
                    if ac.dtype.kind=='b': ac = ac+0
                    color = _color_label(ac,cmap=cmap, order=corder, start=cstart)
            else:
                color = ac

        nxy = self.node.position
        if transform is not None:
            nxy = _geo.normalize(_geo.dot(transform,_geo.homogeneous(nxy)))[:-1]
        nxy = nxy.T
        
        # make LineCollection
        axe_node = self.axe.get_node_list()[0]
        
        ms = max_shift
        shiftx  = _np.arange(len(axe_node))    % (2*ms+1) - ms
        shifty  = _np.arange(len(axe_node))/ms % (2*ms+1) - ms
        line = [nxy[node_list]+[[sx,sy]] for node_list,sx,sy in zip(axe_node,shiftx,shifty)]
        lcol = mcol.LineCollection(line, color=color, **kargs)
        
        # compute bounding box
        slist = []
        map(slist.extend, self.axe.segment)
        nodes = self.node.position[:,self.segment.node[slist]].reshape(2,-1)
        minb  = nodes.min(axis=1)
        maxb  = nodes.max(axis=1)
        bbox = [minb[0], maxb[0], minb[1], maxb[1]]
        
        # plot the tree
        self._plot_collection(bg,collection=lcol, bbox=bbox)
        
    # printing    
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments, %d axes' % (self.node.number(), self.segment.number(), self.axe.number())
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...

    # check version at reload
    def __restore__(self):
        """ update axe version, just in case """
        self.axe._update_version()
        return self


class RootAxialTree(RootTree):
    """ Deprecated class kept for backward compatibility - use RootTree instead """
    def __restore__(self):
        """ update to RootTree objects """
        self.__class__ = RootTree
        return self.__restore__()

