import numpy                        as _np
from   scipy   import ndimage       as _nd

from rhizoscan         import geometry      as _geo
from rhizoscan.ndarray import reshape       as _reshape
from rhizoscan.image.measurements   import color_label as _color_label
from rhizoscan.datastructure        import Data    as _Data
from rhizoscan.datastructure        import Mapping as _Mapping

from rhizoscan.ndarray.graph import ArrayGraph as _ArrayGraph # used by SegmentGraph

from rhizoscan.tool import _property    
from rhizoscan.workflow import node as _node # to declare workflow nodes

"""
##TODO:
  - Node/Segment/AxeList implement the required set_node/segment/axe_list setter
  - RootGraph also, making sure node.segment, segment.node, axe.segment, 
                    segment.axe link well
  - Rename RootGraph/RootAxialTree to SegmentGraph/AxeTree ??
"""


class GraphList(_Mapping):
    ## del dynamic property when saved (etc...) / no direct access (get_prop)?
    @_property
    def properties(self):
        """ list of list properties (use add_property to add one)"""
        if not hasattr(self,'_property_names'):
            self._property_names = set()
        return list(self._property_names)
    def add_property(self, name, value):
        """
        Add the attribute 'name' with value 'value' to this object, and 'name' to
        this object 'properties' attribute. 
        """
        self.properties# assert existence
        self._property_names.add(name)
        self[name] = value

class NodeList(GraphList):
    def __init__(self,position=None, x=None, y=None):
        """
        Input:   Should give either a position array, or the x & y
            position: a 2xN array of the x,y coordinates of N nodes
            x&y:      2 length N vector array of the x,y coordinates
        """
        if x is not None and y is not None:
            self.position = _np.concatenate((x[None],y[None]), axis=0)
        else:
            self.position = position
            
        ##self.size = self.position.shape[1]-1   ## -1 should be removed ?
        
    @_property
    def x(self):  
        """ x-coordinates of nodes """  
        return self.position[0]
    @_property
    def y(self):
        """ y-coordinates of nodes """  
        return self.position[1]
    
    @_property
    def number(self):  
        """ number of nodes, including dummy (i.e. 0) node """  
        return self.position.shape[1]
        
    def set_segment(self, segment):
        if hasattr(segment,'node'):
            ns = [[] for i in xrange(self.number)]
            for s,sn in enumerate(segment.node[1:]):
                ns[sn[0]].append(s+1)
                ns[sn[1]].append(s+1)
            ns[0] = []
            self.segment = _np.array(ns,dtype=object)
        else:
            self.segment = segment
        
    @_property
    def terminal(self):
        """ bool flag, is node terminal """
        if not hasattr(self, '_terminal'):
            self._terminal = _np.vectorize(len)(self.segment)==1
            self.temporary_attribute.add('_terminal')
        return self._terminal
    
class SegmentList(GraphList):
    ##TODO Segmentlist: add a (private) link to node, and lake length, etc... class properties 
    def __init__(self, node_id, node_list):
        """
        Create a SegmentList from an Nx2 array of nodes pairs
        """
        self.node_list = node_list
        self.node = node_id
        ##self.size = node_id.shape[0]-1  ## -1 should be removed ?!
        
    @_property
    def number(self):  
        """ number of segments, including dummy (i.e. 0) segment """  
        return self.node.shape[0]
        
    @_property
    def length(self):
        """ Compute length of segments from NodeList 'node' """
        if not hasattr(self,'_length'):
            nx = self.node_list.x[self.node]
            ny = self.node_list.y[self.node]
            self._length = ((nx[:,0]-nx[:,1])**2 + (ny[:,0]-ny[:,1])**2)**.5
            self.temporary_attribute.add('_length')
        return self._length
    @length.setter
    def length(self, value):
        self._length = value
        self.clear_temporary_attribute('_length')
    
    @_property
    def direction(self):
        """ Compute direction of segments from NodeList 'node' """
        if not hasattr(self,'_direction'):
            sy = self.node_list.y[self.node]
            sx = self.node_list.x[self.node]
            dsx = _np.diff(sx).ravel()
            dsy = _np.diff(sy).ravel()
            self._direction = _np.arctan2(dsy,dsx)
            self.temporary_attribute.add('_direction')
        return self._direction
    @direction.setter
    def direction(self, value):
        self._direction = value
        self.clear_temporary_attribute('_direction')
        
    @_property
    def terminal(self):
        """ Compute terminal property of segments using attribute node_list """
        if not hasattr(self,'_terminal'):
            self._terminal = _np.any(self.node_list.terminal[self.node],axis=1)
            self.temporary_attribute.add('_terminal')
        return self._terminal
    @terminal.setter
    def terminal(self, value):
        self._terminal = value
        self.clear_temporary_attribute('_terminal')
       
    @_property
    def direction_difference(self):
        """ 
        Array of difference in direction between all segments in List
        
        This difference take into account by which node the segment are connected
        but angle diff for unconnected segment is meaningless
        """
        if not hasattr(self,'_direction_difference'):
            angle = self.direction
            dangle = _np.abs(angle[:,None] - angle[None,:])
            dangle = _np.minimum(dangle, 2*_np.pi-dangle)
            # segments sharing start or end nodes needs to be reverted
            to_revert = _np.any(self.node[:,None,:]==self.node[None,:,:],axis=-1)
            dangle[to_revert] = _np.pi - dangle[to_revert]
            dangle[0,:] = dangle[:,0] = _np.pi
            self._direction_difference = dangle
            self.temporary_attribute.add('_direction_difference')
        return self._direction_difference
    
    @_property
    def distance_to_seed(self):
        """ require the property 'length', 'order' and 'parent' """
        if not hasattr(self,'_distance_to_seed'):
            d2seed = self.length.copy()
            p = self.parent
            for i in self.order:
                d2seed[i] += d2seed[p[i]]
            self.temporary_attribute.add('_distance_to_seed')
        return self._distance_to_seed
        
        
    @property
    def neighbors(self):
        """ 
        Edges array of neighboring segments constructed with `neighbor_array`
        *** It requires the `seed` attribute ***
        """
        if not hasattr(self,'_neighbors'):
            nbor = neighbor_array(self.node_list.segment, self.node, self.seed)
            self._neighbors = nbor
            self.temporary_attribute.add('_neighbors')
        return self._neighbors
    @neighbors.setter
    def neighbors(self, value):
        self.clear_temporary_attribute('_neighbors')
        if value is not None:
            self._neighbors = value
    
    def digraph(self, direction):
        """
        Create the digraph induced by `direction`
        
        `direction` should be a boolean vector array with length equal to the 
        segment number. True value means the segment direction reversed.
        
        Return a neighbor type array such that: 
          - neighbors[...,0] are the  incoming neighbors, and
          - neighbors[...,1] are the outcoming neighbors
        """
        # reverse edge direction
        node = self.node.copy()
        node[direction] = node[direction][...,::-1]
        nbor = neighbor_array(self.node_list.segment, node, self.seed)
            
        # remove edges that are invalid for a directed graph
        # 
        # switch: boolean array with same shape as `nbor` that has True value 
        # where (directed) connection through a neighbors edge requires a change 
        # of one of the segment direction. ie.:
        # 
        # for all edge (i,j) stored in `neighbors`, i.e. j in neighbors[i]: 
        #   - i & j are not in the same relative direction
        #   - i.e. is j a neighbor on side s of i, and i on side s of j ?
        #
        # neighbors that requires switch are invalid in the digraph
        switch = _np.zeros(nbor.shape, dtype=bool)
        sid    = _np.arange(nbor.shape[0])[:,None,None]
        switch[...,0] = (nbor[nbor[...,0],:,0]==sid).any(axis=-1) # side 0
        switch[...,1] = (nbor[nbor[...,1],:,1]==sid).any(axis=-1) # side 1
             
        nbor[switch] = 0
        
        return nbor

class AxeList(GraphList):
    def __init__(self, axes=None, order=None, plant=None, segment_list=None, segment_parent='parent'):
        """
        Create an AxeList instance.
        
        :Warning:
            For backward compatibility, it is possible to make an empty AxeList
            without argument, but this will most probably become deprecated
            
        :Inputs:
          - axes:
              A list or array, of the **sorted** list of segment id each 
              root axe contains. The first list (i.e. the 1st axe) should be
              empty: a *dummy* axe.
              This value is stored in this AxeList `segment` attribute
              The list should be in a decreasing priority order (see notes)
          - order:
              An array-like of the order of each axe. Same length as `axe`.
          - plant:
              An array-like of the plant id for each axe. Same length as `axe`.
          - segment_list:
              The SegmentList instance on which this AxeList is constructed.
          - segment_parent:
              The list of parent of all segment in `segment_list`. If a string 
              is given, the respective attribute of `segment_list` is used.
              See notes.
              
        :Notes:
            The AxeList constructor compute the *main axe* of each segment from
            `segment_list` base on the `order` argument then on the order of 
            appearance in input `axe` list.
            The array containing the id of the selected main axe for each 
            segment is stored in the attribute `segment_axe`.

            It is considered that the parent axe of an axe `a` is the main axe of
            the parent segment of the 1st segment of axe `a`.
        """
        if axes is None: 
            DeprecationWarning("AxeList constructor without argument is deprecated") ##
            return
        
        self.segment = _np.asarray(axes)
        self.order   = _np.asarray(order)
        self.plant   = _np.asarray(plant)
        self.set_segment_list(segment_list)
        
        if isinstance(segment_parent, basestring):
            self._segment_parent = segment_list[segment_parent]
        else:
            self._segment_parent = segment_parent
        
        # find id of main axe for all segments
        segment_axe  = _np.zeros(segment_list.number, dtype=int)
        #for aid,slist in enumerate(self.segment[::-1]):
        #    segment_axe[slist] = len(self.segment)-aid-1
        axe_priority = _np.argsort(order[1:])+1
        for o in axe_priority[::-1]:
            slist = self.segment[o]
            segment_axe[slist] = o
        self.segment_axe = segment_axe
        
   
    def set_segment_list(self,segment_list):
        self._segment_list = segment_list
        
    @_property
    def number(self):
        """ number of axes, including dummy (i.e. 0) axe """  
        return len(self.segment)
        
    @_property
    def segment_number(self):
        if not hasattr(self,'_segment_number'):
            sNum = _np.vectorize(len)(self.segment)
            self._segment_number = sNum
            self.temporary_attribute.add('_segment_number')
        return self._segment_number
        
    @_property
    def length(self):
        if not hasattr(self,'_length'):
            length = _np.vectorize(lambda slist: self._segment_list.length[slist].sum())(self.segment)
            self._length = length
            self.temporary_attribute.add('_length')
        return self._length
    @length.setter
    def length(self, value):
        self._length = value
        self.clear_temporary_attribute('_length')
        
    @_property
    def segment1(self):
        """ first segment of axe """
        if not hasattr(self,'_AxeList__segment1'):
            segment1 = _np.array([sl[0] if len(sl) else 0 for sl in self.segment])
            self.__segment1 = segment1
            self.temporary_attribute.add('_AxeList__segment1')
        return self.__segment1
    @_property
    def sparent(self):                    ## change this attrib name?
        """ parent segment of axe """
        if not hasattr(self,'_AxeList__sparent'):
            sparent = self._segment_parent[self.segment1]
            self.__sparent = sparent
            self.temporary_attribute.add('_AxeList__sparent')
        return self.__sparent
    @_property
    def insertion_angle(self):
        """ insertion angle axe """
        if not hasattr(self,'_AxeList__insertion_angle'):
            insertion_angle = self._segment_list.direction_difference[self.segment1,self.sparent]
            self.__insertion_angle = insertion_angle
            self.temporary_attribute.add('_AxeList__insertion_angle')
        return self.__insertion_angle
    
    def _compute_length_properties(self):
        # compute the axe length and arc length of segment w.r.t their axe
        arc_length = _np.array([[] for i in xrange(len(self.segment))])
        axe_length = _np.zeros(len(self.segment))
        segment_number = _np.zeros(len(self.segment),dtype=int)
        for i,slist in enumerate(self.segment):
            if len(slist)==0: continue
            slist = _np.asarray(slist)
            arcL  = _np.cumsum(self._segment_list.length[slist])
            arc_length[i] = arcL
            main_axe = self._segment_list.axe[slist]==i         # if axis are overloaping, update
            arc_length[slist[main_axe]] = arcL[main_axe]        # arc length if axe i is the segment "main" axe 
            axe_length[i] = arcL[-1]
            segment_number[i] = len(arcL)
        self.segment.add_property('axelength',arc_length)
        
    def get_node_list(self):
        """
        Return list of axes as a list of node
        """
        from scipy.sparse import csr_matrix as csr
        from scipy.sparse.csgraph import depth_first_order as dfo
        
        axe_node = []
        for seg_list in self.segment:
            if len(seg_list)==0: 
                axe_node.append([])
                continue
            
            seg_node = self._segment_list.node[seg_list]
            if seg_node.shape[0]==1:
                axe_node.append(seg_node[0])
            else:
                c = csr((_np.ones(2*seg_node.shape[0],dtype='uint8'),_np.hstack((seg_node[::-1].T,seg_node[:,::-1].T))))
                i = set(seg_node[0]).difference(seg_node[1]).pop()
                order = dfo(c,i, return_predecessors=False)
                axe_node.append(order)
                
        return axe_node

def neighbor_array(node_segment, segment_node, seed):
    """
    Create an edges array of neighboring segments
    
    :Inputs:
      - node_segment:
          An array of lists of all segment connected to each node
      - segment_node:
          An array of the nodes pairs connected to each segment
      - seed:
          The seed mask of segment that are the roots of the graph
    
    :Output:
        An array of shape (S,N,2) with S the number of segments, N the maximum
        number for neighbors per segment side and 2 for the 2 segment sides. 
        Each neighbors[i,:,k] contains the list of the neighboring segments ids 
        on side k of segment `i`.
        
        In order to be an array, the missing neighbors are set to 0
    """
    ns   = node_segment.copy()
    invalid_nodes = _np.vectorize(lambda nslist: (seed[nslist]>0).all())(node_segment)
    ns[invalid_nodes] = set()
    ns[0] = set()
    
    # construct nb1 & nb2 the neighbor array of all segments in direction 1 & 2
    nsbor = _np.vectorize(set)(ns)
    snbor = [(s1.difference([i]),s2.difference([i])) for i,(s1,s2) in enumerate(nsbor[segment_node])]
    
    edge_max = max(map(lambda edg:max(len(edg[0]),len(edg[1])),snbor))
    edge = _np.zeros((len(snbor),edge_max,2), dtype='uint32')
    for i,(nb1,nb2) in enumerate(snbor):
        edge[i,:len(nb1),0] = list(nb1)
        edge[i,:len(nb2),1] = list(nb2)
        
    return edge

class SegmentGraph(_ArrayGraph):
    """
    ArrayGraph for a segment list
    """
    def __init__(self,segment, node):
        """
        inputs: segment-list and a node-list
        """
        _ArrayGraph.__init__(self)
        nsbor = _np.asarray(node.segment)
        snbor = [set(s1).union(s2).difference([i]) for i,(s1,s2) in enumerate(nsbor[segment.node])]
        self.setNodes(getattr(segment,'length', _np.ones(segment.number)))
        self.setEdges(snbor)


class RootGraph(_Mapping):
    """
    A graph representation of roots system
    
    Basically a pair of NodeList and SegmentList

    ***** in development *****
    """
    def __init__(self, node=None,segment=None):
        if node    is not None: self.node    = node
        if segment is not None: self.segment = segment
        
            
    @_property
    def sgraph(self):
        """ SegmentGraph of these graph segments """
        if not hasattr(self,'_sgraph'):
            self._sgraph = SegmentGraph(self.segment,self.node)
            self.temporary_attribute.add('_sgraph')
        return self._sgraph
        
    # printing    
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments' % (self.node.number, self.segment.number)
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
            #nx = self.node.x
            #ny = self.node.y
            plt.axis(_np.array(bbox)[[0,1,3,2]])#[nx.min(),nx.max(),ny.max(),ny.min()])
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
            sc=_np.arange(self.segment.number)
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

        ##nxy  = _np.concatenate((self.node.x[:,None],self.node.y[:,None]),axis=-1)
        nxy = self.node.position
        if transform is not None:
            nxy = _geo.normalize(_geo.dot(transform,_geo.homogeneous(nxy[::-1])))[1::-1]
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


class RootAxialTree(RootGraph):
    def __init__(self,node, segment, axe=None, to_tree=0, to_axe=0, single_order1_axe=True):
        """
        Construct a RootAxialTree graph
        
        if axe is not given, i.e. only a general [node,segment] graph, then
            'segment' should have a 'seed' property
            if to_tree>0, call make_tree with method set to to_tree value 
            if to_axe>0,  call find_axes with method set to to_axe  value
                          (which call set_axes with 'single_order_1_axe' parameter)
        """
        RootGraph.__init__(self, node=node, segment=segment)
        self.axe = axe
    
    @_property
    def edges(self):
        """ array of segment neighbors for each segment.
        To have all row of equal length, the array is filled with 0
        """
        if not hasattr(self,'_edges'):
            self._edges = self.sgraph.edges.copy()
            self._edges[self._edges>self.sid.size] = 0  # replace invalid indices by 0
            self.temporary_attribute.add('_edges')
        return self._edges
        
    @_property
    def dtheta(self):
        """ direction difference relative w.r.t edges array """
        if not hasattr(self,'_dtheta'):
            s2 = self.edges
            s1 = _np.arange(s2.shape[0]).reshape(-1,1)
            self._dtheta = self.segment.direction_difference[s1,s2]
            self.temporary_attribute.add('_dtheta')
        return self._dtheta

    def plot(self, bg='k', ac=None, sc=None, max_shift=0, transform=None, cmap=None, corder=1, cstart=1, **kargs):
        """
        Plot tree on top of `bg`
        
        If `sc` is not None, call RootGraph.plot with it (and **kargs)
        
        Otherwise, use `ac` to select color. POssible values are
          -  None:   color is selected w.r.t. axe id
          - 'order': color is selected w.r.t. axe order
          - 'plant': color is selected w.r.t. axe plant id
          -  an array of shape (axe.number,)
        """
        if sc is not None:
            RootGraph.plot(self, bg=bg, sc=sc, transform=transform, cmap=cmap, corder=corder, cstart=cstart, **kargs)
            return
            
        from matplotlib import collections as mcol
        
        # manage color arguments
        if ac=='order':
            ac = self.axe.order
        elif ac=='plant':
            ac = self.axe.plant
        
        if ac is None: 
            ac=_np.arange(self.axe.number)
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

        # get nodes, transformed if necessary
        nxy = self.node.position
        if transform is not None:
            nxy = _geo.normalize(_geo.dot(transform,_geo.homogeneous(nxy[::-1])))[1::-1]
        nxy = nxy.T
        
        # make LineCollection
        axe_node = self.axe.get_node_list()
        
        shift  = _np.arange(len(axe_node)) % (2*max_shift+1) - max_shift
        line = [nxy[node_list]+[[sh,sh]] for node_list,sh in zip(axe_node,shift)]
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
        
    def to_mtg(self, radius='radius', sampling=1):
        from itertools import izip
        # create a MTG graph 
        from openalea.mtg import MTG
        from openalea.plantgl.all import Vector3 as V3
        
        
        # short-named variables
        # ---------------------
        seg = self.segment
        axe = self.segment.axe
        #sid = self.sid
        sn = self.segment.node
        nx = self.node.x
        ny = self.node.y
        # mask of segment to add to mtg
        m = (seg.axe>0)+(seg.seed>0)
        m[0] = False             
        
        radius = seg[radius] if radius in dir(seg) else _np.ones(seg.number)
        
        seed   = self.segment.seed.astype(int)
        parent = self.segment.parent.astype(int)
        seed[seed>200] = 0  ## bug where seed=254
        parent[seed>0] = -seed[seed>0]   # parent of seed segments are -plant_id
        
        # make seed mtg-elements
        # ----------------------
        # seed element for mtg have (virtual) negativ ids (-1 to -n with n = # of plants)
        # here we compute a position and radius for those as the bbox center
        # they are stored in an array using *positive* ids
        
        # sseg : mask of seed segments
        sseg = self.stype==_SEED
        # snodes : mask of nodes of seed segments 
        snodes = self.segment.node[sseg]
        # Compute the bbox of snodes grouped by seed id 
        s_id = seed[sseg][:,None]*_np.ones((1,2))
        suid = _np.unique(s_id)
        if suid[0]==0: suid = suid[1:]
        xmin = _nd.minimum(self.node.x[snodes],labels=s_id,index=suid)
        xmax = _nd.maximum(self.node.x[snodes],labels=s_id,index=suid)
        ymin = _nd.minimum(self.node.y[snodes],labels=s_id,index=suid)
        ymax = _nd.maximum(self.node.y[snodes],labels=s_id,index=suid)
        x = (xmin+xmax)/2.                  # seeds center x-coord
        y = (ymin+ymax)/2.                  # seeds center y-coord
        r = (xmax-xmin+ymax-ymin)/4.        # seed radius = mean of bbox dimension
        
        suid = -suid # set seed id tp negative  
        
        
        # map global segment index in masked subset
        #indmap = _np.zeros(axe.size,dtype=int)
        indmap = _np.cumsum(m)*m -1
        
        # select the 'end' node of the segment
        #  i.e. the node of the segment that is not a node of its parent
        #  in case there is no parent (seed), select the last
        #  in case no node is shared with parent then use last ##=> to avoid error, but is it usefull?
        snode = _np.array([_np.setdiff1d(n1,sn[p2])[-1] if p2>0 else n1[-1] for n1,p2 in izip(sn[m],parent[m])])
        
        # construct the children list of all segment
        #   required to add node in parent-first order 
        children = dict()
        gen_child = ((i,p) for i,p in enumerate(parent) if m[i])
        for i,p in gen_child:
            children.setdefault(p,[]).append(i)
            
        # list info to store by node
        info = dict()
        seg_id = _np.arange(parent.size)
        seed_axe = dict(zip(suid,suid))  # default axe property for seed element
        for si,pi,li,xi,yi,ai,ri in izip(seg_id[m],parent[m], seg.length[m],nx[snode],ny[snode],axe[m], radius[m]):
            if pi==0: continue
            if pi<0:
                # if parent is a seed elements
                if not children.has_key(si):   # fake: has no child
                    children[pi].remove(si)
                    continue
                    
                ai = axe[children[si][0]] # take the axe id of child
                if self.axe.order[ai]==1: # if order 1 axe:
                    edge = '>'            #   direct child of parent (seed) 
                    seed_axe[pi] = ai     #   give same axe property to parent
                else:
                    edge = '+'
            else:
                edge = '<' if (ai==axe[pi]) else '+'
                
            info[si] = dict(sparent=pi,sid=si,axe=ai,position=V3(xi,0,-yi), length=li, radius=ri,edge_type=edge)
        for sid,xi,yi,ri in izip(suid,x,y,r):
            info[sid] = dict(axe=sid,position=V3(xi,0,-yi), x=xi,y=yi,radius=ri, seed=True)
           
           
        # id in mtg for ids in axial tree
        # -------------------------------
        # mtg_id[atree_id] -> vid in mtg
        # also works for (negative) seed id
        mtg_id = dict((i,None) for i in xrange(self.segment.parent.size))
        mtg_id.update((i,None) for u in suid)
        
        
        # make the mtg
        # ------------
        g = MTG()
        # add all segments recursively starting at seed segments
        def add_children(pid):  ##todo: avoid recursivity, replace by push/pop in a 'to_process' stack  
            """"
            recursive add_node function that process in depth order
              > sid is supposed to be already in the mtg graph
            """
            ##print pid, children.get(pid,[])
            for child in children.get(pid,[]):
                inf = info[child]
                pn = snode[indmap[pid]]
                cn = snode[indmap[child]]
                x  = _np.linspace(nx[pn],nx[cn],sampling+1)[1:]
                y  = _np.linspace(ny[pn],ny[cn],sampling+1)[1:]
                p  = mtg_id[pid]
                for xi,yi in zip(x,y):
                    inf['x'] = xi
                    inf['y'] = yi
                    p = g.add_child(parent=p, **inf)
                mtg_id[child] = p
                add_children(child)

        # for each seed, create a scale 1 element
        # then call add children for the nodes (scale 2)
        for seed in suid:
            mtg_pid = g.add_component(g.root)                    # scale 1
            mtg_id[seed] = g.add_component(mtg_pid, **info[seed]) # scale 2
            add_children(seed)            # start from seeds
        
        #pos = g.property('position')
        #assert [pos[i] for i in g.vertices(scale=1)]
        
        return g
        
    # printing    
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments, %d axes' % (self.node.number, self.segment.number, self.axe.number)
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...


