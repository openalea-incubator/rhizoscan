import numpy                        as _np
from   scipy   import ndimage       as _nd

from rhizoscan         import geometry      as _geo
from rhizoscan.ndarray import reshape       as _reshape
from rhizoscan.image.measurements   import color_label    as _color_label
from rhizoscan.workflow             import Data as _Data
from rhizoscan.workflow             import Struct as _Struct

from rhizoscan.ndarray.graph import ArrayGraph as _ArrayGraph # used by SegmentGraph

from rhizoscan.tool import _property    
from rhizoscan.workflow.openalea import aleanode as _aleanode

class GraphList(_Struct): ## del dynamic property when saved (etc...) / no direct access (get_prop)?
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

class AxeList(GraphList):
    def set_segment_list(self,segment_list):
        self._segment_list = segment_list
        
    @_property
    def segment1(self):
        """ first segment of axe """
        if not hasattr(self,'_AxeList__segment1'):
            segment1 = _np.array([sl[0] if len(sl) else 0 for sl in self.segment])
            self.add_property('_AxeList__segment1', segment1)
            self.temporary_attribute.add('_AxeList__segment1')
        return self.__segment1
    @_property
    def sparent(self):
        """ parent segment of axe """
        if not hasattr(self,'_AxeList__sparent'):
            sparent = self._segment_list.parent[self.segment1]
            self.add_property('_AxeList__sparent', sparent)
            self.temporary_attribute.add('_AxeList__sparent')
        return self.__sparent
    @_property
    def insertion_angle(self):
        """ insertion angle axe """
        if not hasattr(self,'_AxeList__insertion_angle'):
            insertion_angle = self._segment_list.direction_difference[self.segment1,self.sparent]
            self.add_property('_AxeList__insertion_angle', insertion_angle)
            self.temporary_attribute.add('_AxeList__insertion_angle')
        return self.__insertion_angle
    

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
            
        self.size = self.position.shape[1]-1   ## -1 should be removed ?
        
    @_property
    def x(self):  
        """ x-coordinates of nodes """  
        return self.position[0]
    @_property
    def y(self):
        """ y-coordinates of nodes """  
        return self.position[1]
    
    def set_segment(self, segment):
        if hasattr(segment,'node'):
            ns = [[] for i in xrange(self.size+1)]
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
            terminal = _np.vectorize(len)(self.segment)==1
            self.add_property('_terminal', terminal)
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
        self.size = node_id.shape[0]-1  ## -1 should be removed ?!
        
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
            self.add_property('_direction_difference', dangle)
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
            self.add_property('_distance_to_seed', d2seed)
            self.temporary_attribute.add('_distance_to_seed')
        return self._distance_to_seed
        
        
    @property
    def neighbors(self):
        """ edges array of neighboring segments
        
        The edge dynamic property is an array of shape (S,N,2) with S the number 
        of segments, N the (maximum) number for neighbors per segment side and
        2 for the 2 segment sides. Each neighbors[i,:,k] contains the list of 
        the neighboring segments ids on side k of segment `i`.
        
        In order to be an array, the missing neighbors are set to 0
        
        *** It requires the `seed` attribute ***
        """
        if not hasattr(self,'_neighbors'):
            seed = self.seed
            node = self.node_list
            ns   = node.segment.copy()
            invalid_nodes = _np.vectorize(lambda nslist: (self.seed[nslist]>0).all())(node.segment)
            ns[invalid_nodes] = set()
            ns[0] = set()
            
            # construct nb1 & nb2 the neighbor array of all segments in direction 1 & 2
            nsbor = _np.vectorize(set)(ns)
            snbor = [(s1.difference([i]),s2.difference([i])) for i,(s1,s2) in enumerate(nsbor[self.node])]
            
            # 
            edge_max = max(map(lambda edg:max(len(edg[0]),len(edg[1])),snbor))
            edge = _np.zeros((len(snbor),edge_max,2), dtype='uint32')
            for i,(nb1,nb2) in enumerate(snbor):
                edge[i,:len(nb1),0] = list(nb1)
                edge[i,:len(nb2),1] = list(nb2)
                
            self._neighbors = edge
            self.temporary_attribute.add('_neighbors')
        return self._neighbors
    @neighbors.setter
    def neighbors(self, value):
        self.clear_temporary_attribute('_neighbors')
        if value is not None:
            self._neighbors = value
    
    @_property
    def nbor_switch_dir(self):
        """
        Tells if `neighbors` edges requires a switch of direction
        
        This boolean array with same shape as `neighbors` has True value where 
        (directed) connection through a `neighbors` edge requires a change of 
        one of the segment direction.
        
        for all edge (i,j) stored in `neighbors`, i.e. j in neighbors[i]: 
          - i & j are not in the same relative direction
          - i.e. is j a neighbor on side s of i, and i on side s of j ?
               
        *** requires the `neighbors` property ***
        """
        if not hasattr(self,'_nbor_switch_dir'):
            nbor = self.neighbors
            nbor_switch = _np.zeros(nbor.shape, dtype=bool)
            sid = _np.arange(nbor.shape[0])[:,None,None]
            nbor_switch[...,0] = (nbor[nbor[...,0],:,0]==sid).any(axis=-1) # side 0
            nbor_switch[...,1] = (nbor[nbor[...,1],:,1]==sid).any(axis=-1) # side 1
            self._nbor_switch_dir = nbor_switch
            self.temporary_attribute.add('_nbor_switch_dir')
        return self._nbor_switch_dir
    @nbor_switch_dir.setter
    def nbor_switch_dir(self, value):
        self.clear_temporary_attribute('_nbor_switch_dir')
        if value is not None:
            self._nbor_switch_dir = value
    
    def clear_neighbors(self):
        """ Reset the `neighbors` and related `nbor_switch_dir` atttributes """
        self.neighbors = None
        self.nbor_switch_dir = None

    def switch_direction(self, direction, digraph=False):
        """
        Change segment direction given by `direction` (switch node ids)
        
        *** It also reset the `neighbors` and `nbor_switch_dir` attributes ***
        
        `direction` should be a boolean vector array with length equal to the 
        segment number. True value means the segment direction should be switched
        
        if `digraph` is True, remove all edges of the (updated) `neighbors` 
        attribute for which the respective `nbor_switch_dir` is True:
        The graph obtained become a valid directed graph where 
          - neighbors[...,0] are the  incoming neighbors, and
          - neighbors[...,1] are the outcoming neighbors
        """
        # reverse edge direction
        self.node[direction] = self.node[direction][...,::-1]
        self.clear_neighbors()
        
        if digraph:
            # remove edges that are invalid for a directed graph
            self.neighbors[self.nbor_switch_dir] = 0
            
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
        self.setNodes(getattr(segment,'length', _np.ones(segment.size+1)))
        self.setEdges(snbor)


class RootGraph(_Struct):
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
        return self.__class__.__name__ + ': %d nodes, %d segments' % (self.node.size, self.segment.size)
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...
        
    # a ploting method
    # ----------------
    def plot(self,bg='k', transform=None, sc=None, cmap=None, corder=1, cstart=1, indices='valid', **kargs):
        """ plot the graph, require matplotlib """
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
        
        lcol,bbox = self._get_LineCollection(sc=sc, cmap=cmap, corder=corder, cstart=cstart, transform=transform, indices=indices, **kargs)
        plt.gca().add_collection(lcol)
        
        if axis is not None:
            plt.axis(axis)
        elif transform is None:
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
            sc=_np.arange(self.segment.size+1)
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


_UNREACHABLE = -2
_SEED        = -1
_UNSET       = 0
_SET         = 1
##todo: remove RootAxialTree.stype, but add 'seed' ?
##todo fully restructure rootAxialTree (subcalss of RootGraph)
##      or simple delete it, add make function to compute a axe attribute to a RG

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
        seed = self.segment.seed
        seed[_np.any(self.segment.node==0,axis=1)] = _UNREACHABLE
        self.sid = _np.arange(self.segment.size+1)
        self.pid = _nd.minimum(self.sid,seed,_np.arange(seed.max()+1))  # plant(seed) unique id
        self.sid[seed>0] = self.pid[seed[seed>0]]
        self.pid = self.pid[1:]
        
        self.stype   = _np.zeros(self.sid.size,dtype=int)    # 0:UNSET
        self.stype[seed>0] = _SEED
        self.stype[seed<0] = _UNREACHABLE
        
        if to_tree:
            self.make_tree(method=to_tree)
            if to_axe: 
                self.find_axes(method=to_axe, single_order1_axe=single_order1_axe)
                #self.set_axes(single_order1_axe=single_order1_axe)
    
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
                
    def make_tree(self, method): 
        """
        method: 1 for length, 2 for direction, 3 for both
        """
        # compute cost to compute shortest-path-tree
        tree_cost  = 1
        if method&1: tree_cost  = self.sgraph.makeDistanceMap('nodes')
        if method&2: tree_cost *= 1 - _np.cos(self.dtheta) + 2**-20
        
        # compute shortest tree
        self.sgraph.shortestPath(_np.nonzero(self.stype==_SEED)[0],distanceMap=tree_cost)#,distanceMap2=self.sgraph.makeDistanceMap('nodes'))
        self.segment.order  = _np.argsort(self.sgraph.distance)  ## make it a tmp property, computed using segment.parent ?

        # set segment parent id, and update segment type
        self.segment.parent = self.sid[self.sgraph.parent]
        self.stype[self.stype>=_UNSET]     = _UNSET
        self.segment.parent[self.stype==_SEED]    = 0   # seeds have no parent
        self.stype[self.segment.parent==self.sid] = _UNREACHABLE
        self.segment.parent[self.stype==_UNREACHABLE] = 0

    def find_axes(self, method, single_order1_axe=True):
        """
        method = 1 for length, 2 for direction, 3 for both
        method = 'arabido' => use find_arabido_axes
        """
        if method=='arabido':
            self.find_arabido_axes()
            return
            
        # find terminal segment
        sterm = self.segment.terminal
        
        # compute cost/gain to compute shortest-tree/axe label respectively
        axial_gain = 1
        if method&1: axial_gain  = self.sgraph.makeDistanceMap('nodes')
        if method&2: axial_gain *= 1 + _np.cos(self.dtheta)

        gain = self.segment.length.copy()#_np.zeros(self.sid.size)
        gain[0] = 0
        for i in self.segment.order:
            #if self.segment.parent[i]!=0 and gain[i]==0:
            #    set_dist(i)
            if self.segment.parent[i]!=0:
                p = self.segment.parent[i]
                j = _np.argmax(self.edges[p,:]==i)
                gain[i] += gain[p] + axial_gain[p,j]
        
        # axes of segments: 
        # -----------------
        #    init: unique id for all terminal segments
        self.segment.axe   = _np.cumsum(sterm)*sterm
        self.segment.axe[self.stype==_UNREACHABLE] = _UNREACHABLE
        self.segment.axe[self.stype==_SEED] = _SEED
        axes = self.segment.axe
        
        # for all unset segments, select it iteratively starting further away from seeds
        index = self.segment.order[::-1]  # indices of segments in decreasing order for their distance
        #index = _np.argsort(gain)[::-1]        
        index = index[axes[index]==0]     # keep only reachable segments with no axes selected  

        for i in index:
            nb = self.edges[i]
            nb = nb[nb>0]
            nb = nb[self.segment.parent[nb]==i]
            
            if nb.size==0:
                axes[i]=axes.max()+1
                ##print 'new axe needed:', i,axes[i]
            else:
                j = _np.argmax(gain[nb])
                axes[i] = axes[nb[j]]
                gain[i] = gain[nb[j]]
                
        self.axial_gain = gain
        self.stype[axes>_UNSET] = _SET
        
        self.set_axes(single_order1_axe=single_order1_axe)
        
    def set_axes(self,s_axe=None, s_parent=None, a_segment=None, single_order1_axe=True):
        """ 
        Finalise creation of root axe - automatically called by find_axes
        
        :Input:
        s_axe:     array of the "main" axe id of all segment   - main if there is overlapping
                   if None, use self.segment.axe
        s_parent:  the parent segment of all segment
                   if None, use self.segment.parent
        a_segment: list, for all axes, of the their (sorted) segment list
                   if None, compute it considering the graph does not contain loop
        single_order1_axe: if True, keep only the longest axe touching a seed as the main axe 
        """
            
        # manage axe & parent arguments
        if s_axe is None: s_axe = self.segment.axe
        else:             self.segment.add_property('axe',    s_axe)
        if s_parent is None: s_parent = self.segment.parent
        else:                self.segment.add_property('parent', s_parent)
        
            
        # make the axe structure
        # ----------------------
        self.axe = AxeList()
        axe = self.axe
        axe.set_segment_list(self.segment)
        
        # set axe.segment, or compute it
        if a_segment is None:
            a_segment = [[] for i in range(s_axe.max()+1)]
            direct_child = (s_axe==s_axe[s_parent])
            ends = _np.setdiff1d(self.sid,s_parent[direct_child])  # segment with no parent
            ends = ends[s_axe[ends]>0]                             # don't process segment with unset axes (unreachable, seed?)
            for sid in ends:
                aid = s_axe[sid]
                slist = []
                while s_axe[sid]==aid:
                    slist.append(sid)
                    sid = s_parent[sid]
                a_segment[aid].extend(slist[::-1])
            
        axe.segment = _np.array(a_segment,dtype=object)
        
        # compute the axe length and arc length of segment w.r.t their axe
        # ----------------------------------------------------------------
        arc_length = _np.zeros_like(self.segment.length)
        axe.length = _np.zeros(len(axe.segment))
        axe.size   = _np.zeros(len(axe.segment),dtype=int)   ## doesn not respect GraphList standart: size is he number of elements (-1) !
        for i,slist in enumerate(axe.segment):
            if len(slist)==0: continue
            slist = _np.asarray(slist)
            arcL = _np.cumsum(self.segment.length[slist])
            main_axe = self.segment.axe[slist]==i            # if axis are overloaping, update
            arc_length[slist[main_axe]] = arcL[main_axe]     # arc length if axe i is the segment "main" axe 
            axe.length[i] = arcL[-1]
            axe.size[i]   = len(arcL)
        self.segment.add_property('axelength',arc_length)
        
        # compute the axes parent, order, and plant id 
        # --------------------------------------------
        axe.parent = _np.array([_UNREACHABLE if len(sl)==0 else s_axe[s_parent[sl[0]]] for sl in axe.segment])
        order = _np.zeros(len(axe.segment),dtype=int)
        plant = _np.zeros(len(axe.segment),dtype=int)
        
        o1 = (axe.parent==_SEED).nonzero()[0]
        order[o1] = 1
        plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
        
        if single_order1_axe:
            plant_id = _np.unique(plant[o1])
            plant_id = plant_id[plant_id>0]
            for pid in plant_id:
                main = (plant==pid).nonzero()[0]
                if len(main)>1:
                    second = main[axe.length[main]<_np.max(axe.length[main])]
                    order[second] = 2
        
        ax_unset = (order==_UNSET) & (axe.parent>0)
        unset_tot = -1
        while _np.any(ax_unset) and (unset_tot!=_np.sum(ax_unset)):
            unset_tot = _np.sum(ax_unset)
            #print order
            #print axe.parent
            p_order = order[axe.parent[ax_unset]]
            p_plant = plant[axe.parent[ax_unset]]
            order[ax_unset] = _np.choose(p_order==_UNSET,(p_order+1,_UNSET))
            plant[ax_unset] = p_plant
            ax_unset = (order==_UNSET) & (axe.parent>0)
            
        axe.order = order
        axe.plant = plant
        
    def find_arabido_axes(self): 
        """ IN DEVELOPMENT: find axes allowing overlap, for arabido model """
        sterm  = self.segment.terminal
        sseed  = self.segment.seed
        parent = self.segment.parent
        d2seed = self.segment.distance_to_seed

        sseed[sseed>=254] = 0     ## there is some bug that make unreachable 254 instead of 0
        
        # construct all axes
        # ------------------
        s_axe = [[] for i in range(sseed.size)]  # list of axes for each segment
        a_seg = [[0]]                            # list of segments for each axe
        a_pid = [ 0 ]                            # plant id (seed) of each axe
        s_axe[0] = [0]                           #    bg(0) axe = [bg segment]
        
        # First pass: find all (order 1) axes that cover all segments
        #   process terminal segments first, then from further away
        #   for each (not already set) segment, store the path to the seed 
        order = _np.argsort(d2seed + d2seed.max()*sterm)[::-1] 
        for sid in _np.arange(sseed.size)[order]:
            if len(s_axe[sid])>0 or sseed[sid]>0: 
                continue
            cur_axe = len(a_seg)   # id of new axe
            ax_seg  = []
            while sseed[sid]==0:
                ax_seg.append(sid)
                sid = parent[sid]
                if sid==0: break
                
            if sid>0: # new axe only if it reach somewhere
                # add seed segment to axe (?)
                ##ax_seg.append(sid)
                # add axe id to all path segments
                for s in ax_seg:
                    s_axe[s].append(cur_axe)
                # append axe
                a_seg.append(ax_seg)
                a_pid.append(sseed[sid])
                
        a_seg[0] = [0]
        a_pid    = _np.array(a_pid)
        
        # compute length of axe
        slength = self.segment.length
        a_length = _np.vectorize(lambda sid: slength[sid].sum())(a_seg)
        
        # set the main axe of each segment as the longest that pass through it
        for sid, alist in enumerate(s_axe):
            if len(alist)==0: 
                s_axe[sid] = 0
            else:
                s_axe[sid] = alist[_np.argmax(a_length[alist])]
        s_axe = _np.array(s_axe)
        
        # for each plant id, find the longest axe and make it primary
        ##   redondant with the single_order1_axe of set_axes...
        a_order = _np.ones(a_pid.size)*2
        a_pid[a_pid>=254] = 0     ## there is some bug that make unreachable 254 instead of 0
        upid = _np.unique(a_pid)
        for pid in upid[upid>0]:
            mask = a_pid==pid
            main = mask.nonzero()[0][_np.argmax(a_length[mask])]          
            a_order[main] = 1
        
        # remove all segments that are in one of the main axe (order 1
        #    and inverse list order to be seed to tip
        #reduce_axe = lambda slist: slist[_np.cumprod(a_order[s_axe[slist]]>1)]
        #a_seg = _np.vectorize(reduce_axe)(a_seg)
        for aid, slist in enumerate(a_seg):
            slist   = _np.array(slist)
            slist   = slist[a_order[s_axe[slist]]==a_order[aid]]
            #sl_mask = (_np.cumprod(a_order[s_axe[slist]]==a_order[aid])>0)
            #slist   = slist[sl_mask]
            
            a_seg[aid] = slist[::-1] #[sid for i,sid in enumerate(slist) if sl_mask[i]]
            
        a_seg = _np.array(a_seg)
        
        s_axe[sseed>0] = _SEED
        ################################
        ## adapted copy of set_axes... #
        ################################
        s_parent = self.segment.parent
        self.segment.add_property('axe',    s_axe)
        self.axe = AxeList()
        self.axe.set_segment_list(self.segment)
        self.axe.segment = _np.array(a_seg,dtype=object)
        axe = self.axe
        
        # compute the axe length and arc length of segment w.r.t their axe
        # ----------------------------------------------------------------
        arc_length = _np.zeros_like(self.segment.length)
        axe.length = _np.zeros(len(axe.segment))
        axe.size   = _np.zeros(len(axe.segment),dtype=int)   ## doesn not respect GraphList standart: size is he number of elements (-1) !
        for i,slist in enumerate(axe.segment):
            if len(slist)==0: continue
            arcL = _np.cumsum(self.segment.length[slist])
            main_axe = self.segment.axe[slist]==i            # if axis are overloaping, update
            arc_length[slist[main_axe]] = arcL[main_axe]     # arc length if axe i is segment "main" axe 
            axe.length[i] = arcL[-1]
            axe.size[i]   = len(arcL)
        self.segment.add_property('axelength',arc_length)
        
        # compute the axes parent, order, and plant id 
        # --------------------------------------------
        ## w.r.t set_axes: here seed axes are part or axes
        #     thus parent[axe_seg[0]] is always 0
        axe.parent = _np.array([_UNREACHABLE if len(sl)==0 else s_axe[s_parent[sl[0]]] for sl in axe.segment])
        ##plant = self.segment.seed[[sl[0] for sl in t.axe.segment]]
        order = _np.zeros(len(axe.segment),dtype=int)
        plant = _np.zeros(len(axe.segment),dtype=int)
        
        o1 = (axe.parent==_SEED).nonzero()[0]
        order[o1] = 1
        plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
        ##o1 = plant>0
        ##order[o1] = 1
        ##plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
        
        if 1: ##single_order1_axe:
            plant_id = _np.unique(plant[o1])
            plant_id = plant_id[plant_id>0]
            for pid in plant_id:
                main = (plant==pid).nonzero()[0]
                if len(main)>1:
                    second = main[axe.length[main]<_np.max(axe.length[main])]
                    order[second] = 2
        
        ax_unset = (order==_UNSET) & (axe.parent>0)
        unset_tot = -1
        while _np.any(ax_unset) and (unset_tot!=_np.sum(ax_unset)):
            unset_tot = _np.sum(ax_unset)
            #print order
            #print axe.parent
            p_order = order[axe.parent[ax_unset]]
            p_plant = plant[axe.parent[ax_unset]]
            order[ax_unset] = _np.choose(p_order==_UNSET,(p_order+1,_UNSET))
            plant[ax_unset] = p_plant
            ax_unset = (order==_UNSET) & (axe.parent>0)
            
        axe.order = order
        axe.plant = plant
        
    def plot(self, bg=None, sc='axe', **kargs):
        if sc is 'axe':
            sc = self.segment.axe
        elif sc=='order':
            sc = self.axe.order[self.segment.axe]
            sc[self.segment.axe<0] = -1
        elif sc=='plant':
            sc = self.axe.plant[self.segment.axe]
            sc[self.segment.axe<0]  = -1
            sc[self.segment.seed>0] = self.segment.seed[self.segment.seed>0]
            
        RootGraph.plot(self, bg=bg, sc=sc, **kargs)
            
    def to_mtg(self):
        from itertools import izip
        # create a MTG graph 
        from openalea.mtg import MTG
        #from openalea.plantgl.all import Vector3 as V3
        g = MTG()
        
        # mtg_id : id -> vid (mtg) 
        mtg_id = _np.zeros(self.segment.parent.size, dtype=int)
        
        # add seed nodes
        
        # seed : roots mask
        seed = self.stype==_SEED
        # lnodes : root vertices 
        lnodes = self.segment.node[seed]
        # sid : id of root vertices 
        # Compute the bbox of each vertex root in the image 
        b_id = self.sid[seed][:,None]*_np.ones((1,2))
        buid = _np.unique(b_id)
        xmin = _nd.minimum(self.node.x[lnodes],labels=b_id,index=buid)
        xmax = _nd.maximum(self.node.x[lnodes],labels=b_id,index=buid)
        ymin = _nd.minimum(self.node.y[lnodes],labels=b_id,index=buid)
        ymax = _nd.maximum(self.node.y[lnodes],labels=b_id,index=buid)
        x = (xmin+xmax)/2.                  # seeds center x-coord
        y = (ymin+ymax)/2.                  # seeds center y-coord
        r = (xmax-xmin+ymax-ymin)/4.        # mean of x & y radius
        #for id,xi,yi,ri in izip(buid,x,y,r):
            #mtg_id[id] = g.add_component(g.root)
            # TODO: parent of a root need to be None or something explicit
            
        # add axe nodes
        m = self.stype>=_UNSET
        seg = self.segment
        axe = self.segment.axe
        sid = self.sid
        parent = self.segment.parent
        sn = self.segment.node
        nx = self.node.x
        ny = self.node.y
        
        # map global segment index in masked subset
        #indmap = _np.zeros(axe.size,dtype=int)
        indmap = _np.cumsum(m)*m -1
        
        # select the 'end' node of the segment
        #  i.e. the node of the segment that is not a node of its parent
        #  in case no node is shared with parent (seeds) then use last
        snode = _np.array([_np.setdiff1d(n1,n2)[-1] for n1,n2 in izip(sn[m],sn[parent[m]])])
        # if parent node no found (1st segment of 1st s_axe) set the segment 1st node
        ##pnode = _np.array([pn[0] if len(pn)>0 else sni[0] for sni,pn in izip(sn[m],pnode)])
        
        # construct the children list of all segment
        #   required to add node in parent-first order 
        children = dict()
        gen_child = ((i,p) for i,p in enumerate(parent) if p>_UNSET)
        for i,p in gen_child:
            children.setdefault(p,[]).append(i)
        
        # list info to store by node
        info = dict()
        for si,pi,li,xi,yi,ai,ri in izip(sid[m],parent[m], seg.length[m],nx[snode],ny[snode],axe[m], seg.radius[m]):
            edge = '<' if (ai==axe[pi]) else '+'
            info[si] = dict(sparent=pi,sid=si,axe=ai,position=(xi,yi,0), length=li, radius=ri,edge_type=edge)
        for seed,xi,yi,ri in izip(buid,x,y,r):
            info[seed] = dict(sid=si,axe=ai,position=(xi,yi,0), x=xi,y=yi,radius=ri, seed=True)
            
        sampling = 1
        # add all segments recursively starting at seed segments
        def add_children(pid):  ##todo: avoid recursivity, replace by push/pop in a 'to_process' stack  
            """"
            recursive add_node function that process in depth order
              > sid is supposed to be already in the mtg graph
            """
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

        for seed in buid:
            plant_id = g.add_component(g.root)
            mtg_id[seed] = g.add_component(plant_id, **info[seed])
            add_children(seed)            # start from seeds
        
        #pos = g.property('position')
        #assert [pos[i] for i in g.vertices(scale=1)]
        
        return g
        
    # printing    
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments, %d axes' % (self.node.size, self.segment.size, len(self.axe.size))
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...

@_aleanode('root_tree')
def make_RootAxialTree(node, segment, to_tree, to_axe, single_axe1):
    """
    Aleanode function which call RootAxialTree constructor
    """
    return RootAxialTree(node=node, segment=segment, to_tree=to_tree, to_axe=to_axe, single_order1_axe=single_axe1)
    
