"""
Package that contains the definition of the NodeList, SegmentList and AxeLIst
used by RootGraph
"""
import numpy as _np

from rhizoscan.misc.decorators import _property
from rhizoscan.datastructure import Mapping as _Mapping

from rhizoscan.root.graph.conversion import segment_to_neighbor as _seg2nbor

class GraphList(_Mapping):
    ## del dynamic property when saved (etc...) / no direct access (get_prop)?
    def properties(self):
        """ list of list-properties (use add_property to add one)"""
        if not hasattr(self,'_property_names'):
            self._property_names = set()
        return list(self._property_names)
    def add_property(self, name, value):
        """
        Add the attribute 'name' with value 'value' to this object, and 'name' to
        this object 'properties' attribute. 
        """
        self.properties()# assert existence
        self._property_names.add(name)
        self[name] = value
   
##TODO: test & use the below decorator for method with buffered outputs
def buffered(method):
    """ decorator to stores outputs of `method` in temporal_attribute """
    tmp_name = '_'+func.__name__
    
    def buffered_method(self):
        if not hasattr(self,tmp_name):
            setattr(self,tmp_name, func(self))
            self.temporary_attribute.add(tmp_name)
        return getattr(self,tmp_name)
        
    tmp_func.__name__ = func.__name__
    tmp_func.__doc__ = func.__doc__
    
    return tmp_func
    
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
        
    def x(self):  
        """ x-coordinates of nodes """  
        return self.position[0]
    def y(self):
        """ y-coordinates of nodes """  
        return self.position[1]
    
    def number(self):  
        """ number of nodes, including dummy (i.e. 0) node """  
        return self.position.shape[1]
        
    def set_segment(self, segment):
        if hasattr(segment,'node'):
            ns = [[] for i in xrange(self.number())]
            for s,sn in enumerate(segment.node[1:]):
                ns[sn[0]].append(s+1)
                ns[sn[1]].append(s+1)
            ns[0] = []
            self.segment = _np.array(ns,dtype=object)
        else:
            self.segment = segment
        
    def terminal(self):
        """ array of bool indicating if node is terminal """
        if not hasattr(self, '_terminal'):
            self._terminal = _np.vectorize(len)(self.segment)==1
            self.temporary_attribute.add('_terminal')
        return self._terminal
    
class SegmentList(GraphList):
    def __init__(self, node_id, node_list):
        """
        Create a SegmentList from an Nx2 array of nodes pairs
        """
        self.node_list = node_list
        self.node = node_id
        ##self.size = node_id.shape[0]-1  ## -1 should be removed ?!
        
    def number(self):  
        """ number of segments, including dummy (i.e. 0) segment """  
        return self.node.shape[0]
        
    def length(self):
        """ length of segments """
        if not hasattr(self,'_length'):
            nx = self.node_list.x()[self.node]
            ny = self.node_list.y()[self.node]
            self._length = ((nx[:,0]-nx[:,1])**2 + (ny[:,0]-ny[:,1])**2)**.5
            self.temporary_attribute.add('_length')
        return self._length
        
    def set_length(self, length):
        """ set the length of segment (replace automatically computed one) """
        self.clear_temporary_attribute('_length')
        self._length = length
    
    def direction(self):
        """ direction of segments as an (radian) angle """
        if not hasattr(self,'_direction'):
            sy = self.node_list.y()[self.node]
            sx = self.node_list.x()[self.node]
            dsx = _np.diff(sx).ravel()
            dsy = _np.diff(sy).ravel()
            self._direction = _np.arctan2(dsy,dsx)
            self.temporary_attribute.add('_direction')
        return self._direction
        
    def terminal(self):
        """ Bool array indicating which segments are terminal """
        if not hasattr(self,'_terminal'):
            self._terminal = _np.any(self.node_list.terminal()[self.node],axis=1)
            self.temporary_attribute.add('_terminal')
        return self._terminal
       
    def direction_difference(self):
        """ 
        Array of difference in direction between all segments in List
        
        This difference take into account by which node the segment are connected
        but angle diff for unconnected segment is meaningless
        """
        if not hasattr(self,'_direction_difference'):
            angle = self.direction()
            dangle = _np.abs(angle[:,None] - angle[None,:])
            dangle = _np.minimum(dangle, 2*_np.pi-dangle)
            # segments sharing start or end nodes needs to be reverted
            to_revert = _np.any(self.node[:,None,:]==self.node[None,:,:],axis=-1)
            dangle[to_revert] = _np.pi - dangle[to_revert]
            dangle[0,:] = dangle[:,0] = _np.pi
            self._direction_difference = dangle
            self.temporary_attribute.add('_direction_difference')
        return self._direction_difference
        
    def neighbors(self):
        """ 
        Returns the `neighbor` segment graph for this SegmentList
        
        See the `rhizoscan.root.graph.conversion` module and its 
        `segment_to_neighbor` function for details.
        
        If this SegmentList contains a `seed` attribute, connection between
        seed segments are not taken into account
        """
        if not hasattr(self,'_neighbors'):
            nbor = _seg2nbor(self.node, self.node_list.segment, self.get('seed',None))
            self._neighbors = nbor
            self.temporary_attribute.add('_neighbors')
        return self._neighbors
        
    def reachable(self):
        """
        Return a boolean mask of segments connected to seeds
        """
        if not hasattr(self,'_reachable'):
            from scipy.sparse.csgraph import connected_components
            from rhizoscan.root.graph.conversion import neighbor_to_csgraph
            
            nbor = self.neighbors()
            cs = neighbor_to_csgraph(nbor)
            n, lab = connected_components(cs)
            reachable_lab = _np.zeros(n,dtype=bool)
            reachable_lab[lab[self.seed>0]] = 1
            self._reachable = reachable_lab[lab]
            self.temporary_attribute.add('_reachable')
        return self._reachable

        
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
        nbor = _seg2nbor(node, self.node_list.segment, self.get('seed',None))
            
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
    def __init__(self, axes, segment_list, parent, plant=None, order=None, parent_segment='parent', ids=None):
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
          - segment_list:
              The SegmentList instance from which this AxeList is constructed
          - parent:
              id of the parent axe, as an array of same length as `axes`
              ##todo: make parent mandatory
          - plant:
              An array-like of the plant id for each axe. Same length as `axe`
              ##todo: plant - provide an automatic method to compute it
          - order:
              An array-like of the order of each axe. Same length as `axe`
              ##todo: order - provide an automatic method to compute it
          - parent_segment:
              The list of the parent segment of all axes. If a string is given, 
              use the the attribute with the name of `segment_list` to infer it.
              See notes.
              ##todo: remove default 'parent' value of parent_segment arg
          - ids:
              optional axe ids. Must be >0 for all axes but dummy
              If None, set don't set axe id. If 'auto', set default ones. 
              
        :Notes:
            The AxeList constructor compute the "main axe" of each segment from
            `segment_list` base on the `order` argument then on the order of 
            appearance in input `axe` list.
            The array containing the id of the selected main axe for each 
            segment is stored in the attribute `segment_axe`.

            It is considered that the parent axe of an axe `a` is the main axe of
            the parent segment of the 1st segment of axe `a`.
            
        ##TODO: AxeList - remove automatic parent computation 
        ##              - but can provide a practical external method 
        """
        self.segment = _np.asarray(axes)
        self.plant   = _np.asarray(plant)
        
        self._segment_list = segment_list
        
        self.parent = _np.asarray(parent)
        self.parent_segment = _np.asarray(parent_segment)
        ##if isinstance(parent_segment, basestring):
        ##    self.parent_segment = segment_list[parent_segment][self.first_segment()]
        ##else:
        ##    self.parent_segment = _np.asarray(parent_segment)
            
        # set axe order if given, see 'order' property 
        if order is not None:
            self._order = _np.asarray(order)
        
        # find id of main axe for all segments
        segment_axe  = _np.zeros(segment_list.number(), dtype=int)
        axe_priority = _np.argsort(self.order()[1:])+1
        ##todo: add priority by starting position on parent axe
        ##todo: store axe_priority - as partial_order?
        for o in axe_priority[::-1]:
            slist = self.segment[o]
            segment_axe[slist] = o
        self.segment_axe = segment_axe
        
        if parent is None:
            self.parent = self.segment_axe[self.parent_segment]
        else:
            self.parent = _np.asarray(parent)
            
        self.set_id(ids=ids)
        
    def set_id(self, ids='auto'):
        """ Set axe ids
        
        If `ids='auto'`, generate automatic ids **if none exist**
        If `ids=None`, don't set ids and remove existing ones
        Else, set axe.id to given `ids`
        """
        if ids is None:   self.pop('id')
        elif ids=='auto': self.setdefault('id',_np.arange(self.number()))
        else:             self.id = ids
        
        return self.get('id')
        
    def _update_version(self, verbose=False):
        """ update AxeList from older version """
        # previous auto compute of parent segment (property sparent)
        if not hasattr(self,'parent_segment'):
            if verbose: print 'sparent property to parent_segment attribute'
            sparent = self._segment_parent[self.first_segment()]
            self.parent_segment = sparent
            del self._segment_parent
            
        # replace property "parent"
        if not hasattr(self,'parent'):
            if verbose: print 'parent property to parent attribute'
            self.parent = self.segment_axe[self.parent_segment]
            del self.segment_axe
            
        # replace 'order' attribute by property
        if self.__dict__.has_key('order'):
            if verbose: print 'order attribute to order accessor'
            self._order = self.__dict__['order']
            del self.__dict__['order']
              
    def number(self):
        """ number of axes, including dummy (i.e. 0) axe """  
        return len(self.segment)
        
    def segment_number(self):
        """ Array of the number of segments in each axe """
        if not hasattr(self,'_segment_number'):
            self._segment_number = _np.vectorize(len)(self.segment)
            self.temporary_attribute.add('_segment_number')
        return self._segment_number
        
    def length(self):
        """ length of the axes (sum of their segments length) """
        if not hasattr(self,'_length'):
            axlen = _np.vectorize(lambda slist: self._segment_list.length()[slist].sum())
            self._length = axlen(self.segment)
            self.temporary_attribute.add('_length')
        return self._length
        
    def set_length(self, length):
        """ set the axes length (replace auto compute one) """
        self.clear_temporary_attribute('_length')
        self._length = _np.asarray(length)
        
    def arc_length(self):
        """ 
        Arc length of segments for each axe, as a dict of (segment_id,arc_length)
        
        To get the arc length along axe 'aid' as a list, ordered by segment, do:
            `[t.arc_length[aid][sid] for sid in t.axe.segment[aid]]`
        """
        if not hasattr(self,'_arc_length'):
            arclen = lambda slist: dict(zip(slist,_np.cumsum(self._segment_list.length()[slist])))
            self._arc_length = _np.array(map(arclen,self.segment))
            self.temporary_attribute.add('_arc_length')
        return self._arc_length
        
    def position_on_parent(self):
        """ Arc length of parent segment """
        if not hasattr(self,'_pos_on_parent'):
            arclen = self.arc_length()
            pos = [0 if p==0 or sp==0 else arclen[p][sp] for p, sp in zip(self.parent,self.parent_segment)]
            self._pos_on_parent = _np.array(pos)
            self.temporary_attribute.add('_pos_on_parent')
        return self._pos_on_parent
        
    def insertion_angle(self):
        """ insertion angle axe """
        if not hasattr(self,'_insertion_angle'):
            dir_diff = self._segment_list.direction_difference()
            insertion_angle = dir_diff[self.first_segment(),self.parent_segment]
            self._insertion_angle = insertion_angle
            self.temporary_attribute.add('_insertion_angle')
        return self._insertion_angle
    
    def first_segment(self):
        """ first segment of axe """
        if not hasattr(self,'_segment1'):
            segment1 = _np.array([sl[0] if len(sl) else 0 for sl in self.segment])
            self._segment1 = segment1
            self.temporary_attribute.add('_segment1')
        return self._segment1
    
    def order(self):
        """ axe topological order """
        if not hasattr(self,'_order'):
            raise NotImplementedError("order property")
        return self._order

    def partial_order(self):
        """ 
        Return the axe ids in partial order
          - children axes appears after their parent
          - sibling axes are sorted by their position on parent
          - dummy axe (0) is not contained in partial order
        """
        if not hasattr(self,'_partial_order'):
            priority = zip(self.order(),self.position_on_parent())
            porder = sorted(range(len(priority)),key=priority.__getitem__)
            self._partial_order = _np.array(porder)[1:]
            self.temporary_attribute.add('_partial_order')
        return self._partial_order
        
    def segment_main_axe(self):
        """ 
        Return the "main" axe of all segments
        
        The main axe of a segment is the first (w.r.t to axe partial_order) 
        of the axe the segment is part of
        """
        if not hasattr(self,'_segment_axe'):
            segment_axe = _np.zeros(self._segment_list.number(),dtype=int)
            for axe in self.partial_order()[::-1]:
                segment_axe[self.segment[axe]] = axe
            self._segment_axe = segment_axe
            self.temporary_attribute.add('_segment_axe')
        return self._segment_axe
        
    def get_node_list(self):
        """
        Return list of axes as a list of node
        and a list of potential invalid axe: 1-segment axes with no parent_segment
        """
        from scipy.sparse import csr_matrix as csr
        from scipy.sparse.csgraph import depth_first_order as dfo
        
        axe_node = []
        invalid  = []
        term_node = self._segment_list.node_list.terminal()
        
        for i,seg_list in enumerate(self.segment):
            if len(seg_list)==0: 
                axe_node.append([])
                continue
                
            seg_node = self._segment_list.node[seg_list]
            spnode   = self._segment_list.node[self.parent_segment[i]]
            if seg_node.shape[0]==1:
                snode0   = set(seg_node[0])
                nparent  = snode0.intersection(spnode)
                if len(nparent)!=1:
                    invalid.append(i)
                    seg_node = seg_node.ravel()
                    axe_node.append(seg_node[term_node[seg_node].argsort()])
                else:
                    axe_node.append(_np.array(list(nparent) + list(snode0.difference(nparent))))
            else:
                c = csr((_np.ones(2*seg_node.shape[0],dtype='uint8'),_np.hstack((seg_node[::-1].T,seg_node[:,::-1].T))))
                s = set(seg_node[0]).difference(seg_node[1]).pop()
                order = dfo(c,s, return_predecessors=False) #nparent.pop()
                axe_node.append(order)
                
        return axe_node,invalid


def parent_segment(axes, segment_parent):
    """
    Return the parent segment of each axe in `axes`
    
    :Inputs:
      - `axes`: list of **sorted** segment in each axes
      - `segment_parent`: id of parent segment of all segments
    
    :Outputs:
      - parent segment of all axes, as a numpy array
    
    """
    p = [segment_parent[slist[0]] if len(slist) else 0 for slist in axes]
    return _np.array(p)
    
def parent_axe(axes, parent_segment):
    """ 
    Find parent axes and segments of all axes based on parent segment
    
    :Inputs:
      - `axes`: list of segment in each axes
      - `parent_segment`: id of parent segment of all axes
    
    :Outputs:
      - parent axe of all axes, as a numpy array
    
    :Note:
        Each segment is associated to the first axe it appears in.
        Thus, it works properly if each segment is part of only one axe. 
        Otherwise, the order of `axes` has a strong influence.
    """
    # find axe of each segment 
    segment_axe = {0:0}
    for axe, slist in enumerate(axes[::-1]):
        axe = len(axes)-axe-1
        segment_axe.update((sid,axe) for sid in slist)
  
    # find parent axe
    axe_parent = [segment_axe[pseg] for pseg in parent_segment]
    return _np.array(axe_parent)
    
    
