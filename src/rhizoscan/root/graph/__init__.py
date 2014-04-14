import numpy                        as _np
from   scipy   import ndimage       as _nd

from rhizoscan         import geometry      as _geo
from rhizoscan.ndarray import reshape       as _reshape
from rhizoscan.image.measurements   import color_label as _color_label
from rhizoscan.datastructure        import Data    as _Data
from rhizoscan.datastructure        import Mapping as _Mapping

##from rhizoscan.ndarray.graph import ArrayGraph as _ArrayGraph # used by SegmentGraph

from rhizoscan.tool import _property    
from rhizoscan.workflow import node as _node # to declare workflow nodes

"""
##TODO:
  - Node/Segment/AxeList: (required) set_node/segment/axe_list setter?
  - RootGraph also, making sure node.segment, segment.node, axe.segment, 
                    segment.axe link well
"""

from rhizoscan.root.graph.nsa import NodeList, SegmentList, AxeList


class RootGraph(_Mapping):
    """
    A graph representation of roots system
    
    Basically a pair of NodeList and SegmentList

    ***** in development *****
    """
    def __init__(self, node=None,segment=None):
        if node    is not None: self.node    = node
        if segment is not None: self.segment = segment
        
            
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
    

    def plot(self, bg='k', ac=None, sc=None, max_shift=0, transform=None, cmap=None, corder=1, cstart=1, **kargs):
        """
        Plot tree on top of `bg`
        
        If `sc` is not None, call RootGraph.plot with it and all other arguments
        
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
        axe_node = self.axe.get_node_list()[0]
        
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
        
    def to_mtg(self, verbose=False):
        """ create a mtg from this axial tree """
        from openalea.mtg import MTG
        
        # initialization
        # ==============
        # node sequence of each axe, node coordinates, ...
        node_list = self.axe.get_node_list()[0]
        node_pos  = [(xi,yi,0) for xi,yi in self.node.position.T]
        max_order = self.axe.order.max()
        
        # compute seed position
        seed_mask  = self.segment.seed>0
        seed_label = self.segment.seed[seed_mask]
        seed_id    = _np.unique(seed_label)
        
        seed_pos   = self.node.position[:,self.segment.node[seed_mask]]
        seed_pos   = seed_pos.reshape(seed_pos.shape[0],-1)
        seed_label = (seed_label[:,None]*_np.ones((1,2),dtype=int)).ravel()
        x = _nd.mean(seed_pos[0], labels=seed_label, index=seed_id)
        y = _nd.mean(seed_pos[1], labels=seed_label, index=seed_id)
        seed_pos = _np.vstack((x,y,_np.zeros(x.size))).T
        seed_pos = dict(zip(seed_id,seed_pos))

        # create mtg tree
        # ===============
        #   parse axe (adding nodes) in min-order, then max-length order
        #   1st time a node is parsed, its current axe is set as its main
        #   hypothesis: 1st axe node (branching node) "main axe" is the parent axe
        g = MTG()
        
        mtg_pid = {}  # plant id in mtg - keys are self plant id  
        mtg_nid = {}  # node  id in mtg - keys are self nodes id -> set the 1st processed
        
        
        # add plant
        # ---------
        for pid in seed_id:
            properties = dict(plant=pid)
            
            # add the plant: scale 1
            mtg_id = g.add_component(g.root, plant=pid, label='P%d'%pid)
            ##mtg_pid[pid] = mtg_id
            
            # add an axe (scale 2) 
            v = g.add_component(mtg_id, label='G')
            
            # add a node in its center (scale 3)
            n = g.add_component(v, position=seed_pos[pid], label='H', **properties)
            mtg_pid[pid] = n

            if verbose: 
                print 'plant added: %d->%d (mtg node:%d)' % (pid, mtg_id, mtg_pid[pid])
            
        # To select the parent axe: 
        #   - axe parsing follows axe asc. order, then length desc. order
        #   - the 1st time a node is added, the current axe is set as its 'main' 
        #     (stored mtg_nid)
        #   - when a new axe is processed, its first node is set as its parent:
        #      - if its first node has already been parsed: it is selected as
        #        its parent node, which induces its parent axe
        #      - otherwise, choose the seed node as its parent node
        for order in range(1,max_order+1):
            axe_mask  = self.axe.order==order
            axe_order = _np.argsort(self.axe.length[axe_mask])[::-1]
            
            # add axe
            # -------
            for aid in axe_mask.nonzero()[0][axe_order]:
                # find parent node id in mtg
                nlist  = node_list[aid]
                if len(nlist)==0: continue
                
                node_0 = nlist[0]
                properties = dict(axe_order=order, plant=pid, axe_id=aid, radius=1)
                
                if not mtg_nid.has_key(node_0):  #order==1 and axes connected to seed 
                    parent_node = mtg_pid[self.axe.plant[aid]]
                    parent_node,cur_axe = g.add_child_and_complex(parent_node, 
                        position=node_pos[nlist[0]], 
                        edge_type='+',
                        **properties)
                    g.node(parent_node).label='S'
                    g.node(cur_axe).label='A'
                    mtg_nid.setdefault(nlist[0],parent_node)
                    if verbose: 
                        print 'seed axe added: %d->%d (complex %d, 1st node %d->%d)' % (aid, cur_axe, g.complex(cur_axe), node_0, parent_node)
                else:
                    parent_node = mtg_nid[node_0]
                    # add 1st and current axe
                    position=node_pos[nlist[1]]
                    parent_node,cur_axe = g.add_child_and_complex(parent_node, 
                            position=position,
                            edge_type='+', **properties)
                    g.node(parent_node).label='S'
                    g.node(cur_axe).label='A'
                    mtg_nid.setdefault(nlist[1],parent_node)
                    nlist = nlist[1:]
                    if verbose: 
                        print 'axe added: %d->%d' % (aid, cur_axe)
                        _p = g.parent(parent_node)
                        if _p is None: print '**** parent None %d ****' % parent_node
                    
                # add nodes
                # ---------
                properties.pop('axe_order')
                for node in nlist[1:]:
                    position=node_pos[node]
                    parent_node = g.add_child(parent_node, 
                           position=position, 
                           x=position[0], 
                           y=-position[2], 
                           edge_type='<', 
                           label='S', 
                           **properties)
                    mtg_nid.setdefault(node,parent_node)
                    _p = g.parent(parent_node)
                    if _p is None: print '**** parent None %d ****' % parent_node
                if verbose>1: 
                    print '    node:', nlist
            
        # add edge_type to axe vertex: '+' for all axe but 1st that have no parent
        ## second (or more) order axes which starts at seeds are not topologically second order !!!
        edge_type = g.property('edge_type')
        for v in g.vertices(scale=2):
            if g.parent(v) is not None:
                #g[v]['edge_type'] = '+' # don't work
                edge_type[v] = '+'

        return g
        
    # printing    
    # --------
    def __str__ (self): 
        return self.__class__.__name__ + ': %d nodes, %d segments, %d axes' % (self.node.number, self.segment.number, self.axe.number)
    def __repr__(self): 
        return self.__str__()  ## for ipython of RootGraph obj...


## tests
def test_to_mtg(t):
    g = t.to_mtg()
    
    # test if plant property is correct
    # ---------------------------------
    # split mtg by plants
    g = map(g.sub_mtg, g.vertices(scale=1))
    
    pl_axe = [list(set(gi.property('axe').values())) for gi in g]
    pl_axe = [[a for a in alist if a>0] for alist in pl_axe]      ## remove negativ axe...
    
    ac = np.zeros(ref.axe.number,dtype=int)
    for i,al in enumerate(pl_axe): ac[al] = i+1
    
    assert (t.axe.plant==ac).all()
