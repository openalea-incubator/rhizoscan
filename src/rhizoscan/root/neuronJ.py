"""
Import of neuronJ *.ndf file
such as understood the 20th dec. 2012

// title
version: 1.4.2
// parmeters
- 8 values
// Type....
...
// Cluster names
Default
Cluster 01
... to 10
// Tracing N*                      - iteratif
id
type: 3=primary 4=secondary
cluster: id of the plants
label: a name (id of the secondary axe ?
// Segment # of the Tracing N*     - iteratif
y0
x0
y1
x1
...
yn
xn
// segment ...
y0 = yn of prev segment
x0 = xn -------"-------
...
"""
# ------ neuronJ loader ------ 
# ----------------------------
import numpy as np
from rhizoscan.datastructure import Data, Mapping ##, Sequence
from rhizoscan.geometry.polygon import distance_to_segment

def parse_neuronJ_file(filename):
    f = NJ_iterator(filename)
    T = []
    while not f.is_eof():
        T.append(NJ_Tracing(f))

    return T
    
def _group_by(elements, key):
    groups = {}
    for elt in elements:
        key_value = elt.get(key) 
        groups.setdefault(key_value,[]).append(elt)
    return groups
        
class NJ_loader(Mapping):
    def __init__(self,filename, tree=True):
        """
        Load neuronJ file `filename`
        concatenate segments of traces
        
        if `tree`: call to_tree (see to_tree doc)
        """
        self.lines = file(filename,'r').readlines()
        self.i = 0
        
        self.tracing = []
        while not self.is_eof():
            block = self.next_block()
            if block=='Tracing':
                t = NJ_Tracing(self)
                self.tracing.append(t)
            elif block=='Segment':
                s = self.read_segment()
                t.add_segment(s)
        
        for t in self.tracing:
            t.concatenate()
            
        if tree:
            self.to_tree()
        
    def is_eof(self):
        return self.i>=len(self.lines)
        
    def read_line(self):
        """ return current line as string and go to next. return None if EOF """
        if self.is_eof():
            return None
        else:
            self.i+=1
            return self.lines[self.i-1][:-1]  # :-1 to remove '\n'
        
    def read_int(self):
        """ return the next line (if it is) an integer or None otherwise """
        i = self.i
        try:
            return int(self.read_line())
        except:
            self.i = i
            return None
            
    def read_segment(self):
        pos = []
        i = self.read_int()
        while i is not None:
            pos.append((i,self.read_int()))
            i = self.read_int()
        return pos 
        
    def next_block(self):
        l = '..'
        while l[:2]!='//':
            l = self.read_line()
        return l[2:].split()[0]
        
    def set_tree_attribute(self):
        """
        Compute 'order' and 'plant' attribute for each tracing
        
        Expected tracing content:
          - plant id of each tracing should be set in its 'cluster' attribute
            and be >= 1
          - order of each tracing should be set in its 'type' attribute, 
            such that:   `order = tracing.type-2`.   
            i.e.:
             - "primary"   is type 3 in neuronJ => giving order 1
             - "secondary" is type 4 in neuronJ => giving order 2
        
        This is called by `to_tree` but can be called previously for testing::
        
          nj = NJ_loader(some_file, to_tree=False)
          err = nj.set_tree_attribute()
          if len(err)==0:
              nj.to_tree()
          else:
              print '\n'.join(err)
        """
        # find plant id of axes: tracing.cluster
        error = []
        
        err = []
        last_plant = 0
        for t in self.tracing:
            t.plant = t.get('cluster')
            if t.plant is None or t.plant<1:
                err.append((t.id, last_plant))                                    
                t.plant = last_plant
            else:
                last_plant = t.plant
        if len(err):
            error.extend(['Plant id (i.e. cluster) of tracing id %d auto-set to %d' % e for e in err])
            
        # find axe order: tracing.type-2
        err = []
        max_order = max([t.type-2 for t in self.tracing])
        for t in self.tracing:
            t.order = t.get('type')-2
            if t.order is None or t.order<1:
                err.append(t.id)
                t.order = max_order
        if len(err):
            error.extend(['Axe order (i.e. cluster) of tracing id %d auto-set to %d' % (e,max_order) for e in err])
            
        plant_ids = sorted(set([t.plant for t in self.tracing]))
        orders    = sorted(set([t.order for t in self.tracing]))
        
        return plant_ids, orders, error

    def to_tree(self, scale=1):
        """   
        final process of constructor: allow call to `as_axial_tree()` 
        
        find parent of all axes of order>1
        add a connection node to these parents
        
        Expected object content:
          - axes are the tracings
          - plant id of each axe should be set in its 'cluster' attribute
          - order of each axe should be set in its 'type' attribute, such that:
            `order = tracing.type-2`.   i.e.:
             - "primary"   is type 3 => giving order 1
             - "secondary" is type 4 => giving order 2
        """
        plant_ids, orders, error = self.set_tree_attribute()
        max_order = max(orders)
        if len(error):
            print '\n'.join(error)
            
        # find axe parent, and connect axe nodes
        # --------------------------------------
        def node_insert(node,index,sub_index,in_node):
            """
            Insert `in_node` in `node` at position `index`
            when sorting `index`, `sub_index` is use to sort those that have same values
            """
            #print order, index[order], node.shape, in_node.shape
            order = sorted(range(len(index)), key=zip(index,sub_index).__getitem__) # argsort
            index = index[order]
            in_node  = in_node[order]
            inserted = index+np.arange(1,index.size+1)
            
            new_shape = (node.shape[0]+inserted.size, node.shape[1])
            old_mask = np.ones(new_shape[0],dtype=bool)
            old_mask[inserted] = 0
            new_node = np.empty(new_shape)
            new_node[old_mask] = node
            new_node[inserted] = in_node
            
            ## tried np.insert, but did not manage to make it work
            #new_node = np.insert(node,index[order]+1,in_node, axis=0)
            #inserted = (index[order]+np.arange(1,index.size+1))[np.argsort(order)]
            
            return new_node, inserted[np.argsort(order)]

        
        # for all axe of order i (>1), find closest node in order i-1
        # set suitable axe parent (its id), and add node at start of axe
        pgroup = _group_by(self.tracing, 'plant')
        for tracing in pgroup.values():
            parent_order = 1
            # set parent to None for all axe of order<=1
            #for t in tracing:
            #    if t.order<=1:
            #        t.parent_id=None
            for o in range(2,max_order+1):
                p_trace = [t for t in tracing if t.order==o-1]
                c_trace = [t for t in tracing if t.order==o]
                
                if len(c_trace)==0: break

                # first node of all child axe
                c_nodes = np.vstack([t.node[0] for t in c_trace]).T
                
                # detect closest parent segment
                p_segments = [np.concatenate((t.node[:-1,:].T[:,:,None],t.node[1:,:].T[:,:,None]),axis=-1) for t in p_trace] # [(xy,[s=n-1],v12)]
                p_segments = np.hstack(p_segments)
                pseg_ax_id = np.hstack([[i]*(t.node.shape[0]-1) for i,t in enumerate(p_trace)])
                
                d, c_proj, seg_dist = distance_to_segment(c_nodes,p_segments)
                pseg_best = d.argmin(axis=1)                         # closest parent segment 
                
                p_axe_id  = pseg_ax_id[pseg_best]                     # axe of closest segment 
                c_proj    = c_proj[:,np.arange(d.shape[0]),pseg_best] # child on (best) parent
                seg_dist  = seg_dist[np.arange(d.shape[0]),pseg_best] # d(node-proj, seg 1 node)
                
                # add child axe node projection to parent axes
                for aid in np.unique(p_axe_id):
                    # for each parent axe, add (sorted) child node projection
                    # and add parent-id & parent-node-id to children 
                    c_id   = (p_axe_id==aid).nonzero()[0]
                    p_node = p_trace[aid].node
                    p_seg  = pseg_best[c_id]
                    s_dist = seg_dist[c_id]
                    #print p_trace[aid].id, [c_trace[i].id for i in c_id], s_dist
                    new_n  = c_proj[:,c_id].T
                    new_node, new_ind = node_insert(p_node,p_seg, s_dist, new_n)
                    p_trace[aid].node = new_node
                    for c,ind in zip(c_id,new_ind):
                        #c_trace[c].node = np.vstack((c_proj[:,c].T,c_trace[c].node))
                        #c_trace[c].node = np.vstack((p_trace[aid].node[ind],c_trace[c].node))
                        c_trace[c].parent_id   = p_trace[aid].id
                        c_trace[c].parent_node = ind
    
    def make_axial_tree(self,scale=1):
        """
        Construct a RootAxialTree from this object
        """
        from rhizoscan.root.graph import NodeList, SegmentList, AxeList, RootAxialTree, RootGraph
        
        def first_id(list_length):
            ids = np.cumsum(list_length)+1   # +1: dummy elements
            ids[1:] = ids[:-1]
            ids[0]  = 1
            return ids
            
        
        # construct NodeList
        # ------------------
        node_pos  = np.hstack([[[0],[0]]]+[t.node.T for t in self.tracing]) # [[0],[0]]: dummy node
        node = NodeList(position=node_pos*scale)
        
        # Convert trace.node to list of node id pairs in node_pos
        # -------------------------------------------------------
        ax_node_num   = np.array([t.node.shape[0] for t in self.tracing])
        ax_node_start  = first_id(ax_node_num)
        
        trace_by_id = dict([(t.id,i) for i,t in enumerate(self.tracing)])
        
        ax_segments = [] # list of segments for all axes
        for n,s,t in zip(ax_node_num,ax_node_start,self.tracing):
            if hasattr(t,'parent_id'):
                seg = np.arange(-1,2*n-1).reshape(n,2)/2 +s
                p_pos = trace_by_id[t.parent_id]
                seg[0,0] = ax_node_start[p_pos]+t.parent_node
            else:
                seg = np.arange( 1,2*n-1).reshape(n-1,2)/2   +s
            ax_segments.append(seg)
        
        # construct SegmentList
        # ---------------------
        snodes = np.vstack([np.array([[0,0]])]+ax_segments)       # [[0,0]]: dummy segment
        segment = SegmentList(node_id=snodes, node_list=node)
        
          # label segments with their axe id
        saxe   = np.hstack([0]+[[i]*len(slist) for i,slist in enumerate(ax_segments)])
        segment.add_property('axe',saxe)
        
          # find and label segments with their parent segment id
          #   for most segment, parent is seg id-1
          #   parent of dummy segment (0) is it-self
          #   parent of 1st seg of child axes = id of 1st seg of parent axe + "n"
        axe_1st_segment = first_id([slist.shape[0] for slist in ax_segments])
        sparent = np.arange(-1,segment.number-1)
        sparent[0] = 0
        for i,t in enumerate(self.tracing):
            c_1st_seg = axe_1st_segment[i]
            if hasattr(t,'parent_id'):
                p_1st_seg = axe_1st_segment[trace_by_id[t.parent_id]]
                #print i, t.id, t.parent_id, t.parent_node, c_1st_seg, p_1st_seg
                sparent[c_1st_seg] = p_1st_seg + t.parent_node-1
            else:
                sparent[c_1st_seg] = 0
        segment.add_property('parent',sparent)
        
        # construct AxeList
        # -----------------
        ax_seg_num = [s.shape[0] for s in ax_segments]
        ax_seg_start  = first_id(ax_seg_num)
        axe_segment = [[]]+[range(s,s+n) for n,s in zip(ax_seg_num,ax_seg_start)]
        axe_order = np.array([0] + [t.order for t in self.tracing])
        axe_plant = np.array([0] + [t.plant for t in self.tracing])
        axe = AxeList(axes=axe_segment,order=axe_order,plant=axe_plant, segment_list=segment)
        axe.add_property('seg_1st',np.hstack(([0],axe_1st_segment)))
        
        return RootAxialTree(node=node, segment=segment, axe=axe)

    def to_tree_0(self, scale=1):
        """
        ###OLD STUFF: not used anymore
        replaced by `to_tree` (called by constructor) and `make_axial_tree`
        """
        from rhizoscan.root.graph import SegmentList, NodeList, RootGraph, RootAxialTree
        ##from rhizoscan.root.graph import _UNSET, _SEED, _UNREACHABLE
        ##  should not do it like that anymore...
        _UNREACHABLE = -2  ##                 
        _SEED        = -1  ##
        
        # find axes order
        t_type  = [t.type for t in self.tracing]
        max_order = max(t_type)-2
        for t in self.tracing:
            t.order = t.type-2
            if t.order<1:                
                print 'Tracing #%d has invalid order: %d. Replaced by %d' % (t.id, t.order,max_order)
                t.order = max_order
        
        # find cluster ids and make suitable "seed" nodes (and bg)
        cplant = set([t.cluster for t in self.tracing])
        cplant.add(0)
        cplant = np.array(sorted(cplant))
        corder = np.ones(cplant.size)*_SEED
        corder[0] = _UNREACHABLE
        cnode  = np.zeros((cplant.size,2))
        for p,n in zip(cplant[1:],cnode[1:]):
            n[:] = [t for t in self.tracing if t.order==1 and t.cluster==p][0]['node'][0]
        
        # nodes position, id of the plant, order of their axe, 
        #   id of their axe and index of the 1st node per axe
        node   = np.concatenate([cnode] +[t.node for t in self.tracing]).T
        nplant = np.concatenate([cplant]+[[t.cluster]*len(t.node) for t in self.tracing])
        norder = np.concatenate([corder]+[[t.type-2] *len(t.node) for t in self.tracing])
        astart = np.cumsum([0]+[len(t.segment)+1 for t in self.tracing])[:-1]+cplant.size
        
        # make segment list
        #   > add bg segment, and one default branching segment to each axe
        segment = [[[[-1,st]],t.segment+st] for t,st in zip(self.tracing,astart)]
        segment = np.concatenate([[[0,0]]]+[sij for si in segment for sij in si])
        #   > fix branching segments
        for sid in np.any(segment==-1,axis=1).nonzero()[0]:
            sn0 = segment[sid,1]
            # find best branching node of parent axe (ie. order-1 & same plant)
            ##  best=closest node > to be improved
            p_order = norder[sn0]-1
            if p_order<1: p_order = _SEED 
            n = ((norder==p_order)&(nplant==nplant[sn0])).nonzero()[0]
            d = np.sum((node[:,n] - node[:,[sn0]])**2,axis=0)**.5
            segment[sid,0] = n[np.argmin(d)]
        
        # compute tree related data
        sparent = segment[:,0]-cplant.size+1  # +1: bg segment
        sparent[sparent<0] = 0

        nseed = nplant*(norder==_SEED)
        sseed = nseed[segment[:,0]]

        naxe   = np.zeros(node.shape[1],dtype=int)
        naxe[astart] = 1
        naxe = np.cumsum(naxe)
        naxe[naxe==0] = _SEED
        naxe[0] = _UNREACHABLE
        saxe = naxe[segment[:,1]]
        saxe[sseed>0] = _SEED
        saxe[0] = _UNREACHABLE
        
        # construct node ans segment list
        n = NodeList(position=node*scale)
        s = SegmentList(node_id=segment, node_list=n)
        n.set_segment(s)
        s.seed = sseed
        
        # construct axe list
        a_seg = [[] for i in xrange(saxe.max()+1)]
        for sid,aid in enumerate(saxe):
            if aid>=0: a_seg[aid].append(sid)
        
        return n,s,saxe,sparent,norder
        t = RootAxialTree(node=n, segment=s)         
        
        ##t.set_axes(s_axe=saxe,s_parent=sparent)
        ##return n, segment, saxe, sparent, t
        
    def plot(self,bg=None, scale=1):
        from matplotlib import pyplot as plt
        if bg is not None:
            plt.clf()
            plt.imshow(bg)
        
        for t in self.tracing:
            plt.plot(t.node[:,0]*scale,t.node[:,1]*scale,'.')
            plt.plot(t.node[:,0]*scale,t.node[:,1]*scale)
                        
class NJ_Tracing(Mapping):
    def __init__(self,nj):
        self.id      = nj.read_int()
        self.type    = nj.read_int()  # type is order+2
        self.cluster = nj.read_int()
        self.label   = nj.read_line()
        self.segment = []
        
    def add_segment(self,positions):
        self.segment.append(positions)
    def concatenate(self):
        s = enumerate(self.segment)
        self.node = np.concatenate([si if i==0 else si[1:] for i,si in s])
        self.segment = np.arange(self.node.shape[0]-1)[:,None]+np.array([[0,1]])
        
        
# ------ pipeline stuff ------ 
# ----------------------------
def test_ref_dataset(ini_file):
    """
    Load all neuronJ file indicated by `ini_file` and check if they have 
    suitable attribute: cluster (i.e. plant)>0 and type (i.e. order+2)>2
    
    return 
      - the list of found file which are not processed
      - a dictionary of files with missing or erroneous attributes
    """
    from os.path import splitext
    from .pipeline.dataset import make_dataset 
    
    rds, invalid, out_dir = make_dataset(ini_file=ini_file, out_dir='__WHAT_EVER__')
    
    error = {}
    for i,ref in enumerate(rds):
        ref.ref_file = splitext(ref.filename)[0]+'.ndf'
        err = NJ_loader(ref.ref_file,tree=False).set_tree_attribute()[-1]
        if len(err):
            print '*** error in', ref.ref_file, '***'
            print '\n'.join(err)
            error[ref.ref_file] = err
            
    return invalid, error

def make_ref_dataset(ini_file, output='tree', load_ndf=True, overwrite=False, verbose=1):
    from os.path import splitext, join, exists
    from .pipeline.dataset import make_dataset 
    
    rds, invalid, out_dir = make_dataset(ini_file=ini_file, out_dir=output)
    
    if verbose>1:
        print '  ---- invalid files: ----'
        for i in invalid: print ':'.join(i[:2]), '\n    '+ i[2]
        print '  ------------------------'
     
    if not load_ndf: return rds,invalid, out_dir
    
    for i,ref in enumerate(rds):
        if ref.get_file().exists():
            ref = ref.load()
        ref.ref_file = splitext(ref.filename)[0]+'.ndf'

        if overwrite or not ref.has_key('tree'):
            if verbose: print 'converting file', ref.ref_file
            meta = ref.metadata
            if hasattr(meta,'scale'): scale = meta.scale
            else:                     scale = 1 
            tree = NJ_loader(ref.ref_file).make_axial_tree(scale)
            tree.metadata = meta
            ref.set('tree',tree,store=True)
            ref.__loader_attributes__ = ['metadata','filename']
            ref.dump()
            
        rds[i] = ref
        
    return rds, invalid, out_dir
    
## TO DELETE? Project stuff
## -------------
##def load_db_with_ref(auto, ref, auto_out='tree', ref_out='tree'):
##    """
##    return 
##        pipeline.dataset.make_dataset(...)[0] for files with ref data only
##        parse_ref_data[0]
##    """
##    from .pipeline.dataset import make_dataset 
##    
##    def to_key(x):
##        return x.multilines_str(max_width=2**60)#repr(to_tuple(x))
##    def to_tuple(x):
##        if hasattr(x,'__dict__'):
##            return tuple((k,to_key(v)) for k,v in x.__dict__.iteritems())
##        else:
##            return x
##
##    auto, inv, auto_dir = make_dataset(auto,out_dir=auto_out)
##    ref,  inv, ref_dir  = make_ref_dataset(ref,output=ref_out, verbose=False)
##    auto = dict([(to_key(a.metadata),a) for a in auto])
##    auto = [auto[to_key(r.metadata)] for r in ref]
##    
##    return auto, ref, auto_dir, ref_dir
##
## # Project stuff
##def load_ref_trees(p):
##    """ load all the ndf file related to sequences input image files """
##    from os.path import splitext, split, join, sep
##    for s in p.sequences:
##        
##        out_fname = s.make_output_filename(names='tree', seq_length=len(s.input))
##        tree = Sequence(output=out_fname + '.tree')
##        ndf_files = [None]*len(s.input)
##        
##        for i,imfile in enumerate(s.input.get_file()):
##            ndf_files[i] = splitext(imfile)[0] + '.ndf'
##            print '\033[30m' + join(*imfile.split(sep)[-2:]), '\033[33m'
##            try:
##                nt = NJ_loader(ndf_files[i])
##                tree[i] = nt.to_tree()[-1]
##            except Exception as e:
##                print '\033[31m  error:(%s)\033[30m' % str(e)
##            #else:
##            #    print 'loaded'
##        
##        s.tree = tree
##        s.ndf_files = ndf_files
##        s.dump()
##
##def recompute_tree(src, src_dir, dst_dir, to_tree, to_axe):
##    from os.path import join
##    from .pipeline import compute_tree  
##    
##    dst = [None]*len(src)
##    for i,s in enumerate(src):
##        d = s.__copy__()
##        d.output = join(dst_dir, s.output[len(src_dir)+1:])
##        
##        print 'processing:', d.output+'.tree'
##        t = Data.load(s.output+'.tree')
##        compute_tree(t, to_tree=to_tree, to_axe=to_axe, metadata=s.metadata, output_file=d.output+'.tree', verbose=False)
##        dst[i] = d
##        
##    return dst
##
##def load_tree_db(auto, ref, output='tree'):
##    """
##    return 
##        original image sequence
##        automatically computed tree sequence
##        reference tree sequence
##    """
##    from rhizoscan.image         import ImageSequence
##    from rhizoscan.datastructure import Sequence
##    
##    auto, ref = load_db_with_ref(auto=auto, ref=ref, output=output)[:2]
##    
##    img_seq  = ImageSequence(files=[a.filename for a in auto])
##    ref_seq  = Sequence(files=[r.output+'.tree' for r in ref])
##    auto_seq = Sequence(files=[a.output+'.tree' for a in auto])
##    
##    return img_seq, auto_seq, ref_seq
##
##def make_TreeCompare(auto, ref, auto_out, ref_out='tree', name=None):
##    from rhizoscan.image import ImageSequence
##    from .measurements import TreeCompare
##    
##    if name is None:
##        from os.path import dirname, join
##        name = join(dirname(ref),'vs_' + auto_out)
##        
##    auto,ref = load_db_with_ref(auto=auto, ref=ref, auto_out=auto_out, ref_out=ref_out)[:2]
##    
##    img  = ImageSequence([a.filename       for a in auto])
##    auto =      Sequence([a.output+'.tree' for a in auto])
##    ref  =      Sequence([r.output+'.tree' for r in ref])
##    
##    return TreeCompare(auto=auto,ref=ref,filename=name) #,image=img
