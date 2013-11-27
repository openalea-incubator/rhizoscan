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
from rhizoscan.datastructure import Data, Mapping, Sequence

def parse_neuronJ_file(filename):
    f = NJ_iterator(filename)
    T = []
    while not f.is_eof():
        T.append(NJ_Tracing(f))

    return T
    
class NJ_loader(Mapping):
    def __init__(self,filename):
        self.lines = file(filename,'r').readlines()
        self.i = 0
        
        self.tracing = []
        while not self.is_eof():
            block = self.next_block()
            if block=='Tracing':
                ##print 'add tracing'
                t = NJ_Tracing(self)
                self.tracing.append(t)
            elif block=='Segment':
                ##print '   add segment'
                s = self.read_segment()
                t.add_segment(s)
        
        for t in self.tracing:
            t.concatenate()
        
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
        
    def to_tree(self, scale=1):
        from rhizoscan.root.graph import SegmentList, NodeList, RootGraph, RootAxialTree
        from rhizoscan.root.graph import _UNSET, _SEED, _UNREACHABLE
    
        # find axes order
        t_type  = [t.type for t in self.tracing]
        max_order = max(t_type)-2
        min_order = min(t_type)-2
        for t in self.tracing:
            t.order = t.type-2
            if t.order<1:
                print 'Tracing #%d has invalid order: %d. Replaced by %d' % (t.id, t.order,max_order)
                t.order = max_order
        
        # find cluster ids and make suitable "seed" nodes (and bg)
        cplant = set([t.cluster for t in self.tracing])
        cplant.add(0)
        cplant = np.array(list(cplant))
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
        
        # construct tree graph
        n = NodeList(position=node*scale)
        s = SegmentList(node_id=segment, node_list=n)
        n.set_segment(s)
        s.seed = sseed
        t = RootAxialTree(node=n, segment=s)
        t.set_axes(s_axe=saxe,s_parent=sparent)
        
        return n, segment, saxe, sparent, t
        
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
        
        
# Project stuff
# -------------
def load_ref_trees(p):
    """ load all the ndf file related to sequences input image files """
    from os.path import splitext, split, join, sep
    for s in p.sequences:
        
        out_fname = s.make_output_filename(names='tree', seq_length=len(s.input))
        tree = Sequence(output=out_fname + '.tree')
        ndf_files = [None]*len(s.input)
        
        for i,imfile in enumerate(s.input.get_file()):
            ndf_files[i] = splitext(imfile)[0] + '.ndf'
            print '\033[30m' + join(*imfile.split(sep)[-2:]), '\033[33m'
            try:
                nt = NJ_loader(ndf_files[i])
                tree[i] = nt.to_tree()[-1]
            except Exception as e:
                print '\033[31m  error:(%s)\033[30m' % str(e)
            #else:
            #    print 'loaded'
        
        s.tree = tree
        s.ndf_files = ndf_files
        s.dump()
        
# ------ pipeline stuff ------ 
# ----------------------------
def make_ref_dataset(ini_file, output='tree', overwrite=False, verbose=1):
    from os.path import splitext, join, exists
    from .pipeline.dataset import make_dataset 
    
    rds, invalid, out_dir = make_dataset(ini_file=ini_file, output=output)
    
    if verbose>1:
        print '\033[31m ---- invalid files: ----'
        for i in invalid: print ':'.join(i[:2]), '\n    '+ i[2]
        print ' ------------------------\033[30m'
     
    for i,ref in enumerate(rds):
        if not overwrite and ref.get_storage_entry().exists() and ref.load().has_key('tree'): continue
        
        ref.ref_file = splitext(ref.filename)[0]+'.ndf'
        if verbose:
            print 'converting file', ref.ref_file
            meta = ref.metadata
            if hasattr(meta,'scale'): scale = meta.scale
            else:                     scale = 1 
            tree = NJ_loader(ref.ref_file).to_tree(scale)[-1]
            tree.metadata = meta
            ref.set('tree',tree,store=True)
            ref.__loader_attributes__ = ['metadata','filename']
            rds[i] = ref.dump()
        
    return rds, invalid, out_dir
    
def load_db_with_ref(auto, ref, auto_out='tree', ref_out='tree'):
    """
    return 
        pipeline.dataset.make_dataset(...)[0] for files with ref data only
        parse_ref_data[0]
    """
    from .pipeline.dataset import make_dataset 
    
    def to_key(x):
        return x.multilines_str(max_width=2**60)#repr(to_tuple(x))
    def to_tuple(x):
        if hasattr(x,'__dict__'):
            return tuple((k,to_key(v)) for k,v in x.__dict__.iteritems())
        else:
            return x

    auto, inv, auto_dir = make_dataset(auto,output=auto_out)
    ref,  inv, ref_dir  = make_ref_dataset(ref,output=ref_out, verbose=False)
    auto = dict([(to_key(a.metadata),a) for a in auto])
    auto = [auto[to_key(r.metadata)] for r in ref]
    
    return auto, ref, auto_dir, ref_dir

def recompute_tree(src, src_dir, dst_dir, to_tree, to_axe):
    from os.path import join
    from .pipeline import compute_tree  
    
    dst = [None]*len(src)
    for i,s in enumerate(src):
        d = s.__copy__()
        d.output = join(dst_dir, s.output[len(src_dir)+1:])
        
        print 'processing:', d.output+'.tree'
        t = Data.load(s.output+'.tree')
        compute_tree(t, to_tree=to_tree, to_axe=to_axe, metadata=s.metadata, output_file=d.output+'.tree', verbose=False)
        dst[i] = d
        
    return dst

def load_tree_db(auto, ref, output='tree'):
    """
    return 
        original image sequence
        automatically computed tree sequence
        reference tree sequence
    """
    from rhizoscan.image         import ImageSequence
    from rhizoscan.datastructure import Sequence
    
    auto, ref = load_db_with_ref(auto=auto, ref=ref, output=output)[:2]
    
    img_seq  = ImageSequence(files=[a.filename for a in auto])
    ref_seq  = Sequence(files=[r.output+'.tree' for r in ref])
    auto_seq = Sequence(files=[a.output+'.tree' for a in auto])
    
    return img_seq, auto_seq, ref_seq

def make_TreeCompare(auto, ref, auto_out, ref_out='tree', name=None):
    from rhizoscan.image import ImageSequence
    from .measurements import TreeCompare
    
    if name is None:
        from os.path import dirname, join
        name = join(dirname(ref),'vs_' + auto_out)
        
    auto,ref = load_db_with_ref(auto=auto, ref=ref, auto_out=auto_out, ref_out=ref_out)[:2]
    
    img  = ImageSequence([a.filename       for a in auto])
    auto =      Sequence([a.output+'.tree' for a in auto])
    ref  =      Sequence([r.output+'.tree' for r in ref])
    
    return TreeCompare(auto=auto,ref=ref,filename=name) #,image=img
