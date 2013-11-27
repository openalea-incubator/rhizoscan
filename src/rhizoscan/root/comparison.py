import numpy as np
import scipy.ndimage as nd

from rhizoscan.workflow import node as _node # to declare workflow nodes
from rhizoscan.ndarray  import unravel_indices as _unravel
from rhizoscan.datastructure  import Mapping as _Mapping

from rhizoscan.root.measurements import compute_tree_stat
from rhizoscan.root.measurements import statistic_name_list


class TreeCompare(_Mapping):
    def __init__(self, reference, compared, image=None, filename=None):
        """
        Allow to compare a reference tree with the compared one.
        
        :Input:
          - reference: loader object to a reference tree
          - compared:  loader object to a compared tree
          - image:     optional related image - or image filename
          - filename:  optional file to save this TreeCompare object
                       If given, automatically saves constructed object
        """
        self.ref = reference
        self.cmp = compared 
        self.img = image
        
        # set save filename
        if filename:
            self.set_storage_entry(filename)
            self.dump()
        
    def match_plants(self, max_distance=None):
        """
        Find a 1-to-1 matching between ref & cmp trees from geometric distance of the seeds.
        
        It adds the following attributes to this  object:
         - plant_map: a dictionary of (ref-plant-id:cmp-plant-id) pairs
         - plant_missed: the number of unmatched plants
        
        This method loads the ref&cmp trees - call `clear()` to unload them.
        """
        def distance_matrix(x1,y1,x2,y2):
            x1 = x1.reshape(-1,1)
            y1 = y1.reshape(-1,1)
            x2 = x2.reshape(1,-1)
            y2 = y2.reshape(1,-1)
            return ((x1-x2)**2 + (y1-y2)**2)**.5

        # match root plant w.r.t seed position 
        # ------------------------------------
        def seed_position(t):
            """
            output: plant-id, x, y
            """
            mask  = t.segment.seed>0
            nseed = t.segment.node[mask]
            lseed = t.segment.seed[mask]
            mask  = nseed.all(axis=1)  # remove bg segment
            nseed = nseed[mask]
            lseed = lseed[mask]
            
            pid = np.unique(lseed)
            x = nd.mean(t.node.x[nseed],labels=lseed.reshape(-1,1),index=pid)
            y = nd.mean(t.node.y[nseed],labels=lseed.reshape(-1,1),index=pid)
            
            return pid,x,y
            
        rpid, rx, ry = seed_position(self.ref)
        cpid, cx, cy = seed_position(self.cmp)
        
        d = distance_matrix(rx,ry,cx,cy)
        ##s1 = set(zip(range(d.shape[0]),np.argmin(d,axis=1)))
        ##s2 = set(zip(np.argmin(d,axis=0),range(d.shape[1])))
        ##match = s1.intersection(s2)
        match,r_unmatch,c_unmatch = direct_matching(d,max_d=max_distance)
        
        self.mapping = _Mapping()
        self.mapping.plant = dict((rpid[p1],cpid[p2]) for p1,p2 in match)
        self.mapping.plant_missed_ref = [rpid[i] for i in r_unmatch]
        self.mapping.plant_missed_cmp = [cpid[i] for i in c_unmatch]
        
        # match root axes w.r.t to axe first node
        # ---------------------------------------
        ##todo
        
        
    def compute_stat(self, stat_names='all', mask=None, save=True):
        """
        Compute trees statistices listed in `stat_names`
        save (dump) it-self if `save`is True.
        ## forgot what `mask` is. check root.measurements
        
        This method loads the ref&cmp trees - call `clear()` to unload them.
        """
        compute_tree_stat(self.ref, stat_names=stat_names, mask=mask)
        compute_tree_stat(self.cmp, stat_names=stat_names, mask=mask)
        if save:
            self.dump()
           
    def clear(self):
        """ replace trees by their loader """
        if not _Mapping.is_loader(self.__dict__['ref']): self.ref = self.ref.loader()
        if not _Mapping.is_loader(self.__dict__['cmp']): self.cmp = self.cmp.loader()


# statistical root tree comparison
# --------------------------------
class TreeCompareSequence(_Mapping):
    """
    :todo:
        once Sequence is finished (not required to have a file per element)
        it should become a subcalsse of Sequence
    """
    def __init__(self, reference, compared, image=None, filename=None):
        """
        :Input:
          - reference: sequence of reference tree
          - compared:  sequence of compared tree
          - image:     optional related ImageSequence
          - filename:  optional file to save this TreeCompareSequence object 
        """
        # create list of TreeCompare objects
        from itertools import izip
        if image:
            self.tc_list = [TreeCompare(r,c,i) for r,c,i in izip(reference,compared,image)]
        else:
            self.tc_list = [TreeCompare(r,c)   for r,c   in izip(reference,compared)]
            
        if filename:
            self.set_storage_entry(filename)
            self.dump()

    def match_plants(self, max_distance=None, save=True):
        """ ##consider doing the match directly through compute_stat """
        for tc in self.tc_list:
            tc.match_plants(max_distance=max_distance)
            tc.clear()
        # compute number of matched tree and unmatched ones
        self.matched_number   = sum(len(tc.mapping.plant) for tc in self.tc_list)
        self.matched_ref_miss = sum(len(tc.mapping.plant_missed_ref) for tc in self.tc_list)
        self.matched_cmp_miss = sum(len(tc.mapping.plant_missed_cmp) for tc in self.tc_list)
        #self.unmatched_number = sum(tc.plant_missed   for tc in self.tc_list)
        
        if save:
            self.dump()
        
    def compute_stat(self, stat_names='all', mask=None, save=True):
        for tc in self.tc_list: 
            tc.compute_stat(stat_names=stat_names, mask=mask, save=tc.get_storage_entry() is not None)
        if save:
            self.dump()
            
    def __store__(self):
        s = self.__copy__()
        s.tc_list = [tc.__parent_store__() for tc in s.tc_list]
        return _Mapping.__store__(s)

def plot(self, stat='axe1_length', title=None, prefilter=None, split=None, legend=True, merge_unique=False, scale=1, cla=True):
        import matplotlib.pyplot as plt
        
        title = title if title is not None else stat
        
        refs = []
        cmps = []
        meta = []
        if prefilter is None: prefilter = lambda st: st
        for tc in self.tc_list:
            sr = tc.ref.stat[stat]
            sc = tc.cmp.stat[stat]

            #tree.extend([(pid,a,r) for pid in pl_id])
            meta.extend([tc.ref.metadata]*len(tc.plant_map))
            refs.extend([prefilter(sr[pid]) for pid in tc.plant_map.keys()])
            cmps.extend([prefilter(sc[pid]) for pid in tc.plant_map.values()])
    
        cmps  = np.array(cmps)*scale
        refs  = np.array(refs) *scale
        
        bound = max(max(refs), max(cmps))
        if cla:
            plt.cla()
        plt.plot([0,bound], [0,bound], 'r')
        
        error_name = 'Average percentage error'#'normalized RMS Error'
        def error(x,y):
            """ OR NOT... normalized root mean square error """
            err = (abs(x-y)/y)
            siz = err.size
            err = np.sort(err)[.1*siz:-.1*siz]
            return err.mean()
            return ((x-y)**2/x.size).sum()**.5 / (max(x.max(),y.max())-min(x.min(),y.min()))
        
        if split is None:
            plt.plot(refs, cmps, '.')
            print error_name + ' of ' + title+':', error(refs,cmps)
        else:
            ##label = [reduce(getattr, [t[1]]+split.split('.')) for t in tree]
            label = [reduce(getattr, [m]+split.split('.')) for m in meta]
            import time
            if isinstance(label[0], time.struct_time):
                label = [' '.join(map(str,(l.tm_year,l.tm_mon,l.tm_mday))) for l in label]
            
            label = np.array(label)
            label_set = np.unique(label)
            color = ['b','g','r','c','m','y','k']
            print '---', title, '---'
            for i,lab in enumerate(label_set):
                x = refs[label==lab]
                y = cmps[label==lab]
                if merge_unique:
                    pos = np.concatenate([x[:,None],y[:,None]],axis=-1)
                    pos = np.ascontiguousarray(pos).view([('x',pos.dtype),('y',pos.dtype)])
                    v,s = np.unique(pos, return_inverse=1)
                    size = np.bincount(s)
                    x,y  = v['x'],v['y']
                else:
                    size = 1
                plt.scatter(x, y, s=10*size, c=color[i%len(color)], edgecolors='none', label=lab)
                print error_name +' of '+lab+':', error(x,y)
            if legend:
                plt.legend(loc=0)
                
        ax = plt.gca()
        ax.set_xlabel('reference')
        ax.set_ylabel('measurements')
        ax.set_title(title)
        
        ax.set_ylim(0,bound)
        ax.set_xlim(0,bound)
        
        ax.tree_data = _Mapping(stat=stat, trees=meta, x=refs, y=cmps) ##tree=>meta?

def make_tree_compare(reference, compared, keys=None, storage_entry=None, verbose=False):
    """
    `reference` and `compared` are 2 list of dataset. They should
     - contain a 'metadata' attribute
     - be loader object (see datastructure.Data)
     - (once loaded) have the 'tree' attribute
     
    `keys` is a list of the metadata attributes used to match dataset elements
    if None, use all metadata
    """
    from .pipeline.dataset import get_column
    # remove extra element of 'compared'
    if keys:
        def multiget(x,multiattr): 
            return reduce(lambda a,b: getattr(a,b,None), multiattr.split('.'), x)
        def to_key(x):
            return tuple(map(lambda attr: repr(multiget(x,attr)), keys))
    else:
        def to_key(x):
            return repr(x)##x.multilines_str(max_width=2**60)#repr(to_tuple(x))
    
    ##for r in reference:
    ##    print to_key(r.metadata)
    ##raise Exception()
    compared = dict([(to_key(c.metadata),c) for c in compared])
    compared = [compared.get(to_key(r.metadata),None) for r in reference]

    # find tree that are not found
    missing = []
    image = []
    ref = []
    cpr = []
    for r,c in zip(reference,compared):
        rt = r.load().__dict__.get('tree', None)
        ct = c.load().__dict__.get('tree', None)
        if rt and ct:
            if verbose: print 'adding trees for', c.filename
            image.append(c.filename)
            ref.append(rt)
            cpr.append(ct)
        else:
            if verbose: print 'unavailable trees for', c.filename
            missing.append(c.filename)
    #return ref,cpr,missing
            
    return TreeCompareSequence(reference=ref, compared=cpr, image=image, filename=storage_entry), missing


# simple matching from distance matrix
# ------------------------------------
def direct_matching(d, max_d=None):
    """
    compute a simple 1-to-1 match from distance matrix `d`
    
    Iteratively select the match `(i,j)` with minimum distance in `d` but only 
    if `i` (from 1st dim) and `j`(from 2nd dim) are not already matched.
    
    If `max_d` is not None, don't match pairs with distance > `max_d`
    
    Return:
    
      - the list match pairs [(i0,j0),(i1,j1),...]
      - the list of unmatch i
      - the list of unmatch j
      
    example::
      
      d = np.ranom.randint(0,10,5,4)
      m,i,j = direct_match(d)
      print '\n'.join(str(ij)+' matched with d='+str(di) for ij,di in zip(m,d[zip(*m)]))
    """
    ni,nj = d.shape
    ij = _unravel(d.ravel().argsort().tolist(),d.shape)
    d  = np.sort(d.ravel()).tolist()
    
    if max_d is None: max_d=d[-1]
    
    match = []
    mi = set()
    mj = set()
    for (i,j),dij in zip(ij,d):
        if dij>max_d or i in mi or j in mj: continue
        match.append((i,j))
        mi.add(i)
        mj.add(j)
        
    return match, mi.symmetric_difference(range(ni)), mj.symmetric_difference(range(nj))


# topological and geometrical tree comparison
# -------------------------------------------
#    use vplants.pointreconstruction
def compare_mtg(ref,auto,display=False):
    """                                                       
    Call vplants.pointreconstruction.comparison.comparison_process
    on the two *mtg* input, `ref` and `auto`.
    """
    from vplants.pointreconstruction.comparison import comparison_process as mtg_cmp
    
    radius = auto.property('radius')
    for k in radius.keys(): radius[k] = 1
    radius = ref.property('radius')
    for k in radius.keys(): radius[k] = 1
    
    return mtg_cmp(ref,auto,display=display)
    
def compare_sequence(vs, storage='mtg_compare', display=True, slices=slice(None)):
    """
    For all tree structure in `vs.auto` and `vs.ref` TreeStat sequence
    call compare_mtg
    """
    import os
    storage = os.path.abspath(storage)
    if not os.path.exists(storage):
        os.mkdir(storage)
        
    res = []
    
    for i in range(len(vs.ref))[slices]:
        a = vs.auto[i].tree
        r = vs.ref[i].tree
        print '--- comparing file: ...', a.get_storage_entry()[-30:], '---'
        r.segment.radius = np.zeros(r.segment.size+1)
        a = split_mtg(a.to_mtg())
        r = split_mtg(r.to_mtg())
        
        for j,(ai,ri) in enumerate(zip(a,r)):
            try:
                match_ratio, topo_ratio = compare_mtg(ri,ai,display=display)
                res.append((i,j, match_ratio, topo_ratio))
            except:
                print '\033[31merror processing', i,j, '\033[30m'
            if display:
                k = raw_input('continue(y/n):')
                if k=='n': return
            else:
                print '----------------', i, j, '----------------' 
        
    return res

def split_mtg(g, scale=1):
    return map(g.sub_mtg, g.vertices(scale=1))
