import numpy as np
import scipy.ndimage as nd

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes

from rhizoscan.workflow  import Struct as _Struct

from rhizoscan.root.measurements import compute_tree_stat
from rhizoscan.root.measurements import statistic_name_list


class TreeCompare(_Struct):
    def __init__(self, reference, compared, image=None, filename=None):
        """
        Allow to compare a reference tree with the compared one.
        
        
        
        :Input:
          - reference: a reference tree
          - compared:  a compared tree
          - image:     optional related image - or image filename
          - filename:  optional file to save this TreeCompare object 
        """
        self.ref = reference.loader(attribute='metadata')
        self.cmp = compared .loader(attribute='metadata')
        self.img = image
        
        # match_tree
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
            
        def distance_matrix(x1,y1,x2,y2):
            x1 = x1.reshape(1,-1)
            y1 = y1.reshape(1,-1)
            x2 = x2.reshape(-1,1)
            y2 = y2.reshape(-1,1)
            return ((x1-x2)**2 + (y1-y2)**2)**.5

        rpid, rx, ry = seed_position(reference)
        cpid, cx, cy = seed_position(compared)
        
        #max_d = np.max(distance_matrix(rx,ry,rx,ry))
        d = distance_matrix(rx,ry,cx,cy)
        s1 = set(zip(range(d.shape[0]),np.argmin(d,axis=1)))
        s2 = set(zip(np.argmin(d,axis=0),range(d.shape[0])))
        
        self.plant_map = dict((rpid[p1],cpid[p2]) for p1,p2 in s1.intersection(s2))
        self.plant_missed = rpid.size - len(self.plant_map)
        
        # set save filename
        self.set_data_file(filename)
        
    def compute_stat(self, stat_names='all', mask=None, save=True):
        r = self.ref.load(merge=False)
        c = self.cmp.load(merge=False)
        compute_tree_stat(r, stat_names=stat_names, mask=mask)
        compute_tree_stat(c, stat_names=stat_names, mask=mask)
        self.ref = r.loader(attribute=['metadata','stat'])
        self.cmp = c.loader(attribute=['metadata','stat'])
        
        if save:
            self.save()


# statistical root tree comparison
# --------------------------------
class TreeCompareSequence(_Struct):
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
            
        # compute number of matched tree and unmatched ones
        self.matched_number   = sum(len(tc.plant_map) for tc in self.tc_list)
        self.unmatched_number = sum(tc.plant_missed   for tc in self.tc_list)
        
        if filename:
            self.set_data_file(filename)
            self.save()

    def compute_stat(self, stat_names='all', mask=None, save=True):
        for tc in self.tc_list: 
            tc.compute_stat(stat_names=stat_names, mask=mask, save=tc.get_data_file() is not None)
        if save:
            self.save()
            
    def _data_to_save_(self):
        s = self.__copy__()
        s.tc_list = [tc._data_to_save_() for tc in s.tc_list]
        return _Struct._data_to_save_(s)

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
        
        ax.tree_data = _Struct(stat=stat, trees=meta, x=refs, y=cmps) ##tree=>meta?

def tree_compare_from_db(reference, compared, tree_suffix='.tree', filename=None):
    from .pipeline.database import get_column
    # remove extra element of 'compared'
    def to_key(x):
        return x.multilines_str(max_width=2**60)#repr(to_tuple(x))
    compared = dict([(to_key(c.metadata),c) for c in compared])
    compared = [compared[to_key(r.metadata)] for r in reference]

    # create tree list
    reference = get_column(reference, suffix=tree_suffix)
    compared  = get_column(compared,  suffix=tree_suffix)

    return TreeCompareSequence(reference=reference, compared=compared, filename=filename)





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
        print '--- comparing file: ...', a.get_data_file()[-30:], '---'
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
