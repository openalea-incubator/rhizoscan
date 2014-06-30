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
          - reference: reference tree         (*)
          - compared:  compared tree          (*)
          - image:     optional related image (*)
          - filename:  optional Data file to save this TreeCompare object
                       If given, automatically saves constructed object
                       
        (*) those should have suitable Data file set, as only loader are saved
        """
        self.ref = reference
        self.cmp = compared 
        self.img = image
        
        self.metadata = compared.metadata
        self._set_key()
        
        # set save filename
        if filename:
            self.set_file(filename)
            self.dump()
        
    def _set_key(self):
        self.key = self.cmp.get_file().get_url().split('/')[-1]
        
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
            x = nd.mean(t.node.x()[nseed],labels=lseed.reshape(-1,1),index=pid)
            y = nd.mean(t.node.y()[nseed],labels=lseed.reshape(-1,1),index=pid)
            
            return pid,x,y
            
        rpid, rx, ry = seed_position(self.get('ref'))
        cpid, cx, cy = seed_position(self.get('cmp'))
        
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
        compute_tree_stat(self.get('ref'), stat_names=stat_names, mask=mask)
        compute_tree_stat(self.get('cmp'), stat_names=stat_names, mask=mask)
        if save:
            self.dump()
           
    def add_comparison(self,name,value):
        self.setdefault('comparison',_Mapping())[name] = value
        
    def compare_structure(self, display=False):
        """
        Call `compare_structure` function and store results as dictionaries in 
        attributes `comparison.geometry` and `comparison.topology`
        """
        from rhizoscan.root.graph.mtg import tree_to_mtg
        from os.path import sep
        c = self.get('cmp')
        r = self.get('ref')
        
        f = sep.join(c.get_file().get_url().split(sep)[-1:])
        
        print '--- comparing file: .../%s ---' % f
        print '============================' + '='*len(f)
        r.segment.radius = np.zeros(r.segment.number())
        c.segment.radius = np.zeros(c.segment.number())
        rmtg = tree_to_mtg(r)
        cmtg = tree_to_mtg(c)
        c = _split_mtg(cmtg)
        r = _split_mtg(rmtg)
        
        geom = {}
        topo = {}
        for i,(ci,ri) in enumerate(zip(c,r)):
            print 'comparing image root %2d' % i
            print '-----------------------'

            try:
                match_ratio, topo_ratio = compare_structure(ci,ri,display=display)
                geom[i] = match_ratio
                topo[i] = topo_ratio
                if display:
                    k = raw_input('geo:%.2f, topo:%.2f  - continue(y/n):' % (match_ratio,topo_ratio))
                    if k=='n': return
            except:
                print '********** error processing root %d **********' % i
            print '------------ end of root %d fo file %s ------------' % (i,f) 
        
        self.add_comparison('geometry', geom)
        self.add_comparison('topology', topo)
        
    def compare_stat(self):
        """
        add the comparison fields for stat entries:
        'axe1_length','axe2_length_mean','axe2_length_total','axe2_number','plant_hull','total_length'
        """
        stats = ['axe1_length','axe2_length_mean','axe2_length_total','axe2_number','plant_hull','total_length']
        for s in stats:
            dif = {}
            r = self.get('ref').stat[s]
            c = self.get('cmp').stat[s]
            for pid in set(r.keys()).intersection(c.keys()):
                dif[pid] = c[pid]/max(r[pid],2**-16)
            self.add_comparison(s,dif)
    
    def compare(self):
        """ call compare_structure and compare_stat """
        self.compare_structure()
        self.compare_stat()
           
    def clear_temporary_attribute(self):
        """ clear the temporary attributs and call clear() """ 
        _Mapping.clear_temporary_attribute(self)
        self.clear()
        
    def clear(self):
        """ replace trees by their loader """
        if not _Mapping.is_loader(self.__dict__['ref']): self.ref = self.ref.get_loader()
        if not _Mapping.is_loader(self.__dict__['cmp']): self.cmp = self.cmp.get_loader()


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
            self.set_file(filename)
            self.dump()

    def match_plants(self, max_distance=None, save=True, verbose=False):
        """ ##consider doing the match directly through compute_stat """
        for tc in self.tc_list:
            if verbose: print 'match plant of:', tc.cmp.get_file().get_url()
            tc.match_plants(max_distance=max_distance)
            if verbose:
                missed = (len(tc.mapping.plant_missed_ref),len(tc.mapping.plant_missed_cmp))
                if sum(missed):
                    print '    *** missed ref: %d cmp: %d' % missed
        # compute number of matched tree and unmatched ones
        self.matched_number   = sum(len(tc.mapping.plant) for tc in self.tc_list)
        self.matched_ref_miss = sum(len(tc.mapping.plant_missed_ref) for tc in self.tc_list)
        self.matched_cmp_miss = sum(len(tc.mapping.plant_missed_cmp) for tc in self.tc_list)
        #self.unmatched_number = sum(tc.plant_missed   for tc in self.tc_list)
        
        if save and self.get_file():
            self.dump()
        
    def compute_stat(self, stat_names='all', mask=None, save=True, verbose=False):
        for tc in self.tc_list: 
            if verbose: print 'compute stat of:', tc.cmp.get_file().get_url()
            tc.compute_stat(stat_names=stat_names, mask=mask, save=tc.get_file() is not None)
        if save and self.get_file():
            self.dump()

    def clear(self):
        """ call clear on all TreeCompare """
        for tc in self.tc_list:
            tc.clear()
        
    def __store__(self):
        s = self.__copy__()
        s.tc_list = [tc.__parent_store__() for tc in s.tc_list]
        return _Mapping.__store__(s)
        
    def compare(self, save=True, verbose=False):
        """
        Call compare of all TreeCompare in tc_list
        """
        for tc in self.tc_list: 
            if verbose: print 'comparison of:', tc.cmp.get_file().get_url()
            tc.compare(stat_names=stat_names, mask=mask, save=tc.get_file() is not None)
        
        if save and self.get_file():
            self.dump()

    def compare_stat(self, save=True, verbose=False):
        """
        Call compare_stat of all TreeCompare in tc_list
        """
        for tc in self.tc_list: 
            if verbose: print 'comparison of stats of:', tc.cmp.get_file().get_url()
            tc.compare_stat()
        
        if save and self.get_file():
            self.dump()
            
            
def plot_stat(vs, value='axe1_length', title=None, prefilter=None, split=None, legend=str, merge_unique=False, scale=1, cla=True, mask=None, clip=None):
    import matplotlib.pyplot as plt
    
    title = title if title is not None else value
    
    ref_val = []
    cmp_val = []
    tc_flat = []
    plantid = []
    if prefilter is None: prefilter = lambda st: st
    for tc in vs.tc_list:
        sr = tc.get('ref').stat[value]
        sc = tc.get('cmp').stat[value]

        tc_flat.extend([tc]*len(tc.mapping.plant))
        ref_val.extend([prefilter(sr[pid]) for pid in tc.mapping.plant.keys()])
        cmp_val.extend([prefilter(sc[pid]) for pid in tc.mapping.plant.values()])
        plantid.extend(tc.mapping.plant.keys())

    if mask:
        mask = [(tc.key,pid) in mask for tc,pid in zip(tc_flat,plantid)]
        tc_flat = [tc_flat[i] for i,m in enumerate(mask) if m]
        ref_val = [ref_val[i] for i,m in enumerate(mask) if m]
        cmp_val = [cmp_val[i] for i,m in enumerate(mask) if m]
        plantid = [plantid[i] for i,m in enumerate(mask) if m]
        
    cmp_val  = np.array(cmp_val)*scale
    ref_val  = np.array(ref_val)*scale
    
    def print_error(title, x,y):
        return
        err = (abs(x-y)/np.maximum(x,y))
        siz = err.size
        err = np.sort(err)[.1*siz:-.1*siz]
        print 'Average percentage error of ' + title+':', err.mean()
        ##return ((x-y)**2/x.size).sum()**.5 / (max(x.max(),y.max())-min(x.min(),y.min()))#'normalized RMS Error'
    
    if clip:
        ref_val = np.clip(ref_val, 0, clip)
        cmp_val = np.clip(cmp_val, 0, clip)
    ax = _plot_tc(tc=tc_flat,x=ref_val,y=cmp_val, plant_id=plantid,
                  title=title, xlabel='reference', ylabel='measurements',
                  split=split, legend=legend, cla=cla, print_fct=print_error,
                  merge_unique=merge_unique)

    bound = max(max(ref_val), max(cmp_val))
    plt.plot([0,bound], [0,bound], 'r')
    ax.set_ylim(0,bound)
    ax.set_xlim(0,bound)

def plot_compare(vs, value='axe1_length', sort=None, title=None, xlabel='plants', split=None, legend=str, content='scatter', cla=True, clip=5, mask=None):
    """
    mask: a set/list of tuple (tc.key,plant_id) to display
    
    example to make a mask::
     
      mask = set((tc.key,k) for tc in vs.tc_list for k,v in tc.comparison.axe1_length.iteritems() if v>.9)
    
    """
    import matplotlib.pyplot as plt
    
    title = title if title is not None else value
    
    # list all values to plot, and related data
    cmp_val = []
    tc_flat = []
    plantid = []
    for tc in vs.tc_list:
        if tc.comparison.has_key(value):
            cmp_dict = tc.comparison[value]
            tc_flat.extend([tc]*len(cmp_dict.keys()))
            plantid.extend(cmp_dict.keys())
            cmp_val.extend(cmp_dict.values())

    if mask:
        mask = [(tc.key,pid) in mask for tc,pid in zip(tc_flat,plantid)]
        tc_flat = [tc_flat[i] for i,m in enumerate(mask) if m]
        cmp_val = [cmp_val[i] for i,m in enumerate(mask) if m]
        plantid = [plantid[i] for i,m in enumerate(mask) if m]
        
    cmp_val = np.clip(cmp_val, 0, clip)
    ax = _plot_tc(tc=tc_flat,x=np.arange(cmp_val.size),y=cmp_val, plant_id=plantid,
                  title=title, xlabel=xlabel, ylabel='measurement/reference',
                  split=split, legend=legend, content=content, cla=cla)
    
    bound = plt.axis()[1]
    plt.plot([0,bound], [1,1], 'g')
    ax.set_ylim(0,max(cmp_val.max(),1.1))
    ax.set_xlim(0,bound)

def _plot_tc(tc,x,y,plant_id, title,xlabel,ylabel, split=False, merge_unique=False, content='scatter', legend=str, print_fct=None, cla=True):
    """
    actual ploting done by `plot_stat` and `plot_compare`
    
    legend: a function that convert label into string
    content: either 'scatter' or 'box'
    """
    import matplotlib.pyplot as plt
    from matplotlib import pylab
    from matplotlib.backends import backend, interactive_bk
    
    if cla:
        plt.cla()
    
    if split is None:
        plt.plot(x, y, '.')
        if print_fct: print_fct(title,x,y)
    else:
        if isinstance(split,basestring):
            split = ['metadata'] + split.split('.')
            label = [reduce(getattr, [t]+split) for t in tc]
            label = np.array(label)
            label_set = np.unique(label)
        else:
            label=[]
            for spl in split:
                spl = ['metadata'] + spl.split('.')
                label.append([reduce(getattr, [t]+spl) for t in tc])
            from rhizoscan.ndarray import unique_rows
            label = np.array(label).T
            label_set = unique_rows(label)
        
        if content=='scatter':
            color = ['b','g','r','c','m','y','k']
            for i,lab in enumerate(label_set):
                lab_mask = label==lab
                if lab_mask.ndim>1: lab_mask = lab_mask.all(axis=-1)
                yi = y[lab_mask]
                xi = x[lab_mask]
                if merge_unique:
                    pos = np.concatenate([xi[:,None],yi[:,None]],axis=-1)
                    pos = np.ascontiguousarray(pos).view([('x',pos.dtype),('y',pos.dtype)])
                    v,s = np.unique(pos, return_inverse=1)
                    size = np.bincount(s)
                    xi,yi  = v['x'],v['y']
                else:
                    size = 2
                label_str = legend(lab)
                colori = color[i%len(color)]
                plt.scatter(xi, yi, s=8*size, c=colori, edgecolors='none', label=label_str)
                if print_fct: print_fct(title+' - '+label_str, xi,yi)
            plt.legend(loc=0) 
        else:  # if content=='box':
            boxes = []
            names = []
            for i,lab in enumerate(label_set):
                lab_mask = label==lab
                if lab_mask.ndim>1: lab_mask = lab_mask.all(axis=-1)
                #xi = x[lab_mask]
                boxes.append(y[lab_mask])
                names.append(legend(lab))
            bp = plt.boxplot(boxes)
            for f in  bp['fliers']: f.remove()
            plt.xticks(range(1,len(names)+1),names)
            
        
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if backend in interactive_bk:
        ax.tree_data = _Mapping(title=title,x=x,y=y,tc=tc, plant_id=plant_id)
        flag = '_ROOT_MEASUREMENT_CB'
        if not hasattr(plt.gcf(),flag):
            cid = pylab.connect('button_press_event', _plot_axe_selected_tree)
            setattr(plt.gcf(),flag,cid)
        
    return ax

def _plot_axe_selected_tree(event):
    """
    Called interactively by `plot_stat` and `plot_compare`
    """
    from matplotlib import pyplot as plt
    #print event
    #if event.button!=3: return
    
    plt.axes(event.inaxes)
    data  = getattr(plt.gca(),'tree_data')
    x,y   = event.xdata, event.ydata
    d2    = ((data.x-x)**2 + (data.y-y)**2)
    match = np.argmin(d2)
    
    tc  = data.tc[match]
    pid = data.plant_id[match]
    ref_tree = tc.get('ref')
    cmp_tree = tc.get('cmp')

    # just in case
    tc.ref.segment.node_list = tc.ref.node
    tc.ref.axe._segment_list = tc.ref.segment
    ref_tree.node.set_segment(ref_tree.segment)
    ##cmp_tree.node.set_segment(cmp_tree.segment)
    
    f=plt.gcf().number
    plt.figure(f+41)
    
    # plot reference in red
    ref_tree.plot(bg='k', ac=.4*np.ones((1,3))*(ref_tree.axe.plant==pid)[:,None],linewidths=3)
    
    # plot auto in color map suitable to shown stat 
    #order = cmp_tree.axe.order()
    #amask = cmp_tree.axe.plant==pid
    #if   'axe1' in title:  sc = order[saxe*(saxe>0)]+2
    #elif 'axe2' in title:  sc = order[saxe*(saxe>0)]+2
    #elif stat=='total_length': sc = 2*(saxe>0)+3*(saxe==-1)
    #else:                      sc = 2*(saxe>0)+3*(saxe==-1)#saxe*(order[saxe*(saxe>0)]==2)
    #elif stat=='total_length': sc = 2*(saxe>0)+3*(saxe==-1)
    #elif stat=='axe2_number':  sc = 2*(saxe>0)+3*(saxe==-1)#saxe*(order[saxe*(saxe>0)]==2)
    cmp_tree.plot(bg=None,max_shift=3)#, ac=cmp_tree.axe.order())
    
    # display a line abov and below the selected root plant
    #slist = set([s for slist in a.tree.axe.segment[a.tree.axe.plant==pid] for s in slist])
    #slist.discard(0)
    #smask = a.tree.axe.plant[a.tree.segment.axe]==pid
    #smask[0] = 0
    #smask[a.tree.segment.seed!=pid] = 0
    smask = cmp_tree.segment.seed==pid
    node  = np.unique(cmp_tree.segment.node[smask])
    node  = node[node>0]
    pos   = cmp_tree.node.position[:,node]
    x0,y0 = pos.min(axis=1)*0.95
    x1,y1 = pos.max(axis=1)*1.05
    plt.plot([x0,x0,x1,x1],[y1,y0,y0,y1], 'g', linewidth=3)
    #plt.plot([x0,x1],[y1,y1], 'g', linewidth=3)
    
    # display title
    plt.title(data.title)
    print '----------------'
    print 'plant id:', pid
    print repr(cmp_tree.metadata)
        
        
def make_tree_compare(reference, compared, keys=None, file_object=None, verbose=False):
    """
    `reference` and `compared` are 2 lists of dataset. They should
     - contain a 'metadata' attribute
     - be loader object (see datastructure.Data)
     - (once loaded) have the 'tree' attribute
     
    `keys` is a list of the metadata attributes used to match dataset elements
    if None, use dataset __key__ (with '/' replaced by '_')
    """
    # remove extra element of 'compared'
    if keys:
        def multiget(x,multiattr): 
            return reduce(lambda a,b: getattr(a,b,None), multiattr.split('.'), x)
        def get_key(x):
            return tuple(map(lambda attr: repr(multiget(x.metadata,attr)), keys))
    else:
        def get_key(x):
            return x.__key__.replace('/','_')
    
    compared = dict([(get_key(c),c) for c in compared])
    compared = [compared.get(get_key(r),None) for r in reference]
    
    # find tree that are not found
    missing = []
    image = []
    ref = []
    cpr = []
    for r,c in zip(reference,compared):
        rt = r.load().get('tree', None)
        ct = c.load().get('tree', None)
        if rt and ct:
            if verbose>1: print 'adding trees for', c.filename
            image.append(c.filename)
            rt.metadata = r.metadata
            ct.metadata = c.metadata
            rt.__loader_attributes__ = ['metadata']
            ct.__loader_attributes__ = ['metadata']
            ref.append(rt.get_loader())
            cpr.append(ct.get_loader())
        else:
            if verbose: print 'unavailable trees for', c.filename
            missing.append(c.filename)
    #return ref,cpr,missing
            
    return TreeCompareSequence(reference=ref, compared=cpr, image=image, filename=file_object), missing


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
      - the set of unmatch i
      - the set of unmatch j
      
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
def compare_structure(ref,auto,display=False):
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
    
def compare_structure_sequence(vs, display=True, slices=slice(None), file_id_part=2, append_file=None):
    """
    For all TreeCompare in `vs.tc_list` call compare_structure
    """
    from os.path import sep
    res = []
    
    for i,tc in np.array(list(enumerate(vs.tc_list)))[slices]:
        c = tc.get('cmp')
        r = tc.get('ref')
        tc.clear()
        
        f = sep.join(c.get_file().get_url().split(sep)[-file_id_part:])
        
        print '--- comparing file: ...', f, '---'
        r.segment.radius = np.zeros(r.segment.number())
        c.segment.radius = np.zeros(c.segment.number())
        rmtg = tree_to_mtg(r)
        cmtg = tree_to_mtg(c)
        c = _split_mtg(cmtg)
        r = _split_mtg(rmtg)
        
        ##return c,r
        for j,(ci,ri) in enumerate(zip(c,r)):
            print 'comparing image %2d root %2d' % (i,j)
            print '----------------------------'
            print 'max scale', ri.max_scale(), ci.max_scale()
            ##return ri,ci
            try:
                match_ratio, topo_ratio = compare_structure(ci,ri,display=display)
                res.append((i,f,j+1, match_ratio, topo_ratio))
            except:
                print '********** error processing', i,j, '**********'
            if display:
                k = raw_input('continue(y/n):')
                if k=='n': return
            print '----------------', i, j, '----------------' 
        
    if append_file:
        with open(append_file,'a') as f:
            print s,e, len(res)
            for re in results:
                re = ' '.join(map(str,re[1:]))
                f.write(r+'\n')
        
    return res
    
def apply_structure_comparison(vs, comparison):
    if isinstance(comparison,basestring):
        comparison = read_structure_comparison_file(comparison)
        
    tc_dict = {}
    for tc in vs.tc_list:
        tc_dict[tc.key] = tc
        
    for tc_key, comp in comparison.iteritems():
        tc = tc_dict[tc_key]
        tc.add_comparison('geometry',comp['geometry'])
        tc.add_comparison('topology',comp['topology'])
        

def read_structure_comparison_file(filename):
    with open(filename) as f:
        saved_topo = map(str.split,map(str.strip,f.readlines()))
        
    topo = {}
    for t in saved_topo:
        comp = topo.setdefault(t[0],dict(geometry={},topology={}))
        pid = int(t[1])
        comp['geometry'][pid] = float(t[2])
        comp['topology'][pid] = float(t[3])
    
    return topo

def _split_mtg(g, scale=1):
    G = map(g.sub_mtg, g.vertices(scale=scale))
    return [gi for gi in G if len(gi.vertices())>1]
