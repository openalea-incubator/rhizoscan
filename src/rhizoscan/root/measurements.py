import numpy as np
import scipy as sp

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes

from rhizoscan.workflow import Sequence as _Sequence 
from rhizoscan.workflow import Struct   as _Struct 
from rhizoscan.workflow import Data     as _Data
from rhizoscan.tool     import _property

# measurments for unordered list of trees object having a metadata attribut
# -------------------------------------------------------------------------
def _tree_stat(f):
    """ decorator that declare a tree statistic from the function that compute it """
    stat_list[f.func_name] = f
    return f
    
stat_list = dict()    # store the declared tree statistics
@_aleanode('stat_name')
def statistic_name_list():
    return sorted(stat_list.keys())

class TreeStat(_Struct):
    def __init__(self,tree):
        self._tree_file = tree.get_data_file()
        self.metadata = tree.metadata
        
    @_property
    def tree(self):  ## would be nice to have such behaviours automatized...
        """ tree object relative to this TreeStat object """
        if not hasattr(self,'_tree'):
            self._tree = _Data.load(self._tree_file)
            self.temporary_attribut.add('_tree')
        return self._tree
        
    def clear_tree_data(self):
        if hasattr(self,'_tree'):
            del self._tree
            self.temporary_attribut.discard('_tree')  ## make the clear-tmp-attr robust to missing attr ?
    
    def compute_stat(self, stat_names='all', mask=None):
        """ compute all statistic listed in stat_name, using optional mask filter function"""
        if stat_names=='all':
            run = stat_list
        else:
            run = dict([(n,stat_list[n]) for n in stat_names])
            
        for name, fct in run.iteritems():
            self[name] = fct(self.tree, mask=mask)
            
    def __str__(self):
        return self.__class__.__name__ + ' of: %s' % self._tree_file
    def __repr__(self):
        return str(self)
        
class TreeCompare(_Struct):
    def __init__(self, auto, ref, filename, image=None):
        if hasattr(auto[0], 'output') and not hasattr(auto[0],'axe'):
            self.auto = [TreeStat(t) for t in _Sequence([a.output+'.tree' for a in auto])]
        else:
            self.auto = [TreeStat(t) for t in auto]
        if hasattr(ref[0], 'output') and not hasattr(ref[0],'axe'):
            self.ref  = [TreeStat(t) for t in _Sequence([r.output+'.tree' for r in ref])]
        else:
            self.ref  = [TreeStat(t) for t in ref]
            
        self.image = image
        
        self.set_data_file(filename)

    def compute_stat(self, stat_names='all', mask=None, save=True):
        for a in self.auto: 
            a.compute_stat(stat_names=stat_names, mask=mask)
            a.clear_temporary_attribut()
        for r in self.ref :
            r.compute_stat(stat_names=stat_names, mask=mask)
            r.clear_temporary_attribut()
        if save:
            self.save()
            
    def _data_to_save_(self):
        s = self.__copy__()
        s.auto = [a._data_to_save_() for a in s.auto]
        s.ref  = [r._data_to_save_() for r in s.ref]
        
        return _Struct._data_to_save_(s)
                
@_aleanode(name='TreeCompare')
def make_TreeCompare(auto, ref, filename='.tmp-TreeCompare', compute_stat='all'):
    tc = TreeCompare(auto=auto, ref=ref, filename=filename)
    if compute_stat:
        tc.compute_stat(stat_names=compute_stat)
    return tc
                
@_tree_stat
def axe1_length(tree, mask=None):
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',None)
    ax1L = get_axes_property(tree, 'length', order=1, mask=None if mask is None else mask(tree), scale=scale)
    return dict((k,v[0]) for k,v in ax1L.iteritems())
@_tree_stat
def axe2_length(tree, mask=None): 
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',None)
    return get_axes_property(tree, 'length', order=2, mask=None if mask is None else mask(tree), scale=scale)
@_tree_stat
def axe2_length_total(tree, mask=None): 
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',None)
    L = get_axes_property(tree, 'length', order=2, mask=None if mask is None else mask(tree), scale=scale)
    return dict([(k,v.sum()) for k,v in L.iteritems()])
@_tree_stat
def axe2_length_mean(tree, mask=None): 
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',None)
    L = get_axes_property(tree, 'length', order=2, mask=None if mask is None else mask(tree), scale=scale)
    return dict([(k,v.mean()) for k,v in L.iteritems()])
@_tree_stat
def total_length(tree, mask=None):
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',None)
    tl = get_axes_property(tree, 'length', mask=None if mask is None else mask(tree),scale=scale)
    for k,l in tl.iteritems():
        tl[k] = l.sum()
    return tl
@_tree_stat
def axe2_number(tree, mask=None):
    ax_mask= (tree.axe.plant>0)
    if mask is not None: 
        ax_mask &= mask(tree)
    pl_id  = np.unique(tree.axe.plant[ax_mask])
    number = np.bincount(tree.axe.plant[ax_mask], tree.axe.order[ax_mask]==2)
    return dict([(i,n) for i,n in enumerate(number) if i in pl_id])
    
@_tree_stat
def ramification_length(tree, mask=None):
    from scipy.ndimage import maximum
    ax_mask= (tree.axe.plant>0)
    if mask is not None: 
        ax_mask &= mask(tree)
    pl_id  = np.unique(tree.axe.plant[ax_mask])
    
    branch = tree.axe.sparent[tree.axe.order==2]
    plant  = tree.axe.plant[tree.segment.axe[branch]]
    bdist  = tree.segment.axelength[branch]
    
    scale = getattr(getattr(tree,'metadata',None),'px_ratio',1)

    return dict(zip(pl_id, maximum(bdist*scale,plant, pl_id)))
    
@_tree_stat
def ramification_percent(tree, mask=None):
    ram_length = ramification_length(tree=tree,mask=mask)
    ax1_length = axe1_length(tree=tree,mask=mask)
    
    return dict([(plid,ram_length[plid]/ax1_l) for plid,ax1_l in ax1_length.iteritems()])
    
# ploting
def plot(tc, stat='axe1_length', title=None, prefilter=None, split=None, legend=True, merge_unique=False, scale=1):
    import matplotlib.pyplot as plt
    
    auto = []
    ref  = []
    tree = []
    if prefilter is None: prefilter = lambda st: st
    for a,r in zip(tc.auto, tc.ref):
        sa = a[stat]
        sr = r[stat]
        # 'valid' plant id: plant id present in both auto and ref 
        pl_id = set(r[stat].keys()).intersection(a[stat].keys())

        tree.extend([(pid,a,r) for pid in pl_id])
        auto.extend([prefilter(sa[pid]) for pid in pl_id])
        ref .extend([prefilter(sr[pid]) for pid in pl_id])

    auto  = np.array(auto)*scale
    ref   = np.array(ref) *scale
    
    bound = max(max(ref), max(auto))
    plt.cla()
    plt.plot([0,bound], [0,bound], 'r')
    
    if split is None:
        plt.plot(ref, auto, '.')
    else:
        label = [reduce(getattr, [t[1]]+split.split('.')) for t in tree]
        import time
        if isinstance(label[0], time.struct_time):
            label = [' '.join(map(str,(l.tm_year,l.tm_mon,l.tm_mday))) for l in label]
        
        label = np.array(label)
        label_set = np.unique(label)
        color = ['b','g','r','c','m','y','k']
        for i,lab in enumerate(label_set):
            x = ref[label==lab]
            y = auto[label==lab]
            if merge_unique:
                pos = np.concatenate([x[:,None],y[:,None]],axis=-1)
                pos = np.ascontiguousarray(pos).view([('x',pos.dtype),('y',pos.dtype)])
                v,s = np.unique(pos, return_inverse=1)
                size = np.bincount(s)
                x,y  = v['x'],v['y']
            else:
                size = 1
            plt.scatter(x, y, s=10*size, c=color[i%len(color)], edgecolors='none', label=lab)
        if legend:
            plt.legend(loc=0)
            
    ax = plt.gca()
    ax.set_xlabel('reference')
    ax.set_ylabel('measurements')
    ax.set_title(title if title is not None else stat)
    
    ax.set_ylim(0,bound)
    ax.set_xlim(0,bound)
    
    ax.tree_data = _Struct(stat=stat, trees=tree, x=ref, y=auto)

@_aleanode(name='treeCompare_plot')
def multi_plot(tc, split='metadata.date', scale=1):
    import matplotlib.pyplot as plt
    from   matplotlib import pylab
    import time
    
    plt.clf()
    plt.subplot(2,2,1)#3,1)#plt.figure(3)
    plot(tc, stat='total_length', title='Total root length', split=split, legend=True, scale=scale)
    plt.subplot(2,2,2)#3,2)#plt.figure(1)
    plot(tc, stat='axe1_length', title='Length of primary axes', split=split, legend=False, scale=scale)
    plt.subplot(2,2,3)#3,3)#plt.figure(2)
    plot(tc, stat='axe2_length_total', title='Total length of secondary axes', split=split, legend=False, scale=scale)
    plt.subplot(2,2,4)#3,4)#plt.figure(4)
    plot(tc, stat='axe2_number', title='Number of secondary axes', split=split, legend=False, merge_unique=1)
    #plt.subplot(2,3,5)#plt.figure(4)
    #plot(tc, stat='ramification_length', title='longueur de ramification', split=split, legend=False)
    
    plt.subplot(2,2,1).set_xlabel('')
    plt.subplot(2,2,2).set_xlabel('')
    plt.subplot(2,2,2).set_ylabel('')
    plt.subplot(2,2,4).set_ylabel('')
    
    def display_tree(event):
        if event.button!=3: return
        plt.axes(event.inaxes)
        data = getattr(plt.gca(),'tree_data')
        x,y  = event.xdata, event.ydata
        d2   = ((data.x-x)**2 + (data.y-y)**2)
        
        stat = data.stat
        pid,a,r = data.trees[np.argmin(d2)]

        f=plt.gcf().number#get_figure()
        #plt.figure(f+41)
        plt.subplot(2,3,6)
        
        # plot reference in red
        r.tree.plot(bg='k', sc='r')
        
        # plot auto in color map suitable to shown stat 
        saxe  = a.tree.segment.axe
        order = a.tree.axe.order
        ind   = saxe>=-1
        amask = a.tree.axe.plant==pid
        smask = amask[a.tree.segment.axe]
        smask &= (a.tree.segment.node!=0).all(axis=1)
        if   stat=='axe1_length':  sc = order[saxe*(saxe>0)]+2
        elif stat=='axe2_length':  sc = order[saxe*(saxe>0)]+2
        elif stat=='total_length': sc = 2*(saxe>0)+3*(saxe==-1)
        else:                      sc = 2*(saxe>0)+3*(saxe==-1)#saxe*(order[saxe*(saxe>0)]==2)
        #elif stat=='total_length': sc = 2*(saxe>0)+3*(saxe==-1)
        #elif stat=='axe2_number':  sc = 2*(saxe>0)+3*(saxe==-1)#saxe*(order[saxe*(saxe>0)]==2)
        a.tree.plot(bg=None, sc=sc, indices=ind)#smask)
        
        # display a line abov and below the selected root plant
        #slist = set([s for slist in a.tree.axe.segment[a.tree.axe.plant==pid] for s in slist])
        #slist.discard(0)
        #smask = a.tree.axe.plant[a.tree.segment.axe]==pid
        #smask[0] = 0
        #smask[a.tree.segment.seed!=pid] = 0
        smask = a.tree.segment.seed==pid
        node  = np.unique(a.tree.segment.node[smask])
        node = node[node>0]
        pos   = a.tree.node.position[:,node]
        x0,y0 = pos.min(axis=1)*0.95
        x1,y1 = pos.max(axis=1)*1.05
        plt.plot([x0,x0,x1,x1],[y1,y0,y0,y1], 'g', linewidth=3)
        #plt.plot([x0,x1],[y1,y1], 'g', linewidth=3)
        
        # display title
        meta = a.tree.metadata
        title = meta.genotype.name + ' at ' + time.asctime(meta.date)
        plt.title(title)
        print title
        #plt.figure(f)
        
    #flag = '_ROOT_MEASUREMENT_CB'
    #if not hasattr(plt.gcf(),flag):
    #    cid = pylab.connect('button_press_event', display_tree)
    #    setattr(plt.gcf(),flag,cid)

@_aleanode()
def cmp_plot(db, stat, key1, key2, update_stat=False, fig=42, outliers=.05):
    """
    db is a database (list) of root image descriptor (filename, metadata, output)
    stat is the stat to to plot
    key is a list of 2 metadata attribut to cluster the data by

    ##todo check it work with only one key
    """
    # retrieve stat from tree sequence
    tree_seq = _Sequence([d.output+'.tree' for d in db])
    db_value = [None]*len(db)
    
    for i,t in enumerate(tree_seq):
        if not hasattr(t,'stat') or update_stat:
            t.stat = TreeStat(t)
            t.stat.compute_stat()
            t.save()
        ##scale = getattr(t.metadata, 'px_ratio',1)
        db_value[i] = t.stat[stat].values()##[v*scale for v in t.stat[stat].values()]
    
    # manage key arguments
    key = [key1,key2]
    key = [['metadata']+k.split('.') for k in key]
    if len(key)==1: 
        key = ['', key]
    k1, k2 = key
    
    # get all possible key values (for both keys)
    def mget(d,key):
        return reduce(lambda x,f: getattr(x,f,None),[d]+key)
    
    k_list1 = [mget(d,k1) for d in db]      # keys of all element in db
    k_list2 = [mget(d,k2) for d in db]
    k_set1  = sorted(set(k_list1))          # sorted unique list of keys
    k_set2  = sorted(set(k_list2))
    k_map1  = dict([(k,i) for i,k in enumerate(k_set1)]) # keys indices in value array
    k_map2  = dict([(k,i) for i,k in enumerate(k_set2)])
    
    #import time
    #k_list1 = [k if not isinstance(k,time.struct_time) else time.asctime(k) for k in k_list1]
    #k_list2 = [k if not isinstance(k,time.struct_time) else time.asctime(k) for k in k_list2]
    #k_set1  = [k if not isinstance(k,time.struct_time) else time.asctime(k) for k in k_set1] 
    #k_set2  = [k if not isinstance(k,time.struct_time) else time.asctime(k) for k in k_set2]
    #k_map1  = dict([(k if not isinstance(k,time.struct_time) else time.asctime(k),v) for k,v in k_map1.iteritems()])
    #k_map2  = dict([(k if not isinstance(k,time.struct_time) else time.asctime(k),v) for k,v in k_map2.iteritems()])
    
    # cluster values by key group
    value = np.zeros((len(k_map1),len(k_map2)), dtype=object)
    value.ravel()[:] = [[] for i in range(value.size)]
    
    for v,k1,k2 in zip(db_value,k_list1,k_list2):
        value[k_map1[k1],k_map2[k2]].extend(v) 
        
    if outliers:
        def rm_outlier(v):
            n = int(len(v)*outliers)
            return sorted(v)[n:-n]
        value = np.vectorize(rm_outlier, otypes=[object])(value) 
        
    # compute mean and std for all keys
    m = np.vectorize(np.mean)(value)
    s = np.vectorize(np.std)(value)
    
    #return value,m,s
    
    # bar plot
    from matplotlib import pyplot as plt
    c = np.array(['b','g','k','w','y'])
    
    coord1,coord2 = np.mgrid[map(slice,value.shape)]
    x = (1+coord2+(value.shape[1]+1)*coord1)
    
    if fig: 
        plt.ion()
        plt.figure(fig)
    plt.cla()
    plt.bar(x.ravel()-.5, m.ravel(), yerr=s.ravel()**.5, width=1, color=c[coord2.ravel()%c.size], ecolor='r')
    
    # legend
    import time
    plt.xticks(x.mean(axis=1), [k if not isinstance(k,time.struct_time) else '%02d/%02d/%d' % (k.tm_mon,k.tm_mday,k.tm_year) for k in k_set1])
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_ylabel(stat)
    plt.title(stat)
    
    for label,k in k_map2.iteritems():
        plt.bar(x[0,0],0,color=c[k%c.size], label=label)
    plt.legend(loc=0, title='.'.join(key[1][1:]))
    
    

# measurements for (some obsolete) structure of tree objects
# ----------------------------------------------------------
def plant_mesure(P):
    projNum  = len(P)
    plantMax = 0
    timeMax  = 0
    plant_id = []
    for proj_id,p in enumerate(P):
        plantMax = max(plantMax,np.max(p.dtree[0].axe.plant))
        timeMax  = max(timeMax, len(p.dtree))
        pid = set(p.dtree[0].axe.plant)
        pid.discard(0)
        plant_id.extend([(proj_id,id) for id in pid])
        
    def kabs(x): return x*(x>0)
        
    P.area = dict() #np.array(projNum, plantMax, timeMax)
    for pid in plant_id:
        P.area[pid] = np.zeros((timeMax, 2)) # frame x [auto/dart]
        area = P.area[pid]
        # auto
        for i,t in enumerate(P[pid[0]].tree):
            nid = np.unique(t.segment.node[t.axe.plant[kabs(t.segment.axe)]==pid[1]])
            nid = nid[nid>0]
            area[i,0] = hull_area(t.node.position.T[nid])
        # dart
        for i,t in enumerate(P[pid[0]].dtree):
            nid = np.unique(t.segment.node[t.axe.plant[kabs(t.segment.axe)]==pid[1]])
            nid = nid[nid>0]
            area[i,1] = hull_area(t.node.position.T[nid])
            
    P.save()
    
    P.total_root_length = dict()
    P.axe1_length = dict()
    P.axe2_total_length = dict()
    P.axe2_number = dict()
    P.axe_number = dict()
    for pid in plant_id:
        P.total_root_length[pid] = np.zeros((timeMax, 2))
        P.axe1_length[pid]       = np.zeros((timeMax, 2))
        P.axe2_total_length[pid] = np.zeros((timeMax, 2))
        P.axe2_number[pid] = np.zeros((timeMax, 2))
        P.axe_number[pid]  = np.zeros((timeMax, 2))
    
    def compute_for(proj_id,tree, dart):
        for i,t in enumerate(tree):
            L = get_axes_property(t,'length')
            for plant_id in L.keys():
                P.total_root_length[(proj_id,plant_id)][i,dart] = L[plant_id].sum()
                P.axe_number[(proj_id,plant_id)][i,dart] = (L[plant_id]>5).sum()
                
            L = get_axes_property(t,'length',1)
            for plant_id in L.keys():
                P.axe1_length[(proj_id,plant_id)][i,dart] = L[plant_id].sum()
            L = get_axes_property(t,'length',2)
            for plant_id in L.keys():
                P.axe2_total_length[(proj_id,plant_id)][i,dart] = L[plant_id].sum()
                P.axe2_number[(proj_id,plant_id)][i,dart] = (L[plant_id]>5).sum()
                
    for proj_id,proj in enumerate(P):
        # auto
        compute_for(proj_id,proj.tree, 0)
        compute_for(proj_id,proj.dtree,1)
    P.save()
            
def plot_plant(P, msr,subplot=True, pxpmm=[1,1]):
    from matplotlib import pyplot as plt
    value = np.array(getattr(P,msr).values())
    value[:,0,:] /= pxpmm[0]
    value[:,1,:] /= pxpmm[1]
    plt.clf()
    if subplot: 
        plt.subplot(221)
        plt.gca().set_title(msr.replace('_',' ') + ' - t1')
    auto, = plt.plot(value[:,0,0],'-r')
    ref,  = plt.plot(value[:,0,1],'-g')
    plt.legend([auto,ref], ('auto', 'ref'))
    if subplot:      
        plt.subplot(222)
        plt.gca().set_title(msr.replace('_',' ') + ' - t2')
    plt.plot(value[:,1,0],'-r' if subplot else '--r')
    plt.plot(value[:,1,1],'-g' if subplot else '--g')
    if not subplot:
        plt.gca().set_title(msr.replace('_',' '))
    
    plt.subplot(223)
    t1, = plt.plot(value[:,0,0],value[:,0,1],'o')
    t2, = plt.plot(value[:,1,0],value[:,1,1],'s')
    plt.legend((t1,t2), ('t1', 't2'),loc=4)
    plt.xlabel('auto')
    plt.ylabel('reference')
    x0,x1,y0,y1 = plt.axis()
    plt.plot([min(x0,y0),max(x1,y1)],[min(x0,y0),max(x1,y1)])
    plt.gca().set_title(msr.replace('_',' ') + ' - correlation')
    
        
def axe_mesure(P):
    projNum  = len(P)
    plantMax = 0
    timeMax  = 0
    plant_id = []
    for proj_id,p in enumerate(P):
        plantMax = max(plantMax,np.max(p.dtree[0].axe.plant))
        timeMax  = max(timeMax, len(p.dtree))
        pid = set(p.dtree[0].axe.plant)
        pid.discard(0)
        plant_id.extend([(proj_id,id) for id in pid])
        
    # create data structure
    P.axe2_length = dict()
    P.axe2_insertion_angle = dict()
    P.axe1_curvature = dict()
    
    for pid in plant_id:
        P.axe2_length[pid]           = np.zeros((timeMax, 2), dtype=object)
        P.axe2_insertion_angle[pid]  = np.zeros((timeMax, 2), dtype=object)
        P.axe1_curvature[pid]        = np.zeros((timeMax, 2), dtype=object)
    
    # function which do the computations 
    def compute_for(proj_id,tree, ref):
        for i,t in enumerate(tree):
            L = get_axes_property(t,'length',2)
            A = get_axes_property(t,'insertion_angle',2)
            for plant_id in L.keys():
                P.axe2_length[(proj_id,plant_id)][i,ref] = L[plant_id]
                P.axe2_insertion_angle[(proj_id,plant_id)][i,ref] = A[plant_id]
            ## axe1_curvature                
                
    for proj_id,proj in enumerate(P):
        # auto
        compute_for(proj_id,proj.tree, 0)
        compute_for(proj_id,proj.dtree,1)
        
    P.save()

def distseq_plot(P, msr, plot=3, hist=10, pxpmm=[1,1]):
    msr_data = getattr(P,msr)
    value = np.array(msr_data.values())
    value[:,0,:] /= pxpmm[0]
    value[:,1,:] /= pxpmm[1]
    plant_id = msr_data.keys()
    
    msr_name = msr.replace('_', ' ')
    
    subplot = [2*plot, 3]

    if hist:
        bins = np.histogram(np.concatenate([v for v in value.ravel()]),bins=hist)[1]
        h = np.zeros(value.shape, dtype=object)
        for pid in range(value.shape[0]):
            for time in range(value.shape[1]):
                for ref in range(value.shape[2]):
                    h[pid,time,ref] = np.histogram(value[pid,time,ref],bins=bins)[0]
        v = h
    else:
        v = value

    # compute correlation
    corr = np.zeros(value.shape[:2])
    for pid in range(value.shape[0]):
        for time in range(value.shape[1]):
            corr[pid,time] = correlation(v[pid,time,0],v[pid,time,1])
        
    best = np.argsort(corr[:,0]*np.vectorize(np.sum)(value[:,0,0]))[-plot:]
    
    # plotting
    from matplotlib import pyplot as plt
    if hist:
        for i,id in enumerate(best):
            plt.subplot2grid((2*plot,3),(2*i,  0),colspan=2)
            hist_cmp(value[id,0,0],value[id,0,1], bins=bins)
            plt.ylabel(str(id) + ' t0')#plant_id[id])
            plt.subplot2grid((2*plot,3),(2*i+1,0),colspan=2)
            hist_cmp(value[id,1,0],value[id,1,1], bins=bins)
            plt.ylabel(str(id) + ' t1')#plant_id[id])
    else:
        for i,id in enumerate(best):
            plt.subplot2grid((2*plot,3),(2*i,  0),colspan=2)
            plt.plot(value[id,0,0])
            plt.plot(value[id,0,1])
            plt.subplot2grid((2*plot,3),(2*i+1,0),colspan=2)
            plt.plot(value[id,1,0])
            plt.plot(value[id,1,1])

    plt.subplot2grid((2*plot,3),(  0 ,2),rowspan=plot)
    plt.plot(corr[:,0])
    plt.gca().set_title('correlation t0')
    plt.subplot2grid((2*plot,3),(plot,2),rowspan=plot)
    plt.plot(corr[:,1])
    plt.gca().set_title('correlation t1')
    
    plt.suptitle(msr_name)
    
    
def correlation(a,b):
    """ return a correlation factor between 'a' and 'b' """
    def score(x): return (x-x.mean())/x.std() 
    return np.mean(score(a)*score(b))

@_aleanode()
def hist_cmp(a, b, bins=10, subplot=None):
    """ plot a histogram of 'a' vs histogram of 'b' """
    from matplotlib import pyplot as plt
    if subplot:
        plt.subplot(subplot)
    plt.cla()
    ca,ba = np.histogram(a, bins=bins)[:2]
    cb,bb = np.histogram(b, bins=ba)[:2]
    ca = ca/float(sum(ca))
    cb = cb/float(sum(cb))
    bw = bb[1]-bb[0]
    plt.bar(ba[:-1]+bw*.25,ca,width=bw*.5, color='g')
    plt.bar(bb[:-1]+bw*.5, cb,width=bw*.5, color='b')
    plt.xlim(ba[0],ba[-1])#ba.size/2])#


@_aleanode('property')
def get_axes_property(t,property_name, mask=None, order=None, per_plant=True, scale=None):
    """ Retrieve the property value for all axes in RootAxialTree t
    
    :Input:
        t: a RootAxialTree 
        property_name: the name of the property to retrieve
        mask:  a mask indicating which axes to retrieve value from
        order: restrict retrieval to axe with this order 
                - same as mask=t.axe.order==order
        per_plant: return a dictionary of axe values (list) for each plant
                   otherwise, return one list
    """
    if mask is None:
        mask = np.ones(len(t.axe.order), dtype=bool)
            
    if order is not None:
        mask &= (t.axe.order==order)
    
    ax_id = mask.nonzero()[0]
    
    value = t.axe[property_name][ax_id]
    if scale:
        value *= scale
    if per_plant:
        plant = t.axe.plant[ax_id]
        plant_id = np.unique(plant)
        plant_id = plant_id[plant_id>0]
        
        res = dict()
        for pid in plant_id:
            res[pid] = value[plant==pid]
        return res
    else:
        return value
        
@_aleanode('joined_list')        
def list_join(list_of_list):
    """ convert list_or_list to a list - just a practical openalea node """
    return [item for sublist in list_of_list for item in sublist]
    
def hull_area(points, plot=False):
    """
    points: Kx2 array of K points in 2-dimensions
    """
    from scipy.spatial import Delaunay
    
    points = np.asarray(points)
    tri = Delaunay(points)
    
    if plot: plot_delaunay(points,tri)
    
    return triangle_area(tri.vertices,points).sum()
    # loop over triangles, and compute the sum of their area
    #   ia, ib, ic = indices of corner points of the triangle
    #for ia, ib, ic in tri.vertices:
    #    def area_of_triangle(p1, p2, p3):
    #    '''calculate area of any triangle given co-ordinates of the corners'''
    #    return n.linalg.norm(n.cross((p2 - p1), (p3 - p1)))/2.    
    
def triangle_area(tri,points):
    u = points[tri[:,1]] - points[tri[:,0]]
    v = points[tri[:,2]] - points[tri[:,0]]
    a = abs(u[:,0]*v[:,1] - u[:,1]*v[:,0])/2
    
    return a
    
def plot_delaunay(points, delaunay=None, fig=42):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    
    if delaunay is None:
        from scipy.spatial import Delaunay
        delaunay = Delaunay(points)
    
    
    # plot it: the LineCollection is just a (maybe) faster way to plot lots of
    # lines at once
    area = triangle_area(delaunay.vertices,points)
    area = area/area.max()
    area = (area[:,None]*[0,1,0])+((1-area[:,None])*[0,0,1])
    poly = PolyCollection(points[delaunay.vertices], facecolors=area, edgecolors='r')
    plt.figure(fig)
    plt.clf()
    plt.title('Delaunay triangulation')
    plt.gca().add_collection(poly)
    plt.plot(points[:,0], points[:,1], 'o', hold=1)
    
    # -- the same stuff for the convex hull
    
    edges = set()
    edge_points = []
    
    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(points[ [i, j] ])
    
    for ia, ib in delaunay.convex_hull:
        add_edge(ia, ib)
    
    lines = LineCollection(edge_points, color='r')
    plt.title('Convex hull')
    plt.gca().add_collection(lines)
    plt.plot(points[:,0], points[:,1], 'o', hold=1)
    plt.xlim(points[:,0].min(), points[:,0].max())
    plt.ylim(points[:,1].min(), points[:,1].max())
    plt.show()    


def chi_square(ref_dist, test_dist):
    return np.sum((ref_dist-test_dist)**2/ref_dist)
    
