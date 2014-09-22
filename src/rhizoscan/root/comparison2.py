"""
Tools to compare rhizoscan automated rsa extraction to references ones 
"""
from rhizoscan.misc.decorators import _property 

from rhizoscan.datastructure import Mapping as _Mapping
from rhizoscan.root.pipeline.dataset import Dataset as _Dataset

from rsml.continuous import discrete_to_continuous as _d2c

class TreeComparison(_Mapping):
    """ Comparison between 2 rsml-type MTG """
    def __init__(self, rsa, ref, metadata, key):
        self.__key__ = key
        self._rsa = rsa
        self._ref = ref
        self.metadata = metadata
        
    @_property
    def rsa(self):
        if not hasattr(self,'_crsa'):
            self._crsa = _d2c(self.get('_rsa').copy())
            self.temporary_attribute.add('_crsa')
        return self._crsa

    @_property
    def ref(self):
        if not hasattr(self,'_cref'):
            self._cref = _d2c(self.get('_ref').copy())
            self.temporary_attribute.add('_cref')
        return self._cref

    def match(self):
        """ match plants and axes using rsml.matching and store results """
        from rsml import matching
        ref = self.ref
        rsa = self.rsa
        
        pm,up_rsa,up_ref = matching.match_plants(ref,rsa)
        self.matched_plants = dict(((p1,p2),d) for p1,p2,d in pm)
        self.unmatch_plant_rsa = up_rsa
        self.unmatch_plant_ref = up_ref
        
        rm,ur_rsa,ur_ref = matching.match_roots(ref,rsa, pm)
        self.matched_roots = dict(((r1,r2),d) for r1,r2,d in rm)
        self.unmatch_root_rsa = ur_rsa
        self.unmatch_root_ref = ur_ref
        
    
    # measurements
    # ------------
    def _get_roots_per_plant(self, order=None):
        """ return ((ref_plant,rsa_plant),(ref_axes,rsa_axes)) """
        from rsml.misc import root_vertices, root_order
        
        if not hasattr(self,'matched_plants'):
            raise UnboundLocalError("plant are not matched: call 'match' first") 
        
        rsa_roots = self.rsa.components
        ref_roots = self.ref.components
        
        roots = dict(((pf,pa),(ref_roots(pf),rsa_roots(pa))) for pf,pa in self.matched_plants.keys())
        
        if order is not None:
            ref_order = root_order(self.ref)
            rsa_order = root_order(self.rsa)
            root_filter = lambda roots: ([r for r in roots[0] if ref_order[r]==order],
                                         [r for r in roots[1] if rsa_order[r]==order])
            
            roots = dict(zip(roots.keys(),map(root_filter,roots.values())))
        
        return roots

    def _get_roots(self, order=None):
        """ return list of amtched (ref_root,rsa_root) """

        if not hasattr(self,'matched_roots'):
            raise UnboundLocalError("roots are not matched: call 'match' first") 

        roots = self.matched_roots.keys()
        
        if order:
            from rsml.misc import root_order
            ref_order = root_order(self.ref)
            
            roots = [(refr,rsar) for refr,rsar in roots if ref_order[refr]==order]
            
        return roots
            

    def _total_root_length(self,order=None):
        """ Compute the total length of root axes, per plant
        
        Return a dict of ((ref_plant,rsa_plant),(ref_length,rsa_length))
        """
        from rsml.measurements import root_length
        roots = self._get_roots_per_plant(order=order)
        
        ref_length = lambda r: sum(self.ref_length(r).values())
        rsa_length = lambda r: sum(self.rsa_length(r).values())
        
        return dict((k,(ref_length(rf),rsa_length(ra))) 
                                for k,(rf,ra) in roots.iteritems())
    def _root_length(self,order=None):
        """ Compute the length of root axes
        
        Return a dict of ((ref_root,rsa_root),(ref_length,rsa_length))
        """
        roots = self._get_roots(order=order)
        
        ref_len = self.ref_length()
        rsa_len = self.rsa_length()
        
        return dict(((rf,ra),(ref_len[rf],rsa_len[ra])) for (rf,ra) in roots)
        
    def _root_branching_position(self):
        """ Compute the branching position of root axes
        
        Return a dict of ((ref_root,rsa_root),(ref_pos,rsa_pos))
        """
        from rsml.measurements import parent_position
        roots = self._get_roots(order=2)
        ref_pos = parent_position(self.ref)
        rsa_pos = parent_position(self.rsa)
        
        ref_len = self.ref_length()
        rsa_len = self.rsa_length()
        
        return dict(((rf,ra),(ref_pos[rf],rsa_pos[ra])) for (rf,ra) in roots)
        
    def ref_length(self,roots=None):
        """ return (and tmp store) roots length """
        if not hasattr(self,'_ref_length'):
            from rsml.measurements import root_length
            self._ref_length = root_length(self.ref)
            self.temporary_attribute.add('_ref_length')
            
        L = self._ref_length
        if roots: return dict((r,L[r]) for r in roots)
        else:     return L 
        
    def rsa_length(self,roots=None):
        """ return (and tmp store) roots length """
        if not hasattr(self,'_rsa_length'):
            from rsml.measurements import root_length
            self._rsa_length = root_length(self.rsa)
            self.temporary_attribute.add('_rsa_length')

        L = self._rsa_length
        if roots: return dict((r,L[r]) for r in roots)
        else:     return L 

    def compute_total_length(self):
        """ compute length measurements 
        'total_root_length', 'order1_length' and 'order2_length' """
        self.total_root_length   = self._total_root_length()
        self.total_order1_length = self._total_root_length(order=1)
        self.total_order2_length = self._total_root_length(order=2)
        
    def compute_order2(self):
        """ compute 'order2_number' """
        from rsml.measurements import parent_position
        
        roots = self._get_roots_per_plant(order=2)
        
        ref_len = self.ref_length()
        rsa_len = self.rsa_length()
        
        ref_pos = parent_position(self.ref)
        rsa_pos = parent_position(self.rsa)
                
        def branch(roots,g,L,ppos):
            return sorted((ppos[r],L[g.parent(r)],L[r]) for r in roots)
        def brsa(roots):
            return branch(roots,self.rsa,rsa_len,rsa_pos)
        def bref(roots):
            return branch(roots,self.ref,ref_len,ref_pos)
        
        b = dict((k,(bref(rf),brsa(ra))) for k,(rf,ra) in roots.iteritems())
        n = dict((k,(len(rf), len(ra)))  for k,(rf,ra) in roots.iteritems())
        self.branching = b
        self.order2_number = n
        
        self.order2_length = self._root_length(order=2)
        self.branching_position = self._root_branching_position()
        

    # string repr
    # -----------
    def __str__(self):
        return self.__class__.__name__ + '(key=' + self.__key__ + ')'
    def __repr__(self):
        s = 'rsa:'+str(self._rsa)+'\nref:'+str(self._ref)+'\n'
        s += _Mapping.__repr__(self)
        return s
        
class TreeComparisonSet(_Dataset):
    @classmethod
    def create(cls, ds, filename, rsa_name='rsa', ref_name='reference_rsa', verbose=False):
        """ Create a TreeComparisonSet from a dataset """
        tc = []
        for d in ds:
            if verbose:
                print d.__key__,
            d = d.copy().load()
            rsa = d.get(rsa_name)
            ref = d.get(ref_name)
            if rsa is not None and ref is not None:
                key = d.__key__
                meta= d.get('metadata')
                tc.append(TreeComparison(rsa=rsa,ref=ref,metadata=meta,key=key))
                if verbose:
                    print 'added'
            elif verbose:
                print 'rsml tree missing'
        
        tcs = cls(tc)
        if filename:
            tcs.set_file(filename)
            tcs.dump()
        
        return tcs
        
    def match(self, verbose=True):
        """ run `match` on all item, then save """
        for item in self:
            if verbose: print item
            item.match()
        self.dump()
        
    def _compute(self, fct_name, verbose,clear):
        """ call method `fct_name` on each item, then save """
        if verbose:
            from math import log10
            header = '%{}d/{}: '.format(int(log10(len(self)))+1,len(self))
            
        for i,item in enumerate(self):
            if verbose: print (header%(i+1)) + str(item)
            getattr(item,fct_name)()
            if clear: item.clear_temporary_attribute()
        self.dump()
            
        
    def compute_total_length(self, verbose=True, clear=True):
        """ run `compute_length` on all item, then save """
        self._compute(fct_name='compute_total_length', verbose=verbose, clear=clear)
        
    def compute_order2(self, verbose=True, clear=False):
        """ run `compute_length` on all item, then save """
        self._compute(fct_name='compute_order2', verbose=verbose, clear=clear)

    def _get_measurement(self, measurement='total_root_length', split=None):
        import numpy as np
        
        ##pid = []
        key = []
        ref = []
        rsa = []
        meta = []
        for tc in self:
            measure = getattr(tc,measurement)
            values = zip(*measure.values())
            ref.extend(values[0])
            rsa.extend(values[1])
            ##pid.extend([k[0] for k in measure.keys()])
            key.extend(zip([tc.__key__]*len(values[0]),measure.keys()))
            meta.extend([tc.metadata]*len(values[0]))

        if split:
            split = split.split('.')
            label = [reduce(getattr, [m]+split) for m in meta]
            label = np.array(label)
        else:
            label = None

        return key, np.array(ref),np.array(rsa), label
        
    def plot(self, measurement='total_root_length', T=None,
                   clf=True, clip=None, logscale=False,
                   key_filter=[], split=None, label_str=str):
        return self._plot(measurement=measurement, T=T,
                          clf=clf,clip=clip, logscale=logscale,
                          key_filter=key_filter, split=split, label_str=label_str)
        
    def hist(self, measurement='total_root_length', T=None, bins=10, 
                   clf=True, clip=None,
                   key_filter=[], split=None, label_str=str):
        return self._plot(measurement=measurement, T=T, hist=bins, 
                          clf=clf,clip=clip, logscale=False,
                          key_filter=key_filter, split=split, label_str=label_str)
        
    def _plot(self, hist=False, measurement='total_root_length', T=None,
                    clf=True, clip=None, logscale=False,
                    key_filter=[], split=None, label_str=str):
        import numpy as np
        from matplotlib import pyplot as plt
            
        key,ref,rsa,label = self._get_measurement(measurement,split=split)
        
        if T:
            key,ref,rsa,label = T(key,ref,rsa,label)
        
        if key_filter:
            ind = [i for i,k in enumerate(key) if key_filter(k)]
            ref = ref[ind]
            rsa = rsa[ind]
            if label is not None:
                label = label[ind]
            
        if clf:
            plt.clf()
            
        def cluster(values, label, names):
            return dict((name,values[label==name]) for name in names) 
            
        if hist:
            if clip: values = np.clip(rsa/ref,*clip)
            else:    values = rsa/ref
            
            if label is not None:
                label_names = np.unique(label)
                value_num = len(values)
                values = cluster(values,label,label_names).values()
                label_names = map(label_str,label_names)
                
                weights = [np.ones_like(v)/len(v) for v in values]
            else:
                label_names = None
                weights = np.ones_like(values)/len(values)
            plt.hist(values,hist,label=label_names, weights=weights)
            
        else:
            if clip: 
                ref = np.clip(ref,*clip)
                rsa = np.clip(rsa,*clip)
                bound = clip
            else:
                bound = [0,max(ref.max(),rsa.max())]
            plt.plot(bound,bound,'r')
            
            if label is None:
                plt.plot(ref,rsa,'.')
            else:
                label_names = np.unique(label)
                ref = cluster(ref,label,label_names).values()
                rsa = cluster(rsa,label,label_names).values()
                for vref,vrsa,name in zip(ref,rsa,label_names):
                    plt.plot(vref,vrsa,'.', label=label_str(name))

            
        ax = plt.gca()
        if label is not None:
            ax.legend(loc=0)
        if logscale:
            ax.set_yscale('symlog')
            ax.set_xscale('symlog')
            suffix = ' (log)'
        else:
            suffix = ''
        ax.set_xlabel('Reference'+suffix)
        ax.set_ylabel('Rhizoscan'+suffix)
        ax.set_title(measurement.replace('_',' '))
            
        return ax

def _lateral_len(branch,min_pos,max_pos):
    """ compute the sum of lateral length of roots branched in [min_pos,max_pos] """
    import numpy as np
    def llen(plant_branch):
        plant_branch = ((b[0]/b[1],b[2]) for b in plant_branch)
        return sum([l for b,l in plant_branch if min_pos<=b<max_pos])
    return np.array(map(llen,branch))

def lateral_total_len(min_pos,max_pos):
    def total_len(key,ref,rsa,label):
        ref = _lateral_len(ref,min_pos,max_pos)
        rsa = _lateral_len(rsa,min_pos,max_pos)
        return key,ref,rsa,label
    return total_len

def append_reference(ds, refds, name='reference_rsa', verbose=False, dry_run=False):
    """
    Append reference trees (as rsml mtg) from `refds` into `ds` Datasets
    
    :Inputs:
      ds 
        Dataset object. It is loaded then dumped after adding refds tree into
      refds
        Dataset object. Once loaded, it should contains a 'tree' attribute
      name
        Name of the attribute to use for storage into `ds`
      verbose
        If True, print a line for each element processed
      dry_run
        If True, don't dump `ds` elements
        
    :Outputs:
      - `ds`
      - the list of __key__ of refds element without tree
    """
    from rhizoscan.root.graph.mtg import tree_to_mtg
    
    # remove extra element of 'compared'
    def get_key(x):
        return x.__key__.replace('/','_')
    
    ds_dict  = dict([(get_key(d),d) for d in ds])
    to_merge = [(ds_dict.get(get_key(r),None),r) for r in refds]
    
    # find tree that are not found
    missing = []
    for d,r in to_merge:
        rtree = r.copy().load().get('tree', None)
        if rtree is not None:
            if verbose: print 'adding trees for', r.__key__
            d = d.copy().load()
            rrsa = tree_to_mtg(rtree)
            d.set(name, rrsa, store=not dry_run)
            if not dry_run:
                d.dump()
        else:
            if verbose: print ' *** unavailable trees for', r.__key__
            missing.append(r.__key__)
            
    return ds, missing


