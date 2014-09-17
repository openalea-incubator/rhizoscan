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
        rsa = self.rsa
        ref = self.ref
        
        pm,up_rsa,up_ref = matching.match_plants(rsa,ref)
        self.matched_plants = dict(((p1,p2),d) for p1,p2,d in pm)
        self.unmatch_plant_rsa = up_rsa
        self.unmatch_plant_ref = up_ref
        
        rm,ur_rsa,ur_ref = matching.match_roots(rsa,ref, pm)
        self.matched_roots = dict(((r1,r2),d) for r1,r2,d in rm)
        self.unmatch_root_rsa = ur_rsa
        self.unmatch_root_ref = ur_ref
        
    def __str__(self):
        return self.__class__.__name__ + '(key=' + self.__key__ + ')'
        
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


