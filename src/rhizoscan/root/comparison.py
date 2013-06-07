import numpy as np
import scipy.ndimage as nd


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
