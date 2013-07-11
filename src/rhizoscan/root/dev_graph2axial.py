"""
Development of algorithms to convert RootGraph to a RootAxialGraph

It uses mainly graph.py and dev_graph.py
"""
import numpy as _np


def path_to_axes(segment, path, top_order, axe=['longest',.1]):
    """
    From a set of covering path, return a set of root axes with order 1 axe 
    detected
    
    *** warning: input `path` is changed in-place ***
    """
    pLength = _np.vectorize(lambda slist:segment.length[slist].sum())(path)
    pPlant  = segment.seed[[p[-1] if len(p) else 0 for p in path]]
    
    max_order = 2
    
    # find axe order of path
    # ----------------------
    pOrder = [max_order]*len(pPlant)
    if axe[0]=='longest':
        puid = _np.unique(pPlant)
        if puid[0]==0: puid=puid[1:]
        for plant in puid:
            main_axe = _np.argmax(pLength*(pPlant==plant))
            pOrder[main_axe] = 1

    # convert path to axes:
    # ---------------------
    # find axe order of segment: order of the axe passing with lowest order 
    sOrder = _np.ones(segment.seed.size,dtype='uint8')*max_order
    for i,p in enumerate(path):
        if i==0: continue
        sOrder[p] = _np.minimum(sOrder[p],pOrder[i])
        
    # crop start of path that are also part of an axe with lower order
    for i,p in enumerate(path):
        if i==0: continue
        start = _np.argmax(sOrder[p]<pOrder[i]) # find the 1st element that has correct order
        path[i] = p[start:]


    return pLength, pPlant, pOrder, sOrder, path
