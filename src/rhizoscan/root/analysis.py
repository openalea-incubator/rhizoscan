"""
Provide some analysis tools for RootTree
 -- obsolete --
"""
import numpy as np


# RootAnalysisPproject stuff
# --------------------------
def total_root_length(p,prop, plant_num=1):
    """ per genotype.type """
    L = dict()  # length
    N = dict()  # number of plate
    
    for s in p.sequences:
        key = reduce(getattr,[s] + prop.split('.'))
        L.setdefault(key, np.zeros(len(s.plgraph)))
        N.setdefault(key, np.zeros(len(s.plgraph)))
        for i,g in enumerate(s.plgraph):
            L[key][i] += g.segment.length.sum()
            N[key][i] += 1
        s.plgraph.clear_buffer()
    
    for key in N.keys():
        L[key] /= N[key]*plant_num
    
    return L

def primary_root_length(p,prop, plant_num=1):
    """ per genotype.type """
    L = dict()  # length
    N = dict()  # number of plate
    
    for s in p.sequences:
        key = reduce(getattr,[s] + prop.split('.'))
        L.setdefault(key, np.zeros(len(s.plgraph)))
        N.setdefault(key, np.zeros(len(s.plgraph)))
        for i,t in enumerate(s.tree):
            L[key][i] += t.axe.length[t.axe.order==1].sum()
            N[key][i] += 1
        s.tree.clear_buffer()
    
    for key in N.keys():
        L[key] /= N[key]*plant_num
    
    return L
    
def plot_tt_length(p, prop, title=None, plant_num=1):
    if title is None: title='total root length w.r.t ' + prop
    
    L = total_root_length(p,prop=prop,plant_num=plant_num)
    
    import matplotlib.pyplot as plt
    plt.clf()
    for k,l in L.iteritems():
        plt.plot(l,label=k,linewidth=2)
    plt.legend(loc=2)
    plt.title(title)
    
def plot_a1_length(p, prop, title=None, plant_num=1):
    if title is None: title='primary root length w.r.t ' + prop
    
    L = primary_root_length(p,prop=prop,plant_num=plant_num)
    
    import matplotlib.pyplot as plt
    plt.clf()
    for k,l in L.iteritems():
        plt.plot(l,label=k,linewidth=2)
    plt.legend(loc=2)
    plt.title(title)
    

