""" some user interface tools related to root graph

require: matplotlib
"""

import numpy as _np

from matplotlib import pyplot as plt

from rhizoscan.workflow import node as _node # to declare workflow nodes
from rhizoscan.image    import Image  as _Image

__icon__ = 'window.png'

@_node()
def plot_tree(tree, background='k', ac=None, sc=None, fig=41):
    if hasattr(background,'filename'):
        background = background.filename
    
    #if isinstance(background, basestring):
    #    background = _Image(background)
    
    if fig is not None: 
        plt.ion()
        plt.figure(fig)
    tree.plot(bg=background, ac=ac, sc=sc)
    

def segment_id(graph):
    """
    return the id of the closest segment to one user click
    `graph` should already be ploted
    """
    env = _segdist_env(graph)
    p = plt.ginput(1)
    d = _segment_distance(p[0],env)
        
    return d.argmin()
    
def segment_ids(graph, verbose=True):
    """ return the id of the closest segments to user click

    if verbose, print out the id at each click
    
    """
    env = _segdist_env(graph)
    ids = []
    
    graph.plot(bg='k')
    p = plt.ginput(1)
    while len(p):
        d = _segment_distance(p[0],env)
        axis = plt.axis()
        graph.plot(bg='k', sc=1+(d==d.min()))
        ids.append(d.argmin())
        if verbose: print 'segment id:', ids[-1]
        plt.axis(axis)
        p = plt.ginput(1)
        
    return ids
    
def _segdist_env(rgraph):
    """ Return precomputed data required by `_segment_distance` """
    segment = rgraph.segment
    
    pos   = segment.node_list.position[:,segment.node]  # shape (xy,S,node12)
    n1    = pos[:,:,0]                # position of 1st segment node
    n2    = pos[:,:,1]                # position of 2nd segment node
    sdir  = n2-n1                     # (unit) vector from n1 to n2
    lsl   = (sdir**2).sum(axis=0)**.5 # distance between n1 and n2
    lsl   = _np.maximum(lsl,2**-5)
    sdir /= lsl
    
    return dict(n1=n1, n2=n2, sdir=sdir, lsl=lsl)
    
def _segment_distance(p, env):
    """
    x,y = p
    env: precomputed variable constructed by `_segdist_env`
    
    return the distance from p to all segments
    """
    n1   = env['n1']
    n2   = env['n2']
    sdir = env['sdir']
    lsl  = env['lsl']
    
    norm = lambda x: (x**2).sum(axis=0)**.5
    
    p = _np.array(p)[:,None]  # shape (2,1)
    
    # distance from n1 to the projection of p on sdir 
    on_edge = ((p-n1)*sdir).sum(axis=0)
    
    # (orthogonal) distance from p to sdir
    d = norm(n1 + on_edge[None]*sdir - p)
    
    mask = on_edge<0
    d[mask] = norm(n1[:,mask]-p)
    mask = on_edge>lsl
    d[mask] = norm(n2[:,mask]-p)

    return d
    

