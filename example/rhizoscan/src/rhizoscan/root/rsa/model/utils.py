"""
Utils functions
"""

def longest_axe(axes, group, selected=[]):
    """ select the longest axe in `axes`, one for each axe.'group'
    
    :Inputs:
      axes
        iterator of (axe_id,axe) where axe are BuilderAxe object
        
      group:
        name of the axe attribute that indicate axes group
    
      selected
        optional list of axe ids already selected
    
    return a set of selected axe ids
    """
    # dict of axes length, clustered by parent
    #   ax_len[group][axe] => axe.length
    ax_len = {}
    for axe_id,axe in axes:
        ax_len.setdefault(getattr(axe,group,None),{})[axe_id] = axe.length
        
    # select axes
    # -----------
    longest = set(selected)
    for group,axes in ax_len.iteritems(): 
        if not longest.intersection(axes.keys()):
            longest.add(max(axes, key=axes.get))
        
    return longest



def find_split_segment(path1,path2):
    """ return index of 1st element of `path1` that is not in `path2` """
    path_cmp = map(lambda x,y: x!=y, path1,path2)
    if True in path_cmp:
        return path_cmp.index(True)
    else:
        return 0

