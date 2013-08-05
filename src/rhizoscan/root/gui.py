from rhizoscan.workflow import node as _node # to declare workflow nodesfrom rhizoscan.image    import Image  as _Image

__icon__ = 'window.png'

@_node()
def plot_tree(tree, background='k', sc='order', fig=41):
    if hasattr(background,'filename'):
        background = background.filename
    
    if isinstance(background, basestring):
        background = _Image(background)
    
    if fig is not None: 
        from matplotlib import pyplot as plt
        plt.ion()
        plt.figure(fig)
    tree.plot(bg=background, sc=sc)
    

