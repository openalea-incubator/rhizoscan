from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes
from rhizoscan.image    import Image  as _Image

__icon__ = 'window.png'

@_aleanode()
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
    

