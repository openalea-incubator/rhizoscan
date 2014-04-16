"""
Package that implement mtg interaction for RootEditor
"""

from treeeditor.tree       import TreePresenter as _TreePresenter
from treeeditor.tree.model import PASModel as _PASModel


class RootPresenter(_TreePresenter):
    def __init__(self, tree=None, theme=None, editor=None):
        self.model = RootModel()
        _TreePresenter.__init__(self,tree=tree, theme=theme, editor=editor)
        self._file_actions = []
        

class RootModel(_PASModel):
    """ manage edition of a RootMTG """
    def __init__(self, presenter=None, mtg=None, position='position', radius='radius'):
        _PASModel.__init__(self, presenter=presenter, mtg=mtg, 
                                 position=position, radius=radius)
        
        self.maxbackup = 10
        
        self._color_fct.append(('axe order',self.order_color))
        self.next_color('axe order')

        
    def order_color(self, segment):
        """ return the color associated to `segment` axe order """ 
        axe = self.get_axe(segment)
        return self.mtg.property('order').get(axe,0)
        

