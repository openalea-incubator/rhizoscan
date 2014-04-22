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

    def order(self, axe):
        """ return the order of `axe` """
        o = self.mtg.property('order').get(axe,None)
        if o is None:
            p = self.mtg.parent(axe)
            if p: o =  self.order(p)+1
        
        return o if o else 0
        
    def order_color(self, segment):
        """ return the color associated to `segment` axe order """ 
        return self.order(self.get_axe(segment))
        
    def _add_branching(self, segment, position):
        """ add a branching vertex (i.e. edge_type '+') to `segment` 

        And add an axe accordingly, with order = parent order +1
        
        return 
          - the id of the created segment
          - the set of updated segment
        """
        new_v, up = _PASModel.add_branching(self, segment=segment, position=position)
        paxe = self.get_axe(segment)
        saxe = self.get_axe(new_v)
        print 'RootMode add branch', paxe, saxe
        self.mtg.property('order')[saxe] = self.order(paxe)+1
        return new_v, up

