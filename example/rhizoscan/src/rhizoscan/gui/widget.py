"""
Widget tools for Visualea
"""


from rhizoscan.workflow import node as _node # to declare workflow nodes
from list_selector_widget import ListSelector as widget_ListSelect

@_node(widgetclass='widget_ListSelect') 
class item_selector(object):
    """
    Node function that manage an item selector widget
    """
    def __init__(self):
        self._wrapper = None 
        self._in_list = []
        self.selected = []
        
    def set_wrapper_node(self, node):
        self._wrapper = node
        
    def __call__(self, input_list=[], selected=0):
        # the node's input has been changed
        # or the out_list has been modified by the widget (?)
        if len(input_list):
            self._in_list = input_list
            self.selected = selected
            return self._in_list[self.selected]
            
        return None
        
    def get_selection_list(self):
        return self._in_list
        
    @property
    def selected(self):
        return self._selected
    @selected.setter
    def selected(self, indice):
        self._selected = indice
        if self._wrapper:
            self._wrapper.set_input('selected', val=indice)
            self._wrapper.set_caption(repr(self._in_list[indice]))
        
    def get_selected_flags(self):
        flags = [False]*len(self.get_selection_list())
        flags[self.selected] = True
        return flags

