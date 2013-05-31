################################################################################
# Widgets
import sys
from openalea.visualea.node_widget import NodeWidget       

from openalea.vpltk.qt import QtCore, QtGui

class ListSelector(NodeWidget, QtGui.QDialog):
    """
    This Widget allows to select some elements in a list
    """
    def __init__(self, node, parent):
        QtGui.QDialog.__init__(self, parent)
        NodeWidget.__init__(self, node)

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.btGroup = QtGui.QButtonGroup()
        self.btGroup.setExclusive(True)

        self.widgets = []
        self.notify(node, ("input_modified", 0))

    def notify(self, sender, event):
        # Notification sent by node 
        if(event[0] != "input_modified"): return
        self.create_buttons()

    def reactToClick(self, index):
        if self.widgets[index].isChecked():
            self.node.func.selected = index

    def create_buttons(self):
        """ Remove old buttons and add new ones.
        """
        inlist = self.node.func.get_selection_list()
        select = self.node.func.get_selected_flags()
        layout = self.layout()
        
        # add missing button
        for i in range(len(inlist)-len(self.widgets)):
            button = QtGui.QRadioButton("NEW") ##QtGui.QCheckBox(elt_name) ##
            self.btGroup.addButton(button)
            self.widgets.append(button)
            self.connect(button, QtCore.SIGNAL("clicked()"), lambda index=i: self.reactToClick(index))
            layout.addWidget(button)
            
        # hide unnecessary button
        for i in range(len(self.widgets)-len(inlist)):
            self.widgets[-i-1].setHidden(True)
            
        # sert name, visible and state
        for i, name in enumerate(inlist):
            bt = self.widgets[i]
            bt.setHidden(False)
            bt.setText(str(name))
            bt.setChecked(select[i])
            
        return
  
        for w in self.widgets:
            layout.removeWidget(w)
        
        layout = self.layout()
        layout.addWidget(self.btGroup)

        self.widgets = []

        for i, elt in enumerate(self.node.func.get_selection_list()):
            elt_name = str(elt)
            button = QtGui.QRadioButton(elt_name) ##QtGui.QCheckBox(elt_name) ##
            self.btGroup.addButton(button)
            button.setChecked(self.node.func.selected[i])

            self.connect(button, QtCore.SIGNAL("clicked()"), lambda index=i: self.reactToClick(index))
            #layout.addWidget(button)
            #self.widgets.append(button)


            








