"""
Package implementing the RootEditor

The RootEditor, based on TreeEditor, is a graphical user interface for the 
rhizoscan pipeline. It allows:
 - to process a dataset (i.e. a list of root image)
 - to configure and view each pipeline step
 - to view and manually edit the extracted root architecture
"""
from openalea.vpltk.qt import QtCore, QtGui

from treeeditor.editor import TreeEditor       as _TreeEditor
from treeeditor.editor import TreeEditorWidget as _TreeEditorWidget
from treeeditor.tree   import TreePresenter    as _TreePresenter

from rhizoscan.root.editor.image import ImagePresenter as _ImagePresenter
from rhizoscan.root.editor.tree  import RootPresenter  as _RootPresenter
##from rhizoscan.root.editor.tree  import RootModel      as _RootModel

from rhizoscan.root.pipeline.dataset import Dataset      as _Dataset
from rhizoscan.root.pipeline.dataset import make_dataset as _make_dataset


from rhizoscan.root.pipeline.arabidopsis import pipeline as _arabido_pl


class RootEditorWidget(_TreeEditorWidget):
    def __init__(self):
        _TreeEditorWidget.__init__(self, parent=None, tree=None, background='default',theme=None)
        
        image_viewer = _ImagePresenter(editor=self,theme=self.theme)
        pmask_viewer = _ImagePresenter(editor=self,theme=self.theme)
        rmask_viewer = _ImagePresenter(editor=self,theme=self.theme)
        seed_viewer  = _ImagePresenter(editor=self,theme=self.theme)
        tree_viewer  = _RootPresenter( editor=self,theme=self.theme)
        self.attach_viewable('image_viewer', image_viewer)
        self.attach_viewable('pmask_viewer', pmask_viewer)
        self.attach_viewable('rmask_viewer', rmask_viewer)
        self.attach_viewable('seed_viewer',  seed_viewer)
        self.attach_viewable('tree_viewer',  tree_viewer)

        self.dataset = _Dataset()
        self.edited_item = None

        self.add_file_action('Load dataset', self.load_dataset, dialog='open', keys=['Ctrl+O'])
        self.add_file_action('Import image', self.import_image, dialog='open', keys=['Ctrl+I'])
        self.add_file_action('Save mtg',     self.save_mtg,                    keys=['Ctrl+S'])
        self.background._file_action = []  # remove load image/point bg from file menu

        self.add_view_action('display image',      self.image_viewer.show, keys=['Shift+1'])
        self.add_view_action('display plate mask', self.pmask_viewer.show, keys=['Shift+2'])
        self.add_view_action('display root mask',  self.rmask_viewer.show, keys=['Shift+3'])
        self.add_view_action('display seed map',   self.seed_viewer.show,  keys=['Shift+4'])


    def run_pipeline(self):
        """ call the rhizoscan pipeline on current image """
        if self.edited_item is None:
            self.show_message("No item selected")
            return
            
        item = self.edited_item
        _arabido_pl.run(namespace=item, compute='all', stored_data=['mtg'], verbose=True)
        item.dump()
        self._update_tree()
        

    def load_dataset(self, filename):
        """ load (replace) the dataset managed by the RootEditor """
        try:
            ds, inv, out = _make_dataset(filename)
            self.dataset = ds
            self.dataset.key_sort()
        except:
            self.show_message('Error loading dataset file: '+filename)
            return
            
        self.dataset_cb.activated.connect(self.set_dataset_item)
        self.dataset_cb.clear()
        self.dataset_cb.addItems(ds.keys())

        self.set_dataset_item(0)
            
    def set_dataset_item(self, index):
        """ change the current view/edited dataset item """
        self.edited_item = self.dataset[index].copy().load()
        item = self.edited_item
        
        # set image
        self.image_viewer.set_image(item.filename)
        self._update_tree()
        
    def save_mtg(self):
        """ Save the current dataset item mtg attribute (using its dump method) """
        item = self.edited_item
        if not item.has_key('mtg'):
            self.show_message("*** Error: No mtg to save ***")
        else:
            item.set('mtg',item.mtg,True)
            item.dump()
            self.show_message("mtg saved")

    def _update_tree(self):
        item = self.edited_item
        # set tree, if exist
        if item.has_key('mtg'):
            self.tree_viewer.set_model(item.mtg)
            self.set_edited_presenter('tree_viewer')
        else:
            self.show_message("dataset item has not 'mtg' attribute")

    def import_image(self, filename):
        raise NotImplementedError("")

    
    def get_toolbar(self):
        """ create the toobar for this editor """
        tb = QtGui.QToolBar(self)
        
        load = self._create_action(description='Load dataset', function=self.load_dataset, dialog='open')
        run  = self._create_action(description='run rhizoscan', function=self.run_pipeline, keys=['Ctrl+R'])
        cbox   = QtGui.QComboBox(parent=tb)
        self.dataset_cb = cbox
        
        tb.addAction(load)
        tb.addWidget(cbox)
        tb.addAction(run)

        return tb


class RootEditor(_TreeEditor):
    def __init__(self):
        editor = RootEditorWidget()
        _TreeEditor.__init__(self, editor)
        self.addToolBar(editor.get_toolbar())
        self.show()


def main():
    """ editor as an executable program """
    ## sys.argv to load dataset file?
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option("-d","--dataset", dest='dataset',help="*.ini dataset file to load")
    parser.add_option("-i","--image",   dest='image',  help="image file to import")
    
    options, args = parser.parse_args()
    
    qapp = QtGui.QApplication([])
    viewer = RootEditor()
    viewer.setWindowTitle("RootEditor")

    if options.dataset: viewer.editor.load_dataset(options.dataset)
    if options.image:   viewer.editor.import_image(options.image)
        
    qapp.exec_()

