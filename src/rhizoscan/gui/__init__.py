"""
Modules containing user interfaces and related stuff.

It requires PyQt and matplotlib installed and working.
Otherwise import will fail
"""
import matplotlib.pyplot as _plt  # check if matplotlib exist

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes

__icon__ = 'window.png'

# check if current backend is QtAgg
if _plt.get_backend()=='Qt4Agg':
    from PyQt4 import QtGui as _QtGui
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as _Canvas
    from matplotlib.figure import Figure as _Figure
    
    BACKEND = 'qt'
    
    qapp = None
    
    def getOpenFileName(directory=""):
        """ call Qt getOpenFileName user interface"""
        getQApplication()
        return str(_QtGui.QFileDialog.getOpenFileName(None,"*.*",directory))
    
    def getQApplication():
        """ get current QT4 application, or create one if none exist """
        global qapp
        qapp = _QtGui.QApplication.instance()
        if qapp is None:
            qapp = _QtGui.QApplication([' '])
        return qapp
    
    
    class MplCanvasWidget(_Canvas):
        """Matplotlib canvas that is a QWidget"""
        def __init__(self, parent=None, width=640, height=400):
            fig = _Figure(figsize=(width/100.0, height/100.0), dpi=100,facecolor='0.93')
    
            _Canvas.__init__(self, fig)
            self.setParent(parent)
    
            _Canvas.setSizePolicy(self, _QtGui.QSizePolicy.Expanding, _QtGui.QSizePolicy.Expanding)
            _Canvas.updateGeometry(self)
# ------------------------------------ #

# if backend is tkinter
elif _plt.get_backend()=='TkAgg':
    
    BACKEND = 'tk'
    
    from tkFileDialog import askopenfilename
    
    def getOpenFileName(directory=""):
        """ call Tkinter file dialog askopenfilename user interface"""
        
        return askopenfilename(initialdir=directory)
# ------------------------------------ #
    
else:
    BACKEND = None
    
    try:
        from tkFileDialog import askopenfilename
    
        def getOpenFileName(directory=""):
            """ call Tkinter file dialog askopenfilename user interface"""
            return askopenfilename(initialdir=directory) 
    except:
        print "*** GUI: Could not create file dialog function ***"
        
        def getOpenFileName(directory=""):
            """ default fake function """
            print "getOpenFileName is not working"
            return None
            
            
# declare getOpenFileName as an aleanode
_aleanode("file_path")(getOpenFileName)
