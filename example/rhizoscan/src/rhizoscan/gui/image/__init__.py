"""
Modules containing user interfaces related to image.

- subimage:  select image area using matplotlib interface
- threshold: threshold an image using a slider  (requires PyQt4)

It requires matplotlib installed and working, otherwise import will fail.
Some functions won't be defined if PyQt4 is not installed 
"""

import numpy as _np
import scipy.ndimage as _nd
    
from matplotlib import pyplot as _plt
from matplotlib.figure import Figure as _Figure

from rhizoscan.workflow import node as _node # to declare workflow nodes
from rhizoscan.image.measurements import color_label
from rhizoscan.ndarray            import lookup as _lookup

@_node('AxesImage', OA_interface=dict(figure='IInt'))
def imshow(image, figure=None):
    """
    Simple call to matplotlib imshow
    """
    if figure:
        _plt.ion()
        _plt.figure(figure)
    return _plt.imshow(image)

def gci():
    """
    Return current displayed image (on current figure and axe) 
    """
    return _plt.gci().get_array().data

@_node('subimage', 'slices')
def subimage(img=None, verbose=False):
    """
    Select subrectangle of an image using matplotlib interface
    
    if 'img' is not None, display it. Otherwise use image currently displayed 
    """
    
    if img is None:
        img = gci()
    else:
        _plt.clf()
        _plt.imshow(img)
    
    if verbose:
        print 'Using zoom & span tool focus on subimage area, then press enter'

    done = False
    while not done: done = _plt.waitforbuttonpress()

    axe = _plt.gca()
    x = slice(*sorted(axe.get_xlim()))
    y = slice(*sorted(axe.get_ylim()))
    
    if verbose:
        print '   => Area selected x=[%d,%d] - y[%d,%d]' % (x.start,x.stop,y.start,y.stop)
    return img[y,x], (y,x)



@_node('axes_image')
def label_show(label, order='shuffle', cmap=None, start=1, negative=0, weight=None, clear=True):
    """
    Display a label (integer) image:
      - 0 value pixels are background, and displayed in black (using default config)
      - other labels are displayed with one of the following colors::
            white, red, green, blue, cyan, pink and yellow
            
    :Inputs:
        **order**
            how to choose the color order - either:
                  - shuffle: shuffle lavels id (>start)
                  - xmin:    order label mapping by the labels minimum x coordinates
                  - ymin:    order label mapping by the labels minimum y coordinates
                  - an integer: use directly the label id times this number
        **cmap**
            the color map - either:
                 - None (default colormap of 8 basic colors), 
                 - a colormap (Nx3 array of N colors),
                 - or a number (simply apply modulus, and return a grey color)
        **start** 
            loop into cmap starting at this label:
               it should be less than the number of colors in the color map
               if order is shuffle, labels below start are not shuffled
        **negative** 
            method to treat negative indices - a value to replace <0 labels by
    
        **weight** 
            an optional array of the same size as input label which gives a 
            float value in [0,1] to multiply displayed image by. 
            0 means black and 1 means full color of the colormap
      
        **clear**
            if True, clear the figure before displaying labeled image
    """
    from .. import ndarray as arr
    if clear: _plt.clf()
    
    toshow = color_label(label,order=order, cmap=cmap, start=start, negative=negative)
    
    if weight is not None:
        weight = weight.reshape(weight.shape + (1,)*(toshow.ndim-weight.ndim))
        toshow = toshow.astype(weight.dtype) * weight
    
    return _plt.imshow(toshow, interpolation='nearest')
    

from   matplotlib.blocking_input import BlockingInput as _BlockInput, BlockingMouseInput as _BlockMouse

class BlockPolyline(_BlockMouse):
    """
    class that block mouse (such as ginput) but allow input of polylines
    
    It allow input of one polyline of either fixed or arbitrary length (p=1)
    or of several polylines of fixed length (p=0). 
    
    It also avoid some bugs on some (my) version of matplotlib/backend
    """
    
    def __init__(self, fig, mouse_add=1, mouse_pop=3, mouse_stop=2, key_pause='p', **key_event):
        _BlockMouse.__init__(self,fig=fig,mouse_add=mouse_add,mouse_pop=mouse_pop,mouse_stop=mouse_stop)
        self._pause = False
        
        key_event[key_pause] = self.switch_pause
        self._key_event = key_event
        
    def key_event(self, *args):
        """ intercept key event """
        intercepted = False
        key = self.events[-1].key
        if self._key_event.has_key(key):
            self._key_event[key](self)
            self.events.pop()
        else:
            _BlockMouse.key_event(self,*args)
        
    def switch_pause(self, *args):
        self._pause = not self._pause
            
    def mouse_event(self, *args):
        """ intercept mouse event if pause """
        if not self._pause:
            event = self.events[-1]  ## bug: does not change 
            _plt.axes(event.inaxes)  ## subplot axes automatically
            _BlockMouse.mouse_event(self,*args)
        elif self.n>0:
            self.n += 1
                    
    def add_click(self,event):
        # add the coordinates of an event to the list of clicks, 
        # and draw markers and lines
        self.clicks.append((event.xdata,event.ydata))

        # make sure we don't mess with the axes zoom
        xlim = event.inaxes.get_xlim()
        ylim = event.inaxes.get_ylim()

        # plot the clicks
        self.marks.extend(event.inaxes.plot([event.xdata,], [event.ydata,], 'r+'))
        # plot the polyline segments
        if ((len(self.clicks)-1)%self.N)!=0:
            x,y = zip(*self.clicks[-2:])
            self.lines.extend(event.inaxes.plot(x,y,'-c',linewidth=2))

        # before we draw, make sure to reset the limits
        event.inaxes.set_xlim(xlim)
        event.inaxes.set_ylim(ylim)
        self.fig.canvas.draw()
        
    def pop_click(self,event,index=-1):
        # remove the coordinates of last event from the list of clicks,
        # and remove respective drawn markers and lines
        self.clicks.pop(index)
        
        # make sure we don't mess with the axes zoom
        xlim = event.inaxes.get_xlim()
        ylim = event.inaxes.get_ylim()

        if (len(self.lines)>0) and (((len(self.marks)-1)%self.N)!=0):
            line = self.lines.pop()
            line.remove()
        mark = self.marks.pop(index)
        mark.remove()

        # before we draw, make sure to reset the limits
        event.inaxes.set_xlim(xlim)
        event.inaxes.set_ylim(ylim)
        self.fig.canvas.draw()
           
    def cleanup(self,event=None):
        # clean the figure, when input is done
        if self.show_clicks:
            if event:
                # make sure we don't mess with the axes zoom
                xlim = event.inaxes.get_xlim()
                ylim = event.inaxes.get_ylim()
                
            for mark in self.marks:
                mark.remove()
            self.marks = []
            for line in self.lines:
                line.remove()
            self.lines = []

            if event:
                # before we draw, make sure to reset the limits
                event.inaxes.set_xlim(xlim)
                event.inaxes.set_ylim(ylim)
                
            self.fig.canvas.draw()
            # Call base class to remove callbacks
            _BlockInput.cleanup(self)
            
    def __call__(self, n=0, p=0, timeout=30, show_clicks=True):
        """
        Blocking call to retrieve the drawn polylines.
        n is the number of points per polylines, and p the number of polylines
            0 means no limit
            
        return the clicks as a list of (x,y) tuples (w/o polyline structure)
        """
        self.N     = n if n!=0 else _np.inf
        self.lines = []
        
        _BlockMouse.__call__(self,n=n*p,timeout=timeout,show_clicks=show_clicks)
        
        return self.clicks
        
@_node('points')
def ginput(n=1, timeout=30, fig=None, **key_event):
    """
    replace matplotlib.pyplot.ginput treating any keyboard event as 'enter' if
    necessary to avoid a bug on some (my) version/backend
    """
    fig  = _plt.gcf() if fig is None else fig if isinstance(fig,_Figure) else _plt.figure(fig)
    bm = BlockPolyline(fig=fig, **key_event)
    return bm(n=1, p=n, timeout=timeout, show_clicks=True)
    
@_node('polylines')
def linput(n=0, p=0, timeout=30, fig=None, **key_event):
    """
    Similar to matplotlib.pyplot.ginput but allow input of a polyline.
    
    :Inputs:
        - n is the number of points per polyline - 0 means no limit
        - p is the number of polylines - 0 means no limit:
            - if 'n' is 0, then it is meaningless. 
            - Otherwise, if p is not 0, then input automatically ends when 
              n*p points has been entered
    """
    # create BlockingInput obj, and override bugged methods
    fig  = _plt.gcf() if fig is None else fig if isinstance(fig,_Figure) else _plt.figure(fig)
    bm = BlockPolyline(fig=fig, **key_event)
    return bm(n=n, p=p, timeout=timeout, show_clicks=True)
    
@_node('X','Y','value')
def gvalue(n=1, image=None, verbose=False, **key_event):
    """
    Similar as ginput, but also return plotted image value on clicked pixels
    
    :Inputs:
        - n:
            number of input mouse click
        - image:
            image the values are taken in. If None, take the last image
            object of the current axe
        - verbose:
            if True, print pixel info at each click
        
    :Outputs:
        Three lists x, y, value (of pixel) for all clicks.
    """
    if image is None: image = gci()
        
    X = []
    Y = []
    V = []
    for i in range(n):
        p = ginput(1, **key_event)
        if len(p)==0: break
        x,y = p[0]
        if verbose: print 'value at (x:%.2f,y:%.2f) = %f' % (x,y,image[round(y),round(x)])
        X.append(x)
        Y.append(y)
        V.append(image[round(y),round(x)])
        
    return X,Y,V

def glvalue(step=1, interpolation=1, image=None, fig=None):
    """
    retrieve plotted image value on a line enter by user click.
    
    The return value are interpolated pixel value sampled at every 'step' number
    of pixels along the line. 
    
    The (spline) interpolation is done using the scipy map_coordinates function
    (through the ndarray.lookup function). It uses spline order equal to the
    given 'interpolation' parameter.
    Values should be between 0 and 5, where 0 is nearest and 1 is linear (default).
    See doc of scipy.ndimage.interpolation.map_coordinates for details
    
    By default, i.e. if image==None, the value are taken from the ploted image.
    Optionally, it can be taken from another one or more images given in the
    'image' argument, which can be an array or a tuple of array.
    
    'fig' is the figure in which the user selection is done (None means current). 
    
    Return 
        x,y,value:    if image is None or an array
        x,y,v1,...vn: if image is a tuple of arrays
    """
    line  = linput(n=2,p=1, fig=fig)
    if len(line)==0:
        return [],[],[]
        
    p1,p2 = _np.asarray(line)
    dist  = _np.sum((p1-p2)**2)**0.5
    na    = _np.newaxis
    x,y   = _np.mgrid[0:1+step/(dist+1):step/dist][na,:] * (p2-p1)[:,na] + p1[:,na]
    
    if image is None:                 image = (gci(),)
    elif not isinstance(image,tuple): image = (image,)
    
    return (x,y) + tuple([_lookup(img,(y,x),order=interpolation) for img in image])


from .. import getOpenFileName as _getOpenFileName
from rhizoscan.image import Image as _Image

@_node('image')
def load(filename="", dtype=None, color=None):## to finish !
    """
    Load image from file. If filename is not a file, provide an interface to select it
    In this case, filename is used as the opening folder of the interface 
    """
    import os
    if not os.path.isfile(filename): 
        filename = _getOpenFileName(filename)

    return _Image(filename,dtype=dtype,color=color) 


def plot_graph(g,x,y, color='b', directed=0):
    """
    plot the graph represented by the adjacency matrix g
    """
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection as lc
    
    plt.plot(x,y,'.'+color)
    
    p = _np.concatenate((x[:,None], y[:,None]),axis=1)
    L = p[_np.array(g.nonzero()).T]
    
    if directed>0:
        p2 = L[:,0,:] + (L[:,1]-L[:,0]) * directed
        plt.plot(p2[:,0],p2[:,1],'.'+color)
    
    lines = lc(L, color=color)
    plt.gca().add_collection(lines)

#  ----------------------------
#  functions that require PyQt4
#  ----------------------------
#   > threshold & ThresholdWindow
#try:
#    from PyQt4      import QtGui  as _QtGui
#    from matplotlib import cm     as _cm
#    from ..gui      import getQApplication as _getQApp
#    from ..gui      import MplCanvasWidget as _MplCanvasWidget
#    from .convert   import gray
#            
#    @_node('thresholded_image')
#    def threshold(image):
#        """ open a GUI to manually threshold a given image using a slider"""
#        gui  = ThresholdWindow(image)
#        gui.execute(_getQAppl())
#        return gui.get_threshold_image()
#    
#    class ThresholdWindow(_QtGui.QMainWindow):
#        """
#        A QMainWindow that show an image and a threshold slider.
#        
#        Constructor create the window, and the execute methods runs it.
#        """
#        
#        def __init__(self, image):
#            """
#            Create the QMainWindow displaying the given image to be thresholded.
#            The window is not visible
#            """
#            from PyQt4 import QtCore as _QtCore
#            
#            _QtGui.QMainWindow.__init__(self)
#            
#            self.image = gray(image)   # assert gray image or convert it to gray
#            self.threshold = -1   # -1 <=> no threshold
#            
#            # define UI
#            self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
#            self.setWindowTitle("Image thresholding")
#            
#            self.file_menu = _QtGui.QMenu('&File', self)
#            self.file_menu.addAction('&Quit', self.closeEvent, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
#            self.menuBar().addMenu(self.file_menu)
#    
#            self.main_widget = _QtGui.QWidget(self)
#            layoutV   = _QtGui.QVBoxLayout(self.main_widget)
#            layoutH   = _QtGui.QHBoxLayout()
#            
#            self.canvas = _MplCanvasWidget(self.main_widget)
#            self.axe    = self.canvas.figure.add_subplot(111)
#            self.axe.hold(False)
#            self.axe.imshow(image,cmap=_cm.gray,aspect='equal')
#            self.axe.set_position([0,0,1,1])
#            self.axe.set_axis_off()
#            
#            self.thSlider = _QtGui.QSlider(QtCore.Qt.Horizontal)
#            self.thSlider.setRange(-1,1000)
#            self.thSlider.setTickInterval(1)
#            self.thSlider.setValue(self.threshold)
#            self.thSlider.valueChanged.connect(self.updateImage)
#            self.thSlider.setTracking(True)
#    
#            self.okButton = _QtGui.QPushButton("OK")
#            
#            layoutV.addWidget(self.canvas)
#            layoutV.addLayout(layoutH)
#            layoutH.addWidget(self.thSlider)
#            layoutH.addWidget(self.okButton)
#    
#            self.main_widget.setFocus()
#            self.setCentralWidget(self.main_widget)
#            
#            
#        def execute(self, qapp):
#            # use qapp to make the current window modal
#            #  i.e. the calling process wait for the user to click the ok button
#            
#            # show UI
#            self.show()
#            self.raise_()
#            self.okButton.clicked.connect(qapp.quit)
#            
#            qapp.exec_()        
#           
#            self.close()
#    
#        def updateImage(self,event):
#            # update displayed image
#            self.threshold = self.thSlider.value()
#            self.axe.imshow(self.get_threshold_image()[0],cmap=_cm.gray,aspect='equal')
#            self.canvas.draw()
#        
#        def get_threshold_image(self):
#            """
#            compute the thresholded image from slider current value
#            and return the thresholded image and the threshold value
#            """
#            
#            t = self.image.max() * self.threshold/1000
#            if self.threshold>=0:  return self.image>=t, t
#            else:                  return self.image, t
#            
#except:
#    print 'image.gui: Could not import PyQt4 (or some related matplotlib modules)'

