"""
Manage image and mask edition and visualisation
"""
import PyQGLViewer as _qgl

from treeeditor.mvp   import Presenter as _Presenter
from treeeditor.mvp   import Model     as _Model
from treeeditor.image import ImageView as _ImageView

from rhizoscan.image import Image as _Image

class ImagePresenter(_Presenter):
    """ General Presenter for ImageView """
    def __init__(self, image=None, editor=None, theme=None):
        """ create an ImagePresenter """
        _Presenter.__init__(self, editor=editor, theme=theme)
        self.set_image(image)
    
    def set_image(self, image):
        """ set the image for this presenter
        
        `image` can be:
          - an ImageModel
          - an array to view&edit
          - a filename
        """
        if isinstance(image,basestring):
            image = _Image(image)
            
        if isinstance(image, _Model):
            image_model = image
            image_view  = _ImageView(image=image.image, presenter=self)
        else:
            image_model = ImageModel(image=image, presenter=self)
            image_view  = _ImageView(image=image, presenter=self)
            
        self.attach_viewable('image_view',image_view)
        self.image_model = image_model

        # update editor
        editor = self.get_editor()
        if editor and image is not None:
            editor.set_camera('2D')
            self.__gl_init__() ## required to have img_width/height below
            ## for the image to exactly fit the screen height
            w,h = self.image_view.img_width, self.image_view.img_height
            editor.camera().fitSphere(_qgl.Vec(w/2,h/2,0),h/2)
            editor.updateGL()



class ImageModel(_Model):
    """ General edition model for images """
    def __init__(self, image=None, presenter=None):
        """ create an ImageModel """
        _Model.__init__(self, presenter=presenter)
        self.image = image
