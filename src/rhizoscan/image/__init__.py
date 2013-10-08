"""
Tools related to image data and image-file I/O
"""

# icon of openalea package
__icon__ = 'mandelbrot.png'


import numpy as _np
import scipy as _sp
import scipy.ndimage as _nd

from rhizoscan.ndarray import add_dim as _add_dim
from rhizoscan.ndarray import aslice  as _aslice

from rhizoscan.workflow import node as _node # to declare workflow nodes

from rhizoscan.datastructure import Data as _Data
from rhizoscan.storage       import StorageEntry as _StorageEntry
from rhizoscan.storage       import create_entry as _create_entry
from rhizoscan.datastructure import Sequence as _Sequence   

from rhizoscan.tool   import static_or_instance_method as _static_or_instance
from rhizoscan.tool   import _property
from rhizoscan.tool.path import assert_directory as _assert_directory

class Image(_Data, _np.ndarray):
    """
    ndarray subclass designed to represent images
    
    ##TODO: doc
    """
    def __new__(cls, array_or_file, color=None, dtype=None, scale='dtype', from_color=None, info={}):
        # read file if input is a file name
        # & cast input array to be our Image class
        # for now
        #   color can be None (no change) or 'gray'            
        #   dtype can be None (no change), any dtype, or 'f*' (assert_float)
        #   scale used by imconvert => keept !!
        
        # create Image instance
        # ---------------------                                       
        if isinstance(array_or_file,(basestring,_StorageEntry)):
            obj = PILSerializer.load_image(array_or_file, 
                        color=color, dtype=dtype, scale=scale, from_color=from_color)
            
            ## how load file info in new storage entry system?
            #obj = _nd.imread(array_or_file).view(cls)
            #obj.set_storage_entry(array_or_file)
            #
            ## load image info and timestamp
            #from PIL import Image
            #from os.path import getmtime
            #obj.info = Image.open(array_or_file).info
            #obj.info['timestamp'] = getmtime(array_or_file)
            #obj.info.update(info)
        else:
            obj = _np.asanyarray(array_or_file).view(cls)
            obj.set_storage_entry('')
            
            # conversion
            if obj.dtype.kind not in ('b','u','i','f'):
                print "\033[31m*** Image is not numeric (and cannot be converted) ***\033[30m"
                obj.color = 'unknown color space'
            else:
                obj = imconvert(obj,color=color,dtype=dtype, from_color=from_color, scale=scale)
                
        obj.info = info
        return obj
        
    def __array_finalize__(self, obj):
        if hasattr(obj,'__dict__'):
            self.__dict__.update(obj.__dict__)
            
        self.color = detect_color_space(self)
        self.__dict__.setdefault('info',        {}) ## still there?
        
    def __init__(self, array_or_file, color=None, dtype=None, from_color=None, scale='dtype', info={}):
        """ create an image objectr form numpy array or file name  """
        ##todo: finish doc of Image
        # required to use __new__ (?) 
        pass

    def normalize(self, min=0, max=1):
        """
        map pixels value from its extremum to (min, max)  - in place
        
        min and max are the value to map to 
        if min (max) is None, use the image min (max)
        
        return itself
        """
        min = float(min if min is not None else _np.nanmin(self))
        max = float(max if max is not None else _np.nanmax(self))
        
        self[:] = (self[:] - min)*(max/(max-min))  
        
        return self
        
    def set_serializer(self, pil_format, pil_mode=None, pil_param=None,
                       ser_color=None, ser_dtype='auto', ser_scale='dtype',
                       extension=None):
        """
        Define the serializer of this image.
        
        If can be call either as:
          (1) img.set_serializer(serializer_object)
          (2) img.set_serializer([PILSerializer parameters])
        
        For (1), `serializer` should implement `dump` and `load` functions
        For (2), this method takes the same parameters as the PILSerializer 
        constructor, but for `img_color` and `img_dtype` which are taken from
        the Image object `img`.
        
        See also: `PILSerializer`
        """
        if all(map(hasattr,[pil_format]*2, ['dump','load'])):
            s = pil_format 
        else:
            s = PILSerializer(img_color=self.color, img_dtype=self.dtype,
                              pil_format=pil_format, pil_mode=pil_mode, pil_param=pil_param,
                              ser_color=ser_color, ser_dtype=ser_dtype, ser_scale=ser_scale,
                              extension=extension)
        self.__serializer__ = s
        
    @_static_or_instance
    def save(image, filename=None, color=None, dtype='auto', scale='dtype',
                    pil_format=None, pil_mode=None, **pil_param):
        """
        Save the image to file 'filename' using PIL.Image
        
        This method can be called as
          1a a static   method:  Image.save(some_array,        filename,      ...)
          1b a static   method:  Image.save(some_Image_object, filename=None, ...)
          2. a instance method:  some_Image_object.save(       filename=None, ...)
        
        :Inputs:
          - image:
              the image to save (1.) or self (2.)
        
          - filename:
              the name of the image file to save. 
                - case 1a
                    filename is mandatory
                - case 1b and 2: 
                    if filename is None, use its Data file attribute.
                    otherwise save it to file filename and change *in place* the
                    Data file attribute of input image
                  
          - color, dtype, scale:
              convert image to this color & dtype after scaling. 
              See the 'important notes' below and the imconvert documentation 
              possible scale: 'normalize','dtype', 'view', or any numeric value  
        
          - pil_format:
              image format to use. If None, guess it from the file extension
          - pil_mode:
              can be passed to PIL.Image.fromarray to enforce conversion mode. 
              None means automatic selection (see below)
          - **pil_param:
              Additional parameter to pass to PIL save function. Depends on 
              pil_format used. See PIL documentation.
              If it contains a `pnginfo` it should be a dictionary, for which 
              all key and value are strings, otherwise are converted using `str`

        :Outputs:
            return an empty Image object that can be used to load the image file:
            empty_image.load()
           
            The loaded Image will have suitable attribute to load the same image
            data as contained by the calling image (up to saving loss).
              - color and from_color are switched
              - dtype is the calling image dtype
              - scale is 
                  * the same as input scale if it is 'dtype' or 'view'
                  * 1./scale if input scale is numeric
                  * 'dtype'  if input scale is 'normalize' (no reverse)
          
        :Important Notes:
           To be saved, the image has to fit one of the PIL mode (see below), then
           it can save to different image file format depending on the mode and 
           installed library.
           the pil_format, pil_mode and pil_param are directly passed to the save
           method of PIL Image. But usually default (None) value are suitable.
           See the PIL Image save method documentation 
           
           However, if the image does not meet PIL requirement in color (ie. shape) 
           and dtype, and if the automatic conversion (see dtype is 'auto' below)
           is not suitable, the color, dtype and scale arguments can be used to 
           convert the image (using the imconvert function) before saving.
           *** See the imconvert documentation ***
         
        :Automatic dtype fitting:
           if dtype is 'auto', apply conversion following the rules
             if color is not gray:                      convert to uint8
             if image.dtype is bool or uint8:           keep dtype
             if filename end by '.tif' or '.tiff',
                or if pil_format is 'TIFF':             convert to float32
             else:                                      convert to uint8
        
        :Automatic conversion to pil mode: 
           PIL provide an automatic conversion method fromarray, which convert 
           data following their dimension and dtype (see PIL.Image._fromarray_typemap)
           Here is a list relating Image color value.
           [image.shape - image.color - image.dtype => pil mode]
           
           (.,.)   gray - bool         => 1     - 1-bit pixels, black and white
           (.,.)   gray - uint8        => L     - 8-bit pixels, black and white
           (.,.)   gray - int8 to 32   => I     - 32-bit signed integer pixels
           (.,.)   gray - float32 & 64 => F     - 32-bit floating point pixels
           (.,.,3) rgb  - uint8        => RGB   - 3x8-bit pixels, true colour
           (.,.,4) rgba - uint8        => RGBA  - 4x8-bit pixels, true colour with transparency mask
             
           PIL modes:        http://www.pythonware.com/library/pil/handbook/concepts.htm
           PIL file formats: http://www.pythonware.com/library/pil/handbook/index.htm#appendixes
        """
        ##todo:  provide scaling factor that is set back at loading
        ##not implemented (.,.,3) yuv  - uint8        => Ycrcb - 4x8-bit pixels, colour separation
        ##check: save without loss ?
        from PIL.Image import fromarray
        
        # manage input image
        # ------------------
        if _Data.has_IO_API(image):
            if filename is not None:
                image.set_storage_entry(filename)
            elif image.get_storage_entry() is None:
                raise TypeError("filename shoud be provided: input Image object has unset Data file")
        elif filename is None:
            raise TypeError("filename should be provided")
            
        if not isinstance(image,Image):
            image = Image(image)
            image.set_storage_entry(filename)
            
        entry = image.get_storage_entry()
        
        # convert image
        # -------------
        if dtype=='auto':
            if image.color!='gray':             dtype = 'uint8'
            elif image.dtype in (bool,'uint8'): dtype = None
            elif pil_format=='TIFF' or \
                 entry.url.lower().endswith(('.tif','.tiff')):
                                                dtype = 'float32'
            else:                               dtype = 'uint8'
            
        img = imconvert(image, color=color, dtype=dtype, scale=scale)
        img.set_storage_entry(entry)
        
        # save image
        # ----------
        ## file/dir preparation to be transfered to Data (_open, _close ?)
        _assert_directory(entry.url) ## should be StorageEntry work? 
        
        # check for pnginfo in pil_param
        if pil_param.has_key('pnginfo'):
            from PIL.PngImagePlugin import PngInfo
            astr = lambda x: x if isinstance(x,basestring) else str(x)
            info = PngInfo()
            for key,value in pil_param['pnginfo'].iteritems():
                info.add_text(astr(key), astr(value)) 
            pil_param['pnginfo'] = info
            
        img = fromarray(img,mode=pil_mode)
        img.save(entry.url, format=pil_format, **pil_param)
        
        #loader = Image([])
        loader = image.loader()
        loader.from_color = color
        loader.set_storage_entry(entry)
        if scale=='normalize':                 loader.scale = 'dtype'
        elif not isinstance(scale,basestring): loader.scale = 1/float(scale)
        
        return loader
        
    @_static_or_instance
    def load(self):
        """
        Method to load image using the empty image returned by Image.loader()
        
        - Use Image constructor to load an image file -
        
        The loaded  image  is *returned*
        the calling object is *unchanged*
        
        :Note:
            This method load the image from file given by the loader Data file 
            attribute and apply conversion to the same color and dtype.
        """
        return Image(self.get_storage_entry(),color=self.color, dtype=self.dtype, scale=self.scale)
        
    def loader(self):
        """
        Return an empty Image which allow to load the image using load()
        
        *** if this Image has no associated file, it won't be able to load ***
        """
        loader = Image([])
        loader.dtype  = self.dtype
        loader.color  = self.color
        loader.from_color = self.from_color
        loader.set_storage_entry(self.get_storage_entry())
        return loader
        
    ## __store/restore__ : return serialized stream ?
    #def __store__(self):
    #def __restore__(self):

    # for pickling
    def __setstate__(self,state):
        _np.ndarray.__setstate__(self,state[0])
        self.__dict__.update(state[1])
        
    def __reduce__(self):
        object_state = list(_np.ndarray.__reduce__(self))
        object_state[2] = (object_state[2],self.__dict__)
        return tuple(object_state)
    
    # for printing
    def __repr__(self):
        arr_txt = str(self)
        desc = 'Image(' + arr_txt.replace('\n', '\n' + ' '*6)
        for k,v in self.__dict__.iteritems():
            if k[0]<>'_':
                v = str(v)
                if k=='info' and len(v)>10: v = v[:6] + '...' 
                desc += ', ' + k + '=' + v
        desc += ')'
        return desc
        

class PILSerializer(object):
    """
    Serializer for Image objects using the Python Image Library (PIL)
    
    implements: 
      - `dump(obj,stream)`: 
            write (serialize) an image to a stream 
      - `load(stream)`:
            read (deserialize) an image from a stream
      - load_image(url,color,dtype,scale,from_color)
            Static method that create an Image object from `url` with storage
            entry and seiralizer set.
    
    The purpose of `PILSerializer` is to provide a serializing/deserializing
    interface for `Image` objects. In pratice, it is used through the image 
    storage entry. See `get/set_storage_entry`, the `datastructure.Data` class
    and the `storage` module for details on automatic serialization protocol.
    
    The (de)serialization is done using writing and reading functionalities of 
    PIL, which usually require suitable data transformation to be defined.
    """
    def __init__(self, img_color, img_dtype,
                       pil_format, pil_mode=None, pil_param=None,
                       ser_color=None, ser_dtype='auto', ser_scale='dtype',
                       extension=None):
        """
        To be serialized, the image data format (`img_color` & `img_dtype`) has
        to fit one of the PIL mode (`pil_mode`) which it-self should be suitable
        for required image format (`pil_format`, eg. 'PNG' or 'TIFF').
        
        If it is not the case, the serialization conversion arguments (`ser_*`)
        can be used to apply automatic conversion (using `imconvert`) to the 
        given image before PIL serialization: 
         - the ser_* arguments values are used by the `dump` method
         - the reverse is used by the `load` method at deserialization
        
        The default parameters for `ser_dtype` and `ser_scale` induce the 
        selection of some "common" values (see "automatic conversion" below)
            
        :Inputs:
          - `img_color`:
              color of input Image object
          - `img_dtype`: 
              dtype of input Image object
          
          - `pil_format`:
              serialization format used by pil (e.g. 'PNG', 'TIFF') (1)
              (see http://effbot.org/imagingbook/formats.htm)
          - `pil_mode`:
              serialization mode used by pil (e.g. 'L', 'RGB', 'F') (1)
              (see http://effbot.org/imagingbook/concepts.htm)
          - `pil_param`:
              additional paramaters used by pil save - depends on `pil_format` (1)
              Should be a dictionary. If it contains a field 'pnginfo', which
              should also be a dictionary, it is converted to the suitable
              data structure for saving.
              
          - `ser_color`:
              color conversion used before serialization (2)
          - `ser_dtype`:
              dtype conversion used before serialization (2)
          - `ser_scale`:
              pixels value scaling done before serialization (2)
              
          - `extension`:
              File extension that indicate the content type.
              By default use: '.' + pil_format.lower()
          
          (1) the pil_* arguments are directly passed to the save method of 
              PIL.Image. See PIL.Image's save method documentation.
              
          (2) the ser_* arguments are passed to `imconvert` before serialization.
              The opposite conversion is applied at deserialization.
              See the `imconvert` documentation and "automatic conversion" below
        
         
        :Automatic conversion:
            pil_mode=None:
            --------------
            At serialization, the automatic conversion use the PIL `fromarray`
            method. If None, pil_mode is selected w.r.t the image shape & dtype: 
            
            shape | color - dtype        => pil mode
            ========================================
            (.,.)   gray  - bool         => 1     1-bit pixels, black and white
            (.,.)   gray  - uint8        => L     8-bit pixels, black and white
            (.,.)   gray  - int8 to 32   => I     32-bit signed integer pixels
            (.,.)   gray  - float32 & 64 => F     32-bit floating point pixels
            (.,.,3) rgb   - uint8        => RGB   3x8-bit pixels, true colour
            (.,.,4) rgba  - uint8        => RGBA  4x8-bit pixels, tc with transparency
              
            (see PIL.Image._fromarray_typemap)
            
            serialization conversion parameter:
            -----------------------------------
            The serialization conversion parameters (ser_*) provide some 
            automatic behavior. They are passed to `imconvert` and provide this
            function behavior. Eg:        
              - `ser_dtype` and `ser_color` = None means no conversion
              - `ser_scale=dtype` maps the value range of source and destination 
            
            On top, if ser_dtype='auto',the following value is selected:
            
              if image color is not gray:            ser_dtype = uint8
              else if image dtype is bool or uint8:  ser_dtype = None
              else if pil_format is 'TIFF':          ser_dtype = float32
              else:                                  ser_dtype = uint8

        :Warning:
            Deserialized previously serialized image, i.e. load(dump(image)), 
            should have the same data format (color and dtype) as the original. 
            However, it is the serialization format (`ser_*`) that is saved. So 
            if the image data format is not suitable w.r.t the pil mode and
            format, the (de)serialization induces a lost of precision!
            
        :See also:
            imconvert
            PIL modes:   http://effbot.org/imagingbook/concepts.htm
            PIL formats: http://effbot.org/imagingbook/formats.htm    
        """
        # manage automatic serialization conversion of dtype
        if ser_dtype=='auto':
            if img_color!='gray':             ser_dtype = 'uint8'
            elif img_dtype in (bool,'uint8'): ser_dtype = None    # no conversion
            elif pil_format=='TIFF':          ser_dtype = 'float32'
            else:                             ser_dtype = 'uint8'
            
        # check for pnginfo in pil_param
        if pil_param is None:
            pil_param = dict()
        elif pil_param.has_key('pnginfo'):
            from PIL.PngImagePlugin import PngInfo
            astr = lambda x: x if isinstance(x,basestring) else str(x)
            info = PngInfo()
            for key,value in pil_param['pnginfo'].iteritems():
                info.add_text(astr(key), astr(value)) 
            pil_param['pnginfo'] = info
            
        if extension is None:
            extension = '.' + str(pil_format).lower()
            
        self.img_color  = img_color 
        self.img_dtype  = img_dtype  
        self.pil_format = pil_format 
        self.pil_mode   = pil_mode   
        self.pil_param  = pil_param  
        self.ser_color  = ser_color  
        self.ser_dtype  = ser_dtype  
        self.ser_scale  = ser_scale
        self.extension  = extension
        
        if ser_scale=='normalize':
            self.img_scale = 'dtype'
        elif isinstance(ser_scale,basestring):
            self.img_scale = ser_scale
        else:
            self.img_scale = 1/float(ser_scale)
        
    def dump(self,image, stream):
        from PIL.Image import fromarray
        img = imconvert(image, color=self.ser_color,     dtype=self.ser_dtype,
                              from_color=self.img_color, scale=self.ser_scale)
        img = fromarray(img, mode=self.pil_mode)
        img.save(stream, format=self.pil_format, **self.pil_param)
        
    def load(self,stream):
        from PIL import Image
        img = _np.array(Image.open(stream))
        return imconvert(img, color=self.img_color,      dtype=self.img_dtype,
                              from_color=self.ser_color, scale=self.img_scale)
        
    @staticmethod
    def load_image(url, color=None, dtype=None, scale='dtype', from_color=None):
        """
        return an Image object created from `url`
        
        The image has the suitable PIL serializer and storage entry set
        """
        # create empty PILSerializer instance
        class TMP(object): pass
        s = TMP()
        s.__class__ = PILSerializer
        
        # create the StorageEntry
        e = _create_entry(url)
        
        # load image storage type
        from PIL import Image
        img = Image.open(e.url)
        s.pil_format = img.format
        s.pil_mode   = img.mode
        s.pil_param  = dict()
        
        # make the Image instance
        img = _np.array(img)
        if from_color is None: from_color = detect_color_space(img)
        img = imconvert(img, color=color, dtype=dtype,from_color=from_color, scale=scale)
        
        # fill serializer 
        s.img_color  = color 
        s.img_dtype  = dtype  
        s.img_scale  = scale
        s.ser_color  = from_color  
        s.ser_dtype  = img.dtype  
        s.ser_scale  = _scale_inverse(scale)
        
        # attach StorageEntry to image
        img.__serializer__ = s
        img.set_storage_entry(e)
        
        return img
        

class ImageSequence(_Sequence):
    """
    provide access to list of image file
    ##ImageSequence todo: doc
    
    ##!! conversion arguments used for output 
    ##      => roi and filter useless 
    ##      => might need to be changed for input !!
    ##      => what about loading image while output ??? => forbiden ?
    ##      => or put everything in auto_save (see Sequence todo)
    ##      => or not, set images hae to be good for saving
    ##          => they are still saving option (pil stuff)
    """
    def __init__(self, files=None, output=None, color=None, dtype=None, scale='dtype',  
                       roi=None,   filter=None, buffer_size=2, auto_save=True):
    
        _Sequence.__init__(self, files=files, output=output, 
                           buffer_size=buffer_size, auto_save=auto_save)
        
        # config
        self.color  = color   # color imposed on loaded images
        self.dtype  = dtype   # dtype ------------------------
        self.scale  = scale   # dtype ------------------------
        self.roi    = roi     # load subarea of image (slice tuple)
        self.filter = filter  # function to apply on loaded images
        
    def set_input(self, color=None, dtype=None, scale='dtype', roi=None, filter=None):
        """ convert to an input only sequence """
        self.color  = color  
        self.dtype  = dtype  
        self.scale  = scale  
        self.roi    = roi    
        self.filter = filter 
        
        _Sequence.set_input(self)
        
        self.clear_buffer()
        
    @_property
    def roi(self):
        """ Slice object that is applied on all loaded image """
        return self._roi
        
    @roi.setter    
    def roi(self,roi=None):
        if isinstance(roi,slice): roi = (roi,)*2  ## image dimension restricted to 2 ?
        if roi is not None:       roi = _np.asanyarray(roi)
        self._roi = roi
        self.clear_buffer()
        
    @_property
    def filter(self):
        """ filter function that is called on all loaded image """
        return self._filter
    @filter.setter
    def filter(self, filter_):
        if filter_ is not None and not hasattr(filter_,'__call__'):
            raise TypeError("Image filter should None or callable (ie. function)")
        self._filter = filter_
        
    def _load_item_(self, filename, index):
        """
        load an image.
        Base Sequence class manage sequencing
        """
        if self.output is not None:
            raise AttributeError("This ImageSequence is output only")
            
        img = Image(filename, color=self.color, dtype=self.dtype, scale=self.scale)
        img.set_storage_entry(filename)
        if self.roi    is not None: 
            if self.roi.ndim>1: img = img[self.roi[index].tolist()]
            else:               img = img[self.roi.tolist()]
        if self.filter is not None: img = self.filter(img)
        
        return img
        
    def _save_item_(self, filename, image):
        """
        Save an image. 
        
        The image should be of suitable color and dtype for the file format
        see Image.save documentation
        """
        if not isinstance(image,Image): image = Image(image)
        image.set_storage_entry(filename)
        return image.save(color=self.color,dtype=self.dtype,scale=self.scale)
        
    def play(self, filter=None):
        """
        very basique image sequence player using matplotlib.
        this method will be moved to the gui.image package
        """
        if filter is None: 
            filter = lambda x: x
        
        import matplotlib.pyplot as plt
        i = 0
        print "enter:next image, p:previous, number:selected frame, 'q':quit"
        while True:
            plt.imshow(filter(self[i]), hold=False)
            plt.draw()
            k = raw_input("%4d > " % i)
            if   k=='q':      break
            elif k=='n':      i -= 1
            elif k.isdigit(): i  = int(k)
            else:             i += 1
            i %= len(self)
        
def detect_color_space(image_array):
    """
    Automatic detection of image color space.
    If number of dimension is < 3       => gray, 
    otherwise if last axis has length 3 => rgb
    otherwise if last axis has length 4 => rgba
    otherwise                           => gray  (n-dimensional)
    """
    if   image_array.ndim<3:       return 'gray'
    elif image_array.shape[-1]==3: return 'rgb'
    elif image_array.shape[-1]==4: return 'rgba'
    else:                          return 'gray'

def _scale_inverse(scale):
    """ return the inverse of scale in imconvert """
    if scale=='normalize':             return 'dtype'   # not bijective!
    elif isinstance(scale,basestring): return scale
    else:                              return 1./scale
    
@_node('converted_image')
def imconvert(image, color=None, dtype=None, scale='dtype', from_color=None):
    """
    Convert Image color space and/or dtype
    
    :Inputs:
        color:  new color space.                         - None means no conversion
        dtype:  the data type to convert data to         - None means no conversion
        scale:  'dtype', 'normalize', 'view' or a value  - see Scale and data type below
        from_color: color space of input data. 
                    If None, select it automatically - see Color space below
        
    :Outputs:
        return an Image object 
        
    :Color space:
        For now only rgb-to-gray and gray-to-rgb are possible.
        Other color conversions will raise an error
        
        if from_color is None, then
        - if input is an Image instance, takes its color attributes
        - otherwise choose it automatically using the detect_color_space function

    :Scale and data type:
        Possible scale are: 'normalize', 'dtype', 'view' or any numeric value.
        
        If the scale arguments is a numeric value, the image is multiplied by 
        this scale before dtype conversion.
        
        If scale is 'dtype', it means it chose a scale factor (multiplaction of
        the data) to map the image extremum typical of input and output dtype. 
        It maps 
            in.dtype [min,max]  to out.dtype [min,max]
            
        where min & max are, w.r.t dtype:
            int8:          min=0   max=127
            uint8 & int16: min=0   max=255
            other (u)int): min=0   max=65531
            float (any):   min=0   max=1
            bool:          min=0   max=1

        If scale is 'normalize', then it map the input image minimum and 
        maximum value to the output dtype min/max (table above)
        
        On top of that, underflow and overflow following scaling and type 
        conversion are clipped to the bound of the output dtype.
        > This differ with the behavior of `numpy.astype`, which does wrapping
          
        In addition to all numeric numpy dtype, dtype argument can also be 'f*'
        which means to assert dtype is float, without enforcing a byte precision.
        It uses the input dtype if it is any floating point dtype. Otherwise
        it converts to python default (64 bits)
        
        When the output dtype is boolean, then it simply return image!=0. 
        Scaling has no use.

    :Note:
        Color conversion is done in a float precision: 
            If the output dtype is floating point, then it use it.
            But it is None, then the returned image will still be float (64).
            
        Thus in the current implementation, dtype conversion to float is implied 
        by color change. However this is a side effect, and should not be 
        accounted for:
        *** it is recommanded to set the output dtype when converting color ***
    """        
    # conversion needed w.r.t initial color and dtype and arguments
    # -------------------------------------------------------------
    if from_color is None:
        if isinstance(image,Image): from_color = image.color
        else:                       from_color = detect_color_space(image)

    from_dtype = image.dtype
    
    if isinstance(dtype,basestring) and dtype=='f*':
        if from_dtype.kind=='f': dtype = cvt_dtype = from_dtype
        else:                    dtype = cvt_dtype = _np.dtype(float)
    elif dtype is not None:      
        dtype     = _np.dtype(dtype)
        cvt_dtype = dtype if dtype.kind=='f' else _np.dtype(float)
    else:                        
        cvt_dtype = _np.dtype(float)
    
    # color conversion
    # ----------------
    if color is not None and from_color!=color:
        if from_color in ('rgb', 'rgba') and color=='gray':
            # convert rgb(a) to gray
            coef  = _np.array([0.3,0.59,0.11],dtype=cvt_dtype).reshape((1,)*(image.ndim-1) + (-1,))
            image = _np.sum(_np.asanyarray(image,dtype=cvt_dtype) * coef, axis=-1)
        elif from_color=='gray' and color in ('rgb', 'rgba'):
            img = _np.ones(_np.asanyarray(image).shape + (len(color),))
            img[...,0] = image
            img[...,1] = image
            img[...,2] = image
            image = img
        elif from_color!='gray' or color!='gray':
            raise ValueError('conversion from %s to %s is not possible' % (from_color,color))
    
    # dtype conversion
    # ----------------
    if scale!='dtype' or (dtype is not None and dtype!=image.dtype):
        # factor to map maximum value between dtype (vmax_dst / vmax_src)
        #   1 for float; 255 for 8bits integers, and 2**16 for > 8bits integer
        ##?? if dtype is None: dtype = image.dtype
        if dtype is None: dtype = image.dtype
        '''
        conversion table: 
          - multiplication done B(efore), A(fter), H(igher precision) 
          - *: Clipping needed
          
        conversion: normalize   dtype   value
        any > b     -------- image!=0 --------
        
          f > u         B        B*      B*
          i > u         B        B*      B*
          u > u         H        H*      H*
          b > u         A        A*      A*

          f > i         B        B*      B*
          i > i         H        H*      H*
          u > i         A        A*      A*
          b > i         A        A       A*

          f > f         H        H       B
          i > f         A        A       H
          u > f         A        A       A
          b > f         A        A       A
        '''
        def dtype_max(dtype):
            """ Typical maximum image value depending of dtype
                for bool and float,  1
                for int8:            2**7  -1 = 127.
                for uint8 and int16: 2**8  -1 = 257.
                for other (u)int:    2**16 -1 = 65535.
            """
            if dtype.kind in ('b','f'):
                return 1.
            else:
                return 2**(8*_np.clip((dtype.itemsize - (dtype.kind=='i')),0.875,2))-1

        if scale=='view':
            image = image.view(dtype=dtype)
        elif dtype.kind=='b':
            image = image!=0
        else:
            # scale before or after
            idt = image.dtype.kind
            odt = dtype.kind
            if odt=='u':
                if idt in ('f','i'): before = True
                elif idt == 'b':     before = False
                else:                before = image.dtype.itemsize <= dtype.itemsize
            elif odt=='i':
                if idt in ('u','b'): before = False
                elif idt == 'f':     before = True
                else:                before = image.dtype.itemsize <= dtype.itemsize
            else: # out dtype is float
                if idt == 'f':       before = image.dtype.itemsize <= dtype.itemsize
                else:                before = False
            
            # compute shift and scaling
            if scale == 'normalize':
                imin = image.min()
                imax = image.max()
                if imin==imax:
                    if idt=='b': imin, imax = sorted((imin, not imin))
                    else:        imin, imax = sorted((imin, imin+1))
                shift  = -imin
                factor = dtype_max(dtype) / (imax-imin) 
            elif scale == 'dtype':
                shift  = 0
                factor = dtype_max(dtype) / dtype_max(from_dtype)
            else:
                shift  = 0
                factor = scale
                
            # do the scaling
            if before:
                img =  _np.asanyarray((shift+image)*factor,dtype=dtype)
            else:    
                img = (_np.asanyarray(image,dtype=dtype)+shift)*_np.array(factor,dtype=dtype)
            
            # apply clipping, if necessary
            if scale!='normalize' and odt!='f' and (scale!='dtype' or odt!='b'):
                omin = _np.iinfo(dtype).min
                omax = _np.iinfo(dtype).max
                
                imin = float(omin)/factor - shift
                imax = float(omax)/factor - shift
                
                img[image<(float(omin)/factor - shift)] = omin
                img[image>(float(omax)/factor - shift)] = omax
                
            image = img
                
    # returned Image object
    # ---------------------
    image  = image.view(type=Image)
    image.color = color if color is not None else from_color
    image.scale = scale
    image.from_color = from_color
    
    return image

def draw_line_2d(image, x, y, value=0, width=1):
    """
    Draw pixels of a line between points p1 = (x[0],y[0]) and p2 = (x[1],y[1])
    """
    x = _np.asarray(x,dtype='float32')
    y = _np.asarray(y,dtype='float32')
    
    # construct pixels indices of the unit width line
    if abs((x[1]-x[0])/(y[1]-y[0]))>=1:  # width longer than height
        if x[1]<x[0]:
            x = x[::-1]
            y = y[::-1]
            
        x = _np.round(x)
        X = _np.mgrid[x[0]:x[1]+1]           # x-coordinates of the line
        Y = _np.linspace(y[0],y[1], X.size)  # y-coordinates of the line
    else:
        if y[1]<y[0]:
            x = x[::-1]
            y = y[::-1]
            
        y = _np.round(y)
        Y = _np.mgrid[y[0]:y[1]+1]           # y-coordinates of the line
        X = _np.linspace(x[0],x[1], Y.size)  # x-coordinates of the line
        
    # manage width (but if line is actually a dot)
    if (width>1) and (X.size>1):
        dx = X[-1] - X[0]           #
        dy = Y[-1] - Y[0]           #
        dl = (dx**2 + dy**2)**0.5   #  direction vector of the line (normalized) 
        dx /= dl                    # (the normal is obtain by switching dx & dy)
        dy /= dl                    #
        
        w  = _np.arange(width) - ((width-1)/2.)
        X  = _add_dim(X,axis=1,size=width) + _add_dim(dy*w,axis=0,size=X.size)
        Y  = _add_dim(Y,axis=1,size=width) + _add_dim(dx*w,axis=0,size=Y.size)
    
    image[_np.rint(Y).astype(int),_np.rint(X).astype(int)] = value
    
