# icon of openalea package
__icon__ = 'mandelbrot.png'


import numpy as _np
import scipy as _sp
import scipy.ndimage as _nd

from rhizoscan.ndarray import add_dim as _add_dim
from rhizoscan.ndarray import aslice  as _aslice

from rhizoscan.workflow.openalea import aleanode as _aleanode  # decorator to declare openalea nodes        
from rhizoscan.workflow    import Data     as _Data     
from rhizoscan.workflow    import Sequence as _Sequence     
from rhizoscan.tool        import static_or_instance_method as _static_or_instance
from rhizoscan.tool        import _property

class Image(_np.ndarray, _Data):
    """
    ndarray subclass designed to represent images
    
    ##TODO: doc
    """
    def __new__(cls, array_or_file, color=None, dtype=None, scale='dtype', from_color=None):
        # read file if input is a file name
        # & cast input array to be our Image class
        # for now
        #   color can be None (no change) or 'gray'            
        #   dtype can be None (no change), any dtype, or 'f*' (assert_float)
        #   scale used by imconvert => keept !!
        
        # create Image instance
        # ---------------------
        if isinstance(array_or_file,basestring):
            obj = _nd.imread(array_or_file).view(cls)
            obj.set_data_file(array_or_file)
        else:
            obj = _np.asanyarray(array_or_file).view(cls)
            obj.set_data_file('')
            
        # conversion
        # ----------
        if obj.dtype.kind not in ('b','u','i','f'):
            print "\033[31m*** Image is not numeric (and cannot be converted) ***\033[30m"
            obj.color = 'unknown color space'
        else:
            obj = imconvert(obj,color=color,dtype=dtype, from_color=from_color, scale=scale)
            
        obj.scale = scale
        obj.from_color = from_color
        
        return obj
        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        #if hasattr(obj,'color'): self.color = obj.color
        #else:                    self.color = detect_color_space(self)
        self.color = detect_color_space(self)
        if hasattr(obj,'scale'):      self.scale = obj.scale
        else:                         self.scale = 'dtype'
        if hasattr(obj,'from_color'): self.from_color = obj.from_color
        else:                         self.from_color = None
        
        if hasattr(obj,'get_data_file'):
            self.set_data_file(obj.get_data_file())
        
        ## image.r/g/b must be done using weakreaf (?) 
        #if self.color in ('rgb','rgba'):
        #    self.r = self[...,0]
        #    self.g = self[...,1]
        #    self.b = self[...,2]
        #if self.color == 'rgba':           
        #    self.a = self[...,3]
            
    def __init__(self, array_or_file, color=None, dtype=None, from_color=None, scale='dtype'):
        ## todo: doc
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
        
    # saving
    def imsave(self, filename):
        """ Simple call to scipy.misc.imsave: save the image as scaled uint8 """
        from scipy.misc import imsave 
        imsave(filename,self)
        
    @_static_or_instance
    def save(image, filename=None, color=None, dtype='auto', scale='dtype',
                    pil_format=None, pil_mode=None, **pil_params):
        """
        Save the image to file 'filename' using PIL.Image
        
        This method can be called as
          1a a static   method:  Image.save(some_array,        file_name,      ...)
          1b a static   method:  Image.save(some_Image_object, file_name=None, ...)
          2. a instance method:  some_Image_object.save(       file_name=None, ...)
        
        Input
        -----
        image:    the image to save (1.) or self (2.)
        
        filename: the name of the image file to save. 
                  case 1a  => filename is mandatory
                  case 1b and 2. 
                    if filename is None, use its Data file attribute.
                    otherwise save it to file filename and change *in place* the
                    Data file attribute of input image
                  
        color:    |      convert image to this color & dtype after scaling 
        dtype:    | See the important notes below and the imconvert documentation 
        scale:    | possible scale: 'normalize','dtype', 'view', or any numeric value  
        
        pil_format: image format to use. If None, guess it from the file extension
        pil_mode:   can be passed to PIL.Image.fromarray to enforce conversion 
                    mode. None means automatic selection (see below)
        pil_params: Additional parameter to pass to PIL save function. Depends 
                    on pil_format used. See PIL documentation.

        Output:
        -------
        return an empty Image object that can be used to load the image file:
            empty_image.load()
           
        The loaded Image will have suitable attribute to load the same image
        data as contained by the calling image (up to saving loss).
            - color and from_color are switch
            - dtype is the calling image dtype
            - scale is * the same as input scale if it is 'dtype' or 'view'
                       * 1./scale if input scale is numeric
                       * 'dtype'  if input scale is 'normalize' (no reverse)
        
        Important Notes:
        ----------------
        To be saved, the image has to fit one of the PIL mode (see below), then
        it can save to different image file format depending on the mode and 
        installed library.
        the pil_format, pil_mode and pil_params are directly passed to the save
        method of PIL Image. But usually default (None) value are suitable.
        See the PIL Image save method documentation 
        
        However, if the image does not meet PIL requirement in color (ie. shape) 
        and dtype, and if the automatic conversion (see dtype is 'auto' below)
        is not suitable, the color, dtype and scale arguments can be used to 
        convert the image (using the imconvert function) before saving.
        *** See the imconvert documentation ***
        
        Automatic dtype fitting:
        ------------------------
        if dtype is 'auto', apply conversion following the rules
          if color is not gray:                      convert to uint8
          if image.dtype is bool or uint8:           keep dtype
          if filename end by '.tif' or '.tiff',
             or if pil_format is 'TIFF':             convert to float32
          else:                                      convert to uint8
        
        Automatic conversion to pil mode: 
        ---------------------------------
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
        if isinstance(image,Image):
            if filename is not None:
                image.set_data_file(filename)
            elif image.get_data_file() is None:
                raise TypeError("filename shoud be provided: input Image object has unset Data file")
        elif filename is None:
            raise TypeError("filename should be provided")
        else:
            image = Image(image)
            image.set_data_file(filename)
            
        filename = image.get_data_file()
        
        # convert image
        # -------------
        if dtype=='auto':
            if image.color!='gray':             dtype = 'uint8'
            elif image.dtype in (bool,'uint8'): dtype = None
            elif pil_format=='TIFF' or \
                 filename.lower().endswith(('.tif','.tiff')):
                                                dtype = 'float32'
            else:                               dtype = 'uint8'
            
        img = imconvert(image, color=color, dtype=dtype, scale=scale)
        img.set_data_file(filename)
        
        # save image
        # ----------
        img = fromarray(img,mode=pil_mode)
        img.save(filename, format=pil_format, **pil_params)
        
        #loader = Image([])
        loader = image.image_loader()
        loader.from_color = color
        loader.set_data_file(filename)
        if scale=='normalize':                 loader.scale = 'dtype'
        elif not isinstance(scale,basestring): loader.scale = 1/float(scale)
        
        return loader
        
    @_static_or_instance
    def load(self):
        """
        Method to load image using an empty image as returned by image_loader()
        
        The loaded  image  is *returned*
        the calling object is *unchanged*
        
        Note:
        This method actually load the image from file given by its Data file 
        attribute and apply conversion to the same color and dtype.
        """
        return Image(self.get_data_file(),color=self.color, dtype=self.dtype, scale=self.scale)
        
    def _data_to_save_(self):
        return self.image_loader()
        
    def image_loader(self):
        loader = Image([])
        loader.dtype  = self.dtype
        loader.color  = self.color
        loader.from_color = self.from_color
        loader.set_data_file(self.get_data_file())
        return loader
        
    def _data_to_load_(self):
        return self.load()
    
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
        img.set_data_file(filename)
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
        image.set_data_file(filename)
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
    
        
def imconvert(image, color=None, dtype=None, scale='dtype', from_color=None):
    """
    Convert Image color space and/or dtype
    
    Input:
    ------
        color:  new color space.                         - None means no conversion
        dtype:  the data type to convert data to         - None means no conversion
        scale:  'dtype', 'normalize', 'view' or a value  - see Scale and data type below
        from_color: color space of input data. 
                    If None, select it automatically - see Color space below
        
    Output:
    -------
        return an Image object 
        
    Color space:
    ------------
        For now only rgb-to-gray and gray-to-rgb are possible.
        Other color conversions will raise an error
        
        if from_color is None, then
        - if input is an Image instance, takes its color attributes
        - otherwise choose it automatically using the detect_color_space function

    Scale and data type:
    --------------------
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
        > This differ with the behavior of numpy astype method, which do wrapping
          
        In addition to all numeric numpy dtype, dtype argument can also be 'f*'
        which means to assert dtype is float, without enforcing a byte precision.
        It uses the input dtype if it is any floating point dtype. Otherwise
        it converts to python default (64 bits)
        
        When the output dtype is boolean, then it simply return image!=0. 
        Scaling has no use.

    Note:
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

@_aleanode('uu','uv','vv')
def gradient_covariance_2d(image, size=3):
    """
    Compute the uu,uv,vv covariances of the gradient over a neighborhood of each image pixel
    u (resp. v) is the image gradient over x (resp. y)-coordinate.
    
    Use eigen_gradient_2d to compute the eigenvalues of the covariance matrix
    
    Input:
        - image: a 2d array, or a tuple containing the precomputed (u,v) gradient (in this order)
        - size:  the size of the neighborhood, either a scalar, or a 2-values tuple
        
    Output:
        - uu,uv,vv: the u-by-u, u-by-v, v-by-v correlation array
        
    The x-by-y covariance is computed as the sum of the x_i by the y_i (with i the
    index of a neighbor) divided by the total number of neighbor pixels.
    Note that the gradient are not centered (i.e. the mean value is removed) as it
    is not relevant to gradient diffusion.
    
    The vu covariance is equal to the uv
    """
    
    if isinstance(image, tuple): u,v = image
    else:                        v,u = _np.gradient(image)
    c = lambda x: _nd.uniform_filter(x,size=size)
    
    return c(u*u), c(u*v), c(v*v)
    
def eigen_gradient_2d(image, size=3, sort='descend'):
    """
    Compute the eigenvalues of the gradient (absolute) covariance for each pixel
    of the image, over its neighborhood
    
    Input:
        - image: a 2d array, or a tuple containing the precomputed (u,v) gradient (in this order)
        - size:  the size of the neighborhood, either a scalar, or a 2-values tuple
        - sort:  Order of returned eigenvalues. It can be either:
                  'ascend':  sort eigenvalue in  ascending order
                  'descend': sort eigenvalue in descending order (default)
                   or any function that takes 2 input arrays: (L1,L2), 
                   and return a boolean array of the same dimension, with True
                   value where L1 is correctly before L2
                   [see eigenvalue_2d()]
        
    Output:
        - an array of shape ([S],2)  where [S] is the shape of input arrays
    
    See also: gradient_covariance_2d, eigenvalue_2d
    """
    #uu,uv,vv = map(_np.abs,gradient_covariance_2d(image,size=size))
    uu,uv,vv = gradient_covariance_2d(image,size=size)
    return eigenvalue_2d(uu,uv,uv,vv, sort=sort)

@_aleanode('eigenvalues')
def eigenvalue_2d(a,b,c,d, sort='descend'):
    """
    For arrays a,b,c and d, compute the eigenvalues of all matrices 
        [[a, b],
         [c, d]]
    
    
    :Input:
        a,b,c,d: arrays of the same size 
        sort:    optional argument which can be either:
                  'ascend':  sort eigenvalue in  ascending order
                  'descend': sort eigenvalue in descending order (default)
                   or any a function that takes 2 input arrays: (L1,L2), 
                   and return a boolean array of the same dimension, with True
                   value where L1 is correctly before L2
                   Example:
                   sort=lambda L1,L2: L1<L2    is equivalent to   sort='ascend'
                   (read:  "L1 should be less than L2")
                     
    :Output:
        an array of shape ([S],2)  where [S] is the shape of input arrays
    """
    ##todo: return 2 matrices, one for each eigenvalue, or make eigv indices at start ?
    shape = a.shape
    a = a.ravel()
    b = b.ravel()
    c = c.ravel()
    d = d.ravel()
    
    # compute trace and eigenvalue difference
    trace = a+d
    delta = _np.sqrt( trace**2 - 4*(a*d - b*c) )/2
    
    # compute eigenvalues 
    eigval = _np.tile( (trace/2)[:,_np.newaxis], (1,2) )
    eigval[:,0] += delta
    eigval[:,1] -= delta
    
    eigval = _np.abs(eigval)
    
    # sort eigenvalues
    if not isinstance(sort,type(lambda:1)):
        if   sort=='ascend':  sort = lambda L1,L2: L1>L2
        elif sort=='descend': sort = lambda L1,L2: L1<L2
        else:
            raise TypeError("sort arguments should be 'ascend', descend' or a function (See doc)")
            
    order  = sort(eigval[:,[1]],eigval[:,[0]])     # check if eigv[0] and eigv[1] should be inverted
    eigval = _np.choose(_np.hstack((-order,order)), (eigval[:,[0]],eigval[:,[1]]))
    
    eigval.shape = shape + (2,)
    return eigval    
    
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
    
