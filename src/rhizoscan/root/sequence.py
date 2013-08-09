import numpy as _np

from rhizoscan.datastructure import Mapping as _Mapping
from rhizoscan.datastructure import Sequence
from rhizoscan.tool  import _property
from rhizoscan.image import ImageSequence

# decorator that copy a function doc
class _append_doc:
    def __init__(self,fct):
        self._fct = fct
    def __call__(self,fct):
        doc =  "----- %s doc -----" % (self._fct.__module__ + '.' + self._fct.__name__)
        doc = "\n    " + doc + self._fct.__doc__ + '-'*len(doc)
        if fct.__doc__ is None:
            fct.__doc__ = ""
        fct.__doc__ += doc 
        return fct
        
def remove_background(input_seq, output, distance, smooth=1):
    """
    Call image.remove_background on all image of seq_in
    
    Input:
    ------
    seq_in   an input ImageSequence
    output   printf-able output filename without extension - see Sequence doc
    distance the maximal diameter of foreground objects
    smooth   used for background interpolation
    
    Output:
    -------
    Return an ImageSequence of the filtered images
    
    See image.remove_background
    """
    from os.path import sep
    from .image import remove_background
    
    out_seq = ImageSequence(output=output+'.png', dtype='uint8', scale=(2**8.-1))  ##(compressed) png
    
    print_start = '  filtering image% '+str(int(_np.ceil(_np.log(len(input_seq)/_np.log(10)))))+ 'd: [...]'
    for i,img in enumerate(input_seq):
        print print_start % i + sep.join(img.get_data_file().rsplit(sep)[-2:])
        img = remove_background(img,distance=distance,smooth=smooth)
        #img -= img.min()
        img /= img.max()
        out_seq[i] = img
        
    out_seq.set_input()
    return out_seq

def segment_root_image(seq, output, mask=None, scale=1, min_dimension=5):
    """
    Call image.segment_root_image on all image of seq
    
    Input:
    ------
      - seq    the ImageSequence to segment
      - output printf-able output filename without extension - see Sequence doc
      - mask   optional ImageSequence of mask
      - scale  scale used for the return transform  ##TODO
      - min_dimension  ##TODO
     
    Output:
    -------
      - the ImageSequence of the segmented image
      ##TODO- a list of 3x3 matrices for scaling of image coordinates by scale
      ##TODO- crop slices
    
    See Image.segment_root_image
    """
    from os.path import sep
    from .image import segment_root_image as segment
    
    # output root mask sequences
    mask   = ImageSequence(output=output+'.png', dtype='uint8', scale=1)
    if mask is None: mask = [None]*len(seq)
    
    print_start = '  segmenting image% '+str(int(_np.ceil(_np.log(len(seq)/_np.log(10)))))+ 'd:'
    for i,img in enumerate(seq):
        print print_start % i + sep.join(img.get_data_file().rsplit(sep)[-2:])
        mask[i] = segment(img,mask=mask[i])
        
    mask.set_input()
    return mask
    
    
def segment_root_in_petri_frame(seq, output, plate_border=0.06, plate_width=1, min_dimension=5):
    """
    Call image.segment_root_in_petri_frame on all image of seq
    
    Input:
    ------
      - seq            the ImageSequence to segment
      - output         the printf-able filename for the output sequence
      - plate_border:  the size of the plate border (overlap of the top part on 
                       the bottom), it should be given as a percentage of the 
                       plate width
      - plate_width:   the plate size in your unit of choice (see output) 
      - min_dimension: remove pixel area that have less that this number of 
                       pixels in width or height.
    Output:
    -------
      - The segmented (labeled) root mask sequence cropped around the petri plates.
        The petri plates has been removed
      - A list of 3x3 transformation matrices that represents the mapping of 
        image coordinates into the plate frame with origin at the top-left 
        corner, y-axis pointing downward, and of size given by plate_width
      - A list of bounding boxes of the detected plates w.r.t the original mask 
        shape given as a tuple pair of slices (the cropping used)
    
    See Image.segment_root_in_petri_frame
    """
    from os.path import sep
    from .image import segment_root_in_petri_frame as segment
    
    # output sequences
    cluster   = ImageSequence(output=output+'.png', dtype='uint8', scale=255)
    transform = [None]*len(seq)
    bbox      = [None]*len(seq)
    
    print_start = '  segmenting image% '+str(int(_np.ceil(_np.log(len(seq)/_np.log(10)))))+ 'd:'
    for i,img in enumerate(seq):
        print print_start % i + sep.join(img.get_data_file().rsplit(sep)[-3:])
        c,T,bbx = segment(img,plate_border=plate_border, plate_width=plate_width, min_dimension=min_dimension)
        cluster[i] = c
        transform[i] = T
        bbox[i] = bbx
        
    cluster.set_input()
    return cluster, transform, bbox
    
def segment_root_in_circle_frame(seq, output, n=4, pixel_size=1, min_dimension=5):
    """
    Call image.segment_root_in_circle_frame on all image of seq
    
    Input:
    ------
      - seq            the ImageSequence to segment
      - n:             the number of circles to be found
      - pixel_size:    size of a pixel in the unit of your choice
                         e.g. 1/(25.4*scaned-dpi) for millimeter unit
      - min_dimension: remove pixel area that have less that this number of 
                       pixels in width or height.
    Output:
    -------
      - The sequence of segmented (labeled) root mask cropped around the circle
        frame. The frame are removed.
      - A list of 3x3 transformation matrix that represents the mapping of image 
        coordinates into the detected frame: the origin is the top-left circle, 
        x-axis pointing toward the top-right circle, and the size (scale) is 
        computed based on the given 'pixel_size'. 
      - A list of the bounding box containing all detected circles w.r.t the 
        original mask shape, given as a tuple pair of slices (the cropping used)
    """
    from os.path import sep
    from .image import segment_root_in_circle_frame as segment
    
    # output sequences
    cluster   = ImageSequence(output=output+'.png', dtype='int32', scale=1)
    transform = [None]*len(seq)
    bbox      = [None]*len(seq)
    
    print_start = '  segmenting image% '+str(int(_np.ceil(_np.log(len(seq)/_np.log(10)))))+ 'd:'
    for i,img in enumerate(seq):
        print print_start % i + sep.join(img.get_data_file().rsplit(sep)[-3:])
        c,T,bbx = segment(img, n=n, pixel_size=pixel_size, min_dimension=min_dimension)
        cluster[i] = c
        transform[i] = T
        bbox[i] = bbx
        
    cluster.set_input()
    return cluster, transform, bbox

def track_seeds(seed, transform):
    """
    For a image sequence of seed maps, set same id to "tracked" cluster
    
    Match seed clusters from each image to the next, to minize the distance of 
    their bounding box, then set the id of the first image to all matched cluster
    
    NOT IMPLEMENTED - see detect_seeds/leaves in image.seed 
    """
    raise NotImplementedError('track_seeds')

def mask_to_rootgraph(mask, image, mapgraph_out, plgraph_out):
    from os.path import sep
    from .image import linear_label
    from .graph import RootGraph
    
    # output cluster sequences
    mpgraph_seq = Sequence(output=mapgraph_out + '.rgraph')
    plgraph_seq = Sequence(output=plgraph_out  + '.rgraph')
       
    compute = _Mapping(cluster=[], node=['terminal'],
                       segment=['length','luminosity','radius','direction','terminal'])
       
    print_start = '  making graph from image % '+str(int(_np.ceil(_np.log(len(mask)/_np.log(10)))))+ 'd:'
    for i,m in enumerate(mask):
        print print_start % i + sep.join(m.get_data_file().rsplit(sep)[-3:])
        smap,sskl,nskl,seed = linear_label(m!=0)  ## seed added
        g = RootGraph.init_from_maps(smap=smap, nmap=nskl, sskl=sskl, image=image[i], compute=compute)
        mpgraph_seq[i] = g
        plgraph_seq[i] = g.compute_polyline_graph(sskl)
    
    return mpgraph_seq, plgraph_seq

def compute_root_tree(graph, output, to_tree=2,to_axe=1):
    from .graph import RootAxialTree
    
    tree = Sequence(output=output+'.tree')
    for i,g in enumerate(graph):
        print 'converting graph %d to axial tree' % i
        tree[i] = RootAxialTree(node=g.node,segment=g.segment, to_tree=to_tree, to_axe=to_axe)
    
    tree.set_input()
    
    return tree


class RootSequence(_Mapping):
    """
    Manage a sequence of root data initiated with as sequence of images
    """        
    def __init__(self, directory, **kargs):
        """
        Construct a empty RootSequence 
        """
        self.output_directory = directory
        self.merge(kargs)
    
    @_property
    def output_directory(self):
        return self.__outdir
    @output_directory.setter
    def output_directory(self,directory):
        import os
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.__outdir = os.path.abspath(directory)
        self.set_data_file(os.path.join(directory,'root_sequence.data'))
        
    @_append_doc(remove_background)
    def remove_background(self, root_max_width, input='input', output='image'):
        """
        Call self[output]=remove_background(self[input], distance=root_max_width, smooth=1)
        """
        self._check_data([input])
        out_fname = self.make_output_filename(names=output,seq_length=len(self[input]))
        self[output] = remove_background(self[input],output=out_fname, distance=root_max_width, smooth=1)
        self.dump()
    @_append_doc(segment_root_image)
    def segment_root_image(self,input='input', output='root_mask', mask=None):
        self._check_data([input])
        out_fname = self.make_output_filename(names=output,seq_length=len(self[input]))
        self[output] = segment_root_image(self[input],output=out_fname, mask=self.get(mask,None))
        self.dump()
        
    def segment_root_in_petri_frame(self,input='input', output=['cluster','transform','bbox'], plate_border=0.06, plate_width=1, min_dimension=5):
        self._check_data([input])
        out_fname = self.make_output_filename(names=output[0],seq_length=len(self[input]))
        out = segment_root_in_petri_frame(self[input],output=out_fname, \
                                          plate_border=plate_border,    \
                                          plate_width=plate_width,      \
                                          min_dimension=min_dimension)
        
        for name,value in zip(output,out): self[name] = value
        
        # create a cropped copy of input image sequence
        ## should be a functions: self.crop_sequence(input='image', roi='bbox')
        self['c_'+input] = ImageSequence(self[input].get_file())
        self['c_'+input].roi = self[output[2]]
        
        self.dump()
        
    def segment_root_in_circle_frame(self,input='input', output='cluster', n=4, pixel_size=1, min_dimension=5):
        out_fname = self.make_output_filename(names=output[0],seq_length=len(self[input]))
        out = segment_root_in_circle_frame(self[input],output=out_fname, \
                                          plate_border=plate_border,     \
                                          plate_width=plate_width,       \
                                          min_dimension=min_dimension)
        
        for name,value in zip(output,out): self[name] = value
        self.dump()
        
    def mask_to_rootgraph(self,mask='cluster', image='input', output=['mapgraph','plgraph']):
        mgr_fname = self.make_output_filename(names=output[0],seq_length=len(self[mask]))
        grp_fname = self.make_output_filename(names=output[1],seq_length=len(self[mask]))
        out = mask_to_rootgraph(mask=self[mask],image=self[image], \
                                          mapgraph_out=mgr_fname, plgraph_out=grp_fname)
        
        for name,value in zip(output,out): self[name] = value
        self.dump()


    def track_graph_seed(self, graph1='mapgraph', graph2='plgraph', seed_prop='luminosity', plant_number=None):
        """
        call seed.track_graph_seed on graph1, then propagae on graph2 (if not None)
        use seed_prop of graph to classify graph segment
        and detect 'plant_number' seeds cluster
        
        graph2 should have been generated from graph1
        if plant_number is None, use self.plant_number
        """
        if plant_number is None: plant_number = self.plant_number
        
        from .seed import track_graph_seed
        track_graph_seed(graph_seq=self[graph1], seed_prop=seed_prop, seed_number=plant_number)
        
        if graph2 is not None:
            for i in range(len(self[graph1])):
                g1 = self[graph1][i]
                g2 = self[graph2][i]
                g2.segment.leaves = g1.segment.leaves[g2.segment.sid]
                g2.dump()
        #self.dump() ## useless: g1 and g2 are saved 
        
    def compute_root_tree(self, graph='plgraph', output='tree', to_tree=2,to_axe=1):
        if not self.has_field(graph):
            raise UnboundLocalError("project is not ready for tree graph computation. Call compute_root_graph() first")
        
        out_name = self.make_output_filename(names=output,seq_length=len(self[graph]))
        self[output] = compute_root_tree(graph=self[graph], output=out_name, \
                                          to_tree=to_tree, to_axe=to_axe)
        self.dump()
    
    def run(self, root_max_width, to_run=['filter','cluster','label','graph']): ## leaves (requires transition...) 
        """
        to_run can be any of
            'filter'  -> compute the 'image' data
            'cluster' -> compute the 'cluster', 'transform', and 'pbox' data
            'label'   -> compute the 'smap', 'sskl', 'nskl' data
            'graph'   -> compute the 'rgraph' and 'plgraph' data
        """
        import time
        t0 = time.time()
        t1 = t0
        
        def uptime():
            print '=> time elapse:', time.ctime(time.time()-t1 - 3600)[11:19]
            return time.time()            

        if 'filter' in to_run:
            self.filter_image(root_max_width=root_max_width)
            t1 = uptime()
        if 'cluster' in to_run:
            self.cluster_roots()        
            t1 = uptime()
        if 'label' in to_run:
            self.label_roots()          
            t1 = uptime()
        if 'graph' in to_run:
            self.compute_root_graph()   
            t1 = uptime()
        if 'tree' in to_run:
            self.compute_root_tree()   
            t1 = uptime()
            
        print '   => total time elapse:', time.ctime(time.time()-t0 - 3600)[11:19]
    
    def _check_data(self, data_name, previous_func=''):
        for name in data_name:
            if not self.has_field(name):
                msg = "Data '%s' does not exist" % name + '.'
                if previous_func is not None: msg += "Call %s() first" % previous_func
                raise UnboundLocalError(msg)
    def _make_output_path(self,output,*args):
        outdir = os.path.join(self.output_directory,output)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        return [os.path.join(outdir,name) for name in args]
    
    def make_output_filename(self,names, dir_name=None, seq_length=None):
        """
        Create a path string for all names, possibly valid for Sequence output 
        
        if dir_name is None, use first name
        if names is a string, return a string
        if names is a list of string, return a list of string
        
        if max_lengh is None: return a simple filename. Otherwise, return a 
        printf-able filename for output sequence whith max _length <= seq_length 
        """
        import os
        if isinstance(names,basestring):
            names = [names]
        if dir_name is None:
            dir_name = names[0]
        outdir = os.path.join(self.output_directory,dir_name)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
            
        if seq_length is None: name_end = ''
        else:                  name_end = '%0'+str(int(_np.ceil(_np.log(seq_length)/_np.log(10))))+'d'
        out_fname = [os.path.join(outdir,name)+name_end for name in names] ## don't put output_directory for contained data?

        if len(out_fname)==1:
            out_fname = out_fname[0]
        return out_fname
        



