import scipy.ndimage as _nd
import numpy as _np
import os    as _os
from ast import literal_eval as _literal_eval

_aleanodes_ = []
from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes
from rhizoscan.workflow.openalea  import find_interface as _interface # decorator to declare openalea nodes

from rhizoscan.tool import static_or_instance_method as _static_or_instance_method

from rhizoscan.workflow import save_data as _save
from rhizoscan.workflow import Struct as _Struct
from rhizoscan.workflow import Data   as _Data
from rhizoscan.image    import Image  as _Image
from rhizoscan.ndarray.measurements import clean_label as _clean_label

from rhizoscan.root.image          import normalize_image             as _normalize_image 
from rhizoscan.root.image          import remove_background           as _remove_background 
from rhizoscan.root.image          import segment_root_image          as _segment_root_image
from rhizoscan.root.image          import segment_root_in_petri_frame as _segment_root_in_petri_frame
from rhizoscan.root.image.seed     import detect_leaves               as _detect_leaves
from rhizoscan.root.image.seed     import detect_seeds                as _detect_seeds
from rhizoscan.root.image.to_graph import linear_label                as _linear_label
from rhizoscan.root.image.to_graph import image_graph                 as _image_graph
from rhizoscan.root.image.to_graph import line_graph                  as _line_graph
# remove local import: CPL
from rhizoscan.root.graph          import RootAxialTree               as _RootAxialTree 

def _print_state(verbose, msg):
    if verbose: print '  ', msg
def _print_error(msg):
    print '  \033[31m*** %s *** \033[30m' % repr(msg)

def _param_eval(string):
    """ safe literal eval """
    try:     return _literal_eval(string)
    except:  return string



class RootAnalysisPipeline(object):
    """
    Implement a pipeline for root image analysis
    
    A pipeline is a sequence of PipelineModule objects for which must inputs
    and outputs names correspond.
    
    See the constructor and run method documentation
    """
    
    def __init__(self, modules):
        """
        'modules' is the list of PipelineModule objects to be implemented 
        by this pipeline object.
        
        When constructing a pipeline object, all the modules 'name' attribute 
        should be different, and the name of input and output data should
        match (the pipeline transfere a shared namespace to all module)
        """
        self.modules = modules
        
    @_static_or_instance_method
    def run(pipeline, image, output=None, metadata=None, update=[], **kargs):
        """
        Run the pipeline module sequence
        
        The pipeline has a namespace (dictionary) that contains all inputs, 
        outputs and parameters. The required inputs are forwarded to the 
        suitable modules, as well as parameters. Then, the modules outputs are
        added to the namespace.
        All modules inputs, outputs and parameters name should have been chosen
        adequatly.
        
        :Inputs:
          - pipeline: the pipeline object to execute
          - image:
              An image array, or the filename of an image to analyse, *OR*
              if it contains any of the fields 'filename', 'metadata', or 
              'output', these are used as the respective run arguments, with
              image taking the value of filename.
              The image is added to the pipeline namespace as the'image' key, 
              and can thus be used by the pipeline module
          - output: 
              Base output filename. It is forwarded to modules as 'base_name'
              ** Required if 'image' argument does not provide it **
          - metadata:
              Optional dictionary-like metadata. 
              It is included in the pipeline namespace (as 'metadata') and it 
              content too, i.e. each of its field.
          - update:
              List of module's name that should be updated: pass update=True to 
              the respective module.run method (call its compute method even if
              data can be reloaded)
          - kargs:
              additional parameters (name,value) pairs to be forwarded to modules
              
        :Outputs:
          The pipeline namespace containing the shared module data and parameters.
        """
        if kargs.get('verbose',False):
            print repr(image)
        
        if hasattr(image, 'metadata'):  metadata = image.metadata
        if hasattr(image, 'output'):    output   = image.output
        if hasattr(image, 'filename'):  image    = image.filename
            
        import os
        output = os.path.expanduser(output)
        
        # create namespace
        nspace  = kargs
        nspace['image'] = load_image(image)
        if metadata is not None:
            nspace['metadata'] = metadata
            nspace.update(metadata)
            
        # execute modules pipeline
        for module in pipeline.modules:
            update_module = module.name in update or 'all' in update
            outputs = module.run(base_name=output, update=update_module, **nspace)
            nspace.update(zip(module.outputs, outputs))
                
        return nspace
        
    def __OA_run__(self, *args):##base_name, update=[], **kargs):
        """
        Call the run method from openalea dataflow 
        """
        karg = dict(zip(self.input_names, args))
        return self.run(**karg)##base_name=base_name, update=update, **kargs)

class pipeline_node(_aleanode):
    """
    Decorator that create a pipeline and attach it to the goven function
    
    Example:
        @pipeline_node([module1, module2, module3])
        def my_pipeline(): pass
    
        create a Pipeline instance call my_pipeline to execute the module 1 to 3 
    """
    def __init__(self, modules):
        self.pipeline = RootAnalysisPipeline(modules)
        #modules = pipeline.modules
        
        # create inputs descriptor
        node_inputs = []
        node_inputs.append(dict(name='image',interface='IStr'))
        node_inputs.append(dict(name='output',interface='IFileStr'))
        node_inputs.append(dict(name='metadata',interface='IDict'))
        node_inputs.append(dict(name='update', interface='ISequence', value=[]))
        input_names = [n['name'] for n in node_inputs]  # list ot keep the order
        
            # create inputs for the inputs of first module
        m1 = modules[0]
        for indata in m1.inputs:
            if indata not in input_names:
                node_inputs.append(dict(name=indata))  ## what about interface and default value ?
            
            # create inputs for all modules parameters and input of first module
        for m in modules:
            for p,v in m.parameters.iteritems():
                if p in input_names: continue
                
                input_names.append(p)
                node_inputs.append(dict(name=p,interface=_interface(v), value=v, hide=(p in m.hidden)))
         
        self.node_inputs = node_inputs
        self.pipeline.input_names = input_names ##
                
    def __call__(self, fct):
        # create the node descriptor
        oanode = dict()
        #oanode['name']    = fct.__name__
        oanode['inputs']  = self.node_inputs
        oanode['outputs'] = [dict(name='pipeline_namespace',interface='IDict')]
        oanode['nodemodule'] = fct.__module__
        oanode['nodeclass']  = fct.__name__ + '.__OA_run__'
        
        self.pipeline.__name__   = fct.__name__
        self.pipeline.__module__ = fct.__module__
        self.kwargs = oanode
        
        return _aleanode.__call__(self, self.pipeline) 
        #self.pipeline._aleanode_ = oanode
        #return self.pipeline
        
class PipelineModule(object):
    """
    Implement a module for a RootAnalysisPipeline object. 
    
    The module has the following attributs:
      - name: 
          The name of the module
      - load:
          The function that load previously computed data
      - compute:
          The function that compute the data from input and parameters
      - suffix:
          The suffix to append to the run base_name argument to form the name of
          the stored data file (both for load and compute)
      
      - inputs: 
          The names of the data inputs (of the compute function)
      - outputs:
          The names of the data outputs
      - parameters:
          The name and value of the module parameters, stored as a dictionary.
          Parameters are special type of inputs for which a default value are 
          provided. They are not required argument for running the module.
          
      - load_param:
          The load function is expected to have only one argument ('filename').
          If other arguments are necessary, they are stored in 'load_param'.
          Note that those parameter are given by construction: they are fixed.
          
      - run: 
          The function that actually run the module functionality (see run doc)
          
    :todo: This should become a decorator...
    """
    def __init__(self, name, load, compute, suffix, outputs, parameters={}, load_kargs={}, hidden=[]):
        """
        :Input:
            name:
                Name of the module
            load:
                The function to call to load already computed data
            compute: 
                The function to call if data needs to be computed or updated
            suffix:
                The suffix to append to the base_name (see the run method) to
                generate output filename
            outputs:
                list of names of the output data
            parameters:
                Optional dictionary of inputs names and replacement default
                values to override defaults value of the parameters of the
                compute functions
            load_kargs:
                Replacement arguments (name,value) pairs for the load function
            hidden:
                list of parameter name that should be hidden by the aleanode
                
        :Note:
            The inputs and parameters are automatically extracted from the load 
            and compute functions. The inputs are the arguments that does not 
            have a default value, and parameters are those with given default 
            values.
            This implies that required functions inputs must be explicitely 
            stated in the function definition (args and kargs are not processed)
            
            In addition to these automatically detected input and parameters, 
            both functions load and compute should be able to process the key
            argument 'filename': the name of the file to load or save data to.
            
            Finally, alternate default values for the functions parameters can 
            be given through this constructor 'parameters' argument.

        :Warning:
            Both load and compute functions should return a tuple of outputs, 
            in particular if they return only one object (which can be iterable)

            Example:
                def compute(filename, question='ultimate', answer=42):
                    import this
                    data = "".join([this.d.get(c, c) for c in this.s])
                    
                    return data,   # note the ','
        """
        self.name = name
        self.load = load
        self.compute = compute
        self.suffix  = suffix
        self.outputs = outputs
        self.hidden  = set(hidden)
        
        def args(f):
            """
            extract arguments name of function f and 
               - input: names of arguments without default value
               - param: name:value dictionary of arguments with default values
            
            remove 'filename' from argument/input/param
            """
            if not isinstance(f, type(lambda x:x)):
                return [], [], {}
            narg = f.func_code.co_argcount
            defv = [] if f.func_defaults is None else f.func_defaults
            ndef = len(defv)
            args = [a for a in f.func_code.co_varnames[:narg] if a<>'filename']
            inputs = [a for a in (args[:-ndef] if ndef else args) if a<>'filename']
            param  = dict(zip(args[-ndef:],defv))
            param.pop('filename', None)
            return args, inputs, param
            
        load_args, load_input, load_param = args(self.load)
        comp_args, comp_input, comp_param = args(self.compute)

        # parameters override function defaults
        load_param.update(load_kargs)
        comp_param.update(parameters)

        self.load_args    = load_args
        self.compute_args = comp_args
        self.parameters   = comp_param
        self.load_kargs   = load_param
        self.inputs = list(set(comp_input))
        
        self.updated = None
        
        
    def run(self, base_name='', update=False, verbose=False, **kargs):
        """
        Run the module.
        
        If prevously computed data can be loaded, retrieve it using the load 
        functions. Otherwise, call this module's compute function.
        
        :Input:
          - base_name:
                Base name for storage data file. For each module, the module 
                suffux is appended to this base_name to form the the name of the
                storage file.
          - update:
                If True, force computation, even of data can be loaded.
          - verbose:
                If True, print current process stage at run time
          
          **kargs:
                the arguments to give to the module compute function (and load)
                But for the storage filename, which is constructed from base_name
                
        :Outputs:
          The computed, or reloaded, data. It should correspond to the names in 
          this module's 'outputs' attribute.
        """
        data = None
        filename = base_name + self.suffix
        
        # try to load the data
        if self.load and not update and _os.path.exists(filename):
            try:
                ##data  = self.load(filename=filename, **dict([(k,param[k]) for k in self.load_args]))
                data  = self.load(filename=filename, **self.load_kargs)
            except:
                _print_error('Error while loading data %s: unreadable file or missing/invalid metadata' % self.name)  ## print error line ?
                
        # compute the data
        if data is None:
            param = self.parameters.copy()
            param.update(kargs)
            
            _print_state(verbose, '  computing '+ self.name)
            data = self.compute(filename=filename, **dict([(k,param[k]) for k in self.compute_args]))
            self.updated = True
        else:
            _print_state(verbose, '  > '+ self.name + ' loaded')
            self.updated = False
            
        return data

    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        cls = self.__class__.__name__
        # return cls + ' ' + self.__module__ + '.' + self.name
        return cls + ' ' + self.compute.__module__ + '.' + self.name 
        

# load image function and module
# ------------------------------
def load_image(filename, *args, **kargs):
    return _normalize_image(_Image(filename,dtype='f',color='gray'))

load_root_image = PipelineModule(name='image', \
                                 load=None,    compute=lambda filename, imagefile: (load_image(filename=imagefile),), \
                                 suffix='',    outputs=['image'])


#### to be (re)moved
##def arabido(ini_file, indices=None, **kargs):
##    from .database import parse_image_db
##    if isinstance(ini_file, basestring):
##        flist, invalid, outdir = parse_image_db(ini_file=ini_file, output='tree')
##    else:
##        flist = ini_file
##    
##    if indices is None: indices = slice(None)
##    imgNum = len(flist[indices])
##    failed = []
##    for i,f in enumerate(flist[indices]):
##        print 'processing (img %d/%d):' %(i+1,imgNum), f.filename
##        try:
##            image_pipeline_arabido(f, **kargs)
##        except Exception as e:
##            print 'image %s failed:' % f, e.message
##            failed.append((f,e))
##            
##    return failed
##     
##
##@_aleanode('mask', 'tree')
##def image_pipeline_no_frame(image, rm_bg, root_max_radius, min_dimension, seed_type, seed_number, to_tree, to_axe, verbose=True):
##    img = image
##    
##    if all(map(hasattr,(image,)*3, ('filename', 'metadata', 'output'))):
##        metadata = image.metadata
##        output   = image.output
##        image    = image.filename
##        
##    if isinstance(image,basestring):
##        _print_state(verbose, 'load image file')
##        img = _normalize_image(_Image(img,dtype='f',color='gray'))
##        
##    # background removal
##    if rm_bg:
##        _print_state(verbose,'remove background')
##        img = _remove_background(img, distance=root_max_radius, smooth=1)
##    
##    # image binary segmentation
##    _print_state(verbose,'segment binary mask')                                      
##    mask = _nd.binary_closing(_segment_root_image(img)>0)
##    mask = _clean_label(_nd.label(mask)[0], min_dim=min_dimension)>0
##    
##    
##    # detect seed:
##    _print_state(verbose,'detect seeds')
##    if seed_type=='seed':
##        seed_map = _detect_seeds(mask=mask, seed_number=seed_number, radius_min=root_max_radius, sort=True)
##    else: # seed_type=='leaves'
##        seed_map = _detect_leaves(mask=mask, image=img, leaf_number=seed_number, root_radius=root_max_radius, sort=True)
##
##    # image linear decomposition
##    _print_state(verbose,'compute mask linear decomposition')
##    sskl, nmap, smap, seed = _linear_label(mask=mask, seed_map=seed_map, compute_segment_map=True)
##    
##    # make "image-graph"
##    _print_state(verbose,'compute graph representation of mask decomposition')
##    im_graph = _image_graph(segment_skeleton=sskl, node_map=nmap, segment_map=smap, seed=seed)
##    
##    # make polyline graph
##    _print_state(verbose,'compute graph of roots')
##    pl_graph = _line_graph(image_graph=im_graph, segment_skeleton=sskl)
##                                                              
##    # extract axial tree
##    _print_state(verbose,'convert graph to axial tree (%d segmens)' % pl_graph.segment.size)
##    tree = _RootAxialTree(node=pl_graph.node, segment=pl_graph.segment, to_tree=to_tree, to_axe=to_axe)
##
##    return mask, tree
##
##
##@_aleanode('tree')
##def image_pipeline_arabido(image, root_max_radius, plate_border, plate_width, min_dimension, plant_number, to_tree, to_axe, smooth=1, leaf_height=[0,0.25], metadata=None, output=None, update=[], verbose=True):
##    import os
##    
##    if all(map(hasattr,(image,)*3, ('filename', 'metadata', 'output'))):
##        metadata = image.metadata
##        output   = image.output
##        image    = image.filename
##        
##    if output is not None:
##        out_mask = output+'_mask.png'
##        out_seed = output+'_seed.png'
##        out_tree = output+'.tree'
##    else:
##        update = ['all']
##        
##    # segment image:
##    # --------------
##    mask = None
##    if 'mask' not in update and 'all' not in update and os.path.exists(out_mask):
##        # try to load mask
##        from ast import literal_eval 
##        import PIL
##        try:
##            info = PIL.Image.open(out_mask).info
##            bbox = literal_eval(info['bbox'])
##            bbox = map(lambda x: slice(*x),bbox)
##            T    = literal_eval(info['T'])
##            T    = _np.array(T)
##            mask = _Image(out_mask,dtype=bool)      
##        except:
##            _print_error('Error while loading mask: unreadable file or missing/invalid metadata')  ## print error line ?
##            
##    if mask is None:
##        update = ['all']  # from here on, update everything
##        
##        # load image
##        if isinstance(image,basestring):
##            _print_state(verbose, 'load image file')
##            image = _normalize_image(_Image(image,dtype='f',color='gray'))
##            
##        if smooth:
##            img = _nd.gaussian_filter(image, sigma=smooth)
##        else:
##            img = image
##        
##        # background removal
##        _print_state(verbose,'remove background')
##        img = _remove_background(img, distance=root_max_radius, smooth=1)
##        
##        # image binary segmentation
##        _print_state(verbose,'segment binary mask')
##        cluster,T,bbox = _segment_root_in_petri_frame(img,plate_border=plate_border, plate_width=plate_width, min_dimension=min_dimension)
##        mask = cluster>0
##        
##        if output is not None:   
##            from PIL.PngImagePlugin import PngInfo
##            dir_mask = os.path.dirname(out_mask)
##            if len(dir_mask) and not os.path.exists(dir_mask):
##                os.makedirs(dir_mask)
##            meta = PngInfo()
##            meta.add_text('bbox', repr([(bbox[0].start,bbox[0].stop),(bbox[1].start,bbox[1].stop)])) 
##            meta.add_text('T',    repr(T.tolist())) 
##            _Image(mask).save(out_mask, dtype='uint8', scale=255, pnginfo=meta)
##        
##    # detect seed:
##    # ------------
##    seed_map = None
##    seed_save_scale = 36  ## jsut for visualisation, auto scaling should be done, with clean_label at loading
##    if 'seed' not in update and 'all' not in update and os.path.exists(out_seed):
##        try:
##            seed_map = _Image(out_seed, dtype='uint8', scale = 1./seed_save_scale)
##        except:
##            _print_error('Error while loading seed_map file') 
##        
##    if seed_map is None:
##        update = ['all']  # from here on, update everything
##        
##        _print_state(verbose,'detect seeds')
##        seed_map = _detect_leaves(mask=mask, image=img[bbox], leaf_number=plant_number, root_radius=int(root_max_radius/4), leaf_height=leaf_height, sort=True) ##
##    
##        if output is not None:   
##            _Image(seed_map).save(out_seed, dtype='uint8', scale=36)
##
##    # compute graph:
##    # --------------
##    pl_graph = None
##    if 'graph' not in update and 'all' not in update and os.path.exists(out_tree):  ## only tree is saved
##        try:
##            pl_graph = _Data.load(out_tree)
##        except:
##            _print_error('Error while loading graph file (actually, the tree file which stores the graph)') 
##            
##    if pl_graph is None:
##        update = ['all']  # from here on, update everything
##        
##        # image linear decomposition
##        _print_state(verbose,'compute mask linear decomposition')
##        sskl, nmap, smap, seed = _linear_label(mask=mask, seed_map=seed_map, compute_segment_map=True)
##        
##        # make "image-graph"
##        _print_state(verbose,'compute graph representation of mask decomposition')
##        im_graph = _image_graph(segment_skeleton=sskl, node_map=nmap, segment_map=smap, seed=seed)
##        
##        # make polyline graph
##        _print_state(verbose,'compute graph of roots')
##        pl_graph = _line_graph(image_graph=im_graph, segment_skeleton=sskl)    
##        
##        # shift graph node position by cropped box left corner
##        pl_graph.node.x[:] += bbox[1].start
##        pl_graph.node.y[:] += bbox[0].start
##        pl_graph.node.position[:,0] = 0
##
##    # extract axial tree:
##    # -------------------
##    tree = None
##    if 'tree' not in update and 'all' not in update and os.path.exists(out_tree):
##        try:
##            tree = _Data.load(out_tree)
##        except:
##            _print_error('Error while loading tree file') 
##    
##    if tree is None:
##        _print_state(verbose,'extract axial tree')
##        tree = compute_tree(pl_graph, to_tree=to_tree, to_axe=to_axe, metadata=metadata, output_file=out_tree, verbose=verbose) 
##    
##    return tree


    
def compute_tree(rg, to_tree, to_axe, metadata=None, output_file=None, verbose=False):
    # extract axial tree
    _print_state(verbose,'extract axial tree')
    tree = _RootAxialTree(node=rg.node, segment=rg.segment, to_tree=to_tree, to_axe=to_axe)

    if metadata    is not None: tree.metadata = metadata
    if output_file is not None: tree.save(output_file)#; print 'tree saved', output_file

    return tree

