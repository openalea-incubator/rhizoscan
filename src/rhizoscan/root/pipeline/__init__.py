import scipy.ndimage as _nd
import numpy as _np
import os    as _os
from ast import literal_eval as _literal_eval

_aleanodes_ = []
from rhizoscan.workflow import node as _node # to declare workflow nodesfrom rhizoscan.workflow.openalea  

from rhizoscan.tool import static_or_instance_method as _static_or_instance_method

from rhizoscan.datastructure import save_data as _save
from rhizoscan.datastructure import Data   as _Data
from rhizoscan.image         import Image  as _Image
from rhizoscan.ndarray.measurements import clean_label as _clean_label

from rhizoscan.root.image          import normalize_image             as _normalize_image 
from rhizoscan.root.image          import remove_background           as _remove_background 
from rhizoscan.root.image          import segment_root_image          as _segment_root_image
from rhizoscan.root.image          import segment_root_in_petri_frame as _segment_root_in_petri_frame
from rhizoscan.root.image          import plate                       as _plate
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

class pipeline_node(_node):
    """
    Decorator that create a pipeline and attach it to the given function
    
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
                node_inputs.append(dict(name=p,value=v, hide=(p in m.hidden)))
         
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
        
        return fct##_aleanode.__call__(self, self.pipeline) 
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
                If True, force computation, even if data can be loaded.
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

# petri plate detection, function and module
def detect_petri_plate(image, border_width=.05, plate_size=120, plate_shape='square', smooth=5, gradient_classes=(2,1)):
    fg_mask = plate.detect_foreground(image=image, smooth=smooth, gradient_classes=gradient_classes)
    pmask, px_scale, hull = plate.detect_petri_plate(fg_mask=fg_mask, border_width=border_width,
                                                     plate_size=plate_size, plate_shape=plate_shape)
    return pmaks, px_scale, hull
    
def compute_tree(rg, to_tree, to_axe, metadata=None, output_file=None, verbose=False):
    # extract axial tree
    _print_state(verbose,'extract axial tree')
    tree = _RootAxialTree(node=rg.node, segment=rg.segment, to_tree=to_tree, to_axe=to_axe)

    if metadata    is not None: tree.metadata = metadata
    if output_file is not None: tree.save(output_file)#; print 'tree saved', output_file

    return tree

