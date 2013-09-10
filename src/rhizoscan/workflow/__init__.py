""" 
For now, only contains the subpackage openalea
"""

_workflows = []
_node_parser = []
try:
    from . import openalea
    _workflows.append(openalea.declare_node)
except:
    pass
try:
    _node_parser.append(openalea.attributes_parser)
except:
    pass

from types import MethodType   as _MethodType
from types import FunctionType as _FunctionType

from rhizoscan.storage import StorageEntry as _StorageEntry

# default printing functionality
def _print_state(verbose, msg):
    if verbose: print '  ', msg
def _print_error(msg):
    print '  \033[31m*** %s *** \033[30m' % repr(msg)



class node(object):
    """
    Class that can be used as a decorator to declare workflow nodes.
    
    A node is a function decorated with the following methods:
    
      - `get_node_attribute(attribute_name, default=None)`
      - `set_node_attribute(attribute_name, value)`
      - `node_attribute()`  to get the full node dictionary 
      - `run(...)` by default, simply call the function (see `run` doc)
      - `copy()` make an independant copy of the node (see doc)

    Decoration data are stored in `__node__` dictionary, but the above accessors
    should be used instead of calling it directly.
      
    Any key-value pairs can be attached as decoration. But specific entries are
    expected and, if not provided, a default value is set. Those entries are: 
    
      - `name`:
            The name of the node. 
            [default: use the function `__name__` attribute]
      - `inputs`: 
            A list of dictionaries, each entry being for one of the function`s 
            input. These dictionnay should at least have the key `name`.
            [default: the names of the function's inputs are taken]
      - `outputs`:
            A list of dictionaries, each entry being for one of the function`s 
            outputs.These dictionnay should at least have the key `name`.
            [default: one output named `None` is set]
      - `doc`:
            A description of the function (as a string). 
            [default: the function `__doc__` attribute is taken]
            
      - `dump`:
            Serializer function used to save outputs in a writable stream.
            The given function should take as two arguments: the outputs of the
            function (dict) and a writable object (e.g. a file open for writing)
            [default: use datastructure.storage default]
      - `load`:
            Deserializer function used to load previously stored outputs.
            The given function should take as argument a readable object
            (such as a file open for reading).
            It should return the stored outputs as a dictionary.
            [default: use datastructure.storage default]
      - `suffix`: 
            suffix used by workflow storage management. [default ''] ##
          
    :IO functionalities:
        The `run` function attached to a `node` function provide the option to 
        load previously stored output data instead of calling the function, and
        to store outputs once computed. The main goal is for workflows to 
        store intermediate data for later re-run.

        The IO operation are done using the `save_fct` and `load_fct` node 
        attributes (which use pickle by default).
        
        See the run method documentation for details
      
    :Notes:
      - The decorator only attach the given information to the decorated 
        function. Default values (etc...) is done by node_attribute accessors.
      - At least the names of outputs of the decorated function should be given
        as it cannot be infered automatically.
      - If openalea is installed (if `workflow.openalea` can be imported), then
        additional processing is done for the declared nodes in order to be 
        usable as openalea nodes.
    
    :`outputs` arguments:
       By default, the `node` decorator expect key-arguments, which are attached
       to the declared function. However, if unnamed arguments are given, they 
       are treated as outputs descriptors:
      
         1. dictionaries that describes outputs  - ex: `{'name':'out1'}
         2. the names (strings) of the outputs   - ex: `'out1'
      
       If any of these are used, they overwrite the `outputs` key-args, if given.
    
    :Example:
    
        >>> # to simply give outputs names, use 2.
        >>> @node('x2','y2')
        >>> def f(x,y):  return x**2, y**2
        >>> 
        >>> # to give the outputs names and iterfaces (for openalea), use 1.
        >>> @node({'name':'x2','interface':'IFloat'},{'name':'y2','interface':'IFloat'})
        >>> def g(x,y):  return float(x)**2, float(y)**2
        >>> 
        >>> # rename inputs and give the name to 2 outputs
        >>> @node(inputs=[{'name':'a'},{'name':'b'}],outputs=[{'name':'a2'},{'name':'b2'}])
        >>> def h(x,y):
        >>>     return x**2, y**2
        
    """
    # creation of the aleanode object - store given (key)arguments
    def __init__(self, *args, **kwargs):
        if len(args):
            # if some arguments are string, convert them the 'name' dictionary valid as output
            args = list(args)
            for i,v in enumerate(args):
                if isinstance(v,basestring): args[i] = {'name':v}
            kwargs['outputs'] = args
        
        # for output storage functionality
        if kwargs.has_key('dump') and kwargs.has_key('load'):
            kwargs['serializer'] = (kwargs['dump'],kwargs['load'])
        else:
            kwargs['serializer'] = None
        
        self.kwargs = kwargs
        
    # call creates a pointer to the functions and attached (key)arguments to it
    def __call__(self,f):
        # attache alea parameter to decorated function
        setattr(f,'__node__', self.kwargs)
        f.run                = _MethodType(self.run,  f)
        f.copy               = _MethodType(self.copy, f)
        f.node_attributes    = _MethodType(self.node_attributes, f)
        f.set_node_attribute = _MethodType(self.set_node_attribute, f)
        f.get_node_attribute = _MethodType(self.get_node_attribute, f)
        
        for declare_node in _workflows:
            declare_node(f)
            
        return f
            
    @staticmethod
    def run(function, storage=None, io_mode=True, verbose=False, **kargs):
        """
        Call the node function with optional output IO functionality
        
        The function arguments should be given as key-arguments
        
        If the function is a "savable" node: its `io_mode` is not False
        If prevously computed data can be loaded, and 'io_mode' is not 'update', 
        retrieve it using the load functions. If it does not work, call this 
        node's function.
        
        :Input:
          - storage:
                A storage entry (see datastructure.storage) or a url (string)
                to/from where is stored/loaded the function's outputs
          - io_mode:
                if False, do not attempt to load data and don't save output:
                          run is the same as calling the function directly.
                if True, try to load `data` before computation and save outputs
                if 'update', save outputs (do not try to load).
                *** if `storage` is None, False is used ***
          - verbose:
                If True, print current process stage at run time
          
          **kargs:
                the arguments of the decorated function, as key-arguments
                
        :Outputs:
            Outputs of the node function, as a dictionary.
        """
        func_attr = function.get_node_attribute
        
        if io_mode is False or storage is None:
            return node._run(function=function, verbose=verbose, **kargs)
        
        # try to load the data
        import os
        data = None
        storage = _StorageEntry.create_entry(storage)
        if io_mode is True:
            try:
                data = node.format_outputs(function,storage.load())
            except:
               _print_error(verbose, 'Error while loading data %s' % self.name)
                
        # compute the data
        if data is None:
            data = node._run(function=function, verbose=verbose, **kargs)
            # save computed data
            if io_mode is True:
                storage.save(dict(data.items()+save_arg.items()))
        elif verbose:
            start_txt = '  output of "'+ fct_name + '" loaded from: '
            start_len = len(start_txt)
            if len(filename)>80-start_len:
                end_txt = '...' + filename[-(77-start_len):]
            else:
                end_txt = filename
            _print_state(verbose, start_txt + end_txt)
            
        return data
        
    @staticmethod
    def _run(function, verbose=False, **kargs):
        """
        Call the node function, and return the function outputs in a dictionary
        
        The function's argument should be given as key-arguments:
        
        >>> @node('out1','out2')
        >>> def fct(a,b=1):
        >>>     return a+b, a*b
        >>>
        >>> out = fct.run(a=6, b=7)
        >>> print out
        >>> # {'out1':13,'out2':42}
        """
        _print_state(verbose, '  running '+ function.get_node_attribute('name', '"unnamed function"'))
        param = function.get_node_attribute('inputs', [])
        param = dict([(p['name'],p['value']) for p in param])
        kargs_key = kargs.keys()
        param.update((name,kargs[name]) for name in param.keys() if name in kargs_key)
        
        return node.format_outputs(function,function(**param))
        
    @staticmethod
    def format_outputs(function, outputs):
        """
        Make a dictionary from `outputs` and the function's node outputs name
        
        if outputs is a dictionary-like object, return it
        """
        if hasattr(outputs, '__getitem__'):
            return outputs
            
        ##todo: raise error if outputs length != node outputs length ?
        out_name = [o['name'] for o in function.get_node_attribute('outputs')]
        if len(out_name)==1 or not hasattr(outputs,'__iter__'):
            return {out_name[0]:outputs}
        else:
            return dict(zip(out_name, outputs))
    
    @staticmethod
    def set_node_attribute(function, name, value):
        function.__node__[name] = value
    @staticmethod
    def get_node_attribute(function, name, default=None):
        if not function.__node__.has_key(name):
            node = function.node_attributes()
        else:
            node = function.__node__
        return node.get(name, default)
    @staticmethod
    def node_attributes(function):
        """
        Update and return `function.__node__` (filling missing entries)
        
        function can be ether a function or an instance of a callable class
        """
        import inspect
        if not hasattr(function,'__node__'):
            function.__node__ = dict()
        node = function.__node__
        
        # set suitable reference to the function the node is refering to
        if not node.has_key('name'):
            node['name'] = function.__name__
        
        if not hasattr(function,'__code__') and hasattr(function,'__call__'):
            function = function.__call__
            
        # if node doesn't have 'inputs', infer it from function
        if not node.has_key('inputs'):
            argspec = inspect.getargspec(function)
            names = argspec.args
            value = argspec.defaults if argspec.defaults is not None else ()
            value = [None]*(len(names)-len(value)) + list(value)
            
            if inspect.ismethod(function):
                names = names[1:]
                value = value[1:]
            
            node['inputs'] = [dict(name=n, value=v) for n,v in zip(names,value)]
        else:
            for i,d in enumerate(node['inputs']):
                if isinstance(d,basestring):
                    d = dict(name=d, value=None)
                    node['inputs'][i] = d
                else:
                    d.setdefault('name', 'in'+str(i+1))
                    d.setdefault('value',None)
        
        if not node.has_key('outputs'):
            node['outputs'] = [dict(name='None')]
    
        # if node doesn't have 'doc', take the function's doc
        if not node.has_key('doc'):
            node['doc'] = inspect.getdoc(function)
                
        for parser in _node_parser:
            parser(function)
        
        return node
        
    @staticmethod
    def copy(fct_node):
        f = fct_node
        
        # copy function object
        g = _FunctionType(f.func_code, f.func_globals, name=f.func_name, argdefs=f.func_defaults, closure=f.func_closure)
        g.__dict__.update(f.__dict__)
        
        # copy methods
        from inspect import ismethod
        for name in [name for name in g.__dict__ if ismethod(getattr(g,name))]:
            setattr(g,name, _MethodType(getattr(g,name).im_func, g))
            
        # copy node attribute dictionary
        g.__node__ = g.__node__.copy()
        
        return g

        
class Pipeline(object):
    """
    Simple pipeline workflow: execute a sequence of `workflow.node` 
    
    (for now) all nodes inputs are map to previous nodes outputs by their name.
    The list of input names that are not outputs of previous nodes defines the 
    inputs of the pipeline.
    """
    def __init__(self, name, modules):
        """
        node in `modules` list are copied 
        """
        modules = [m.copy() for m in modules]
        
        self.__name__ = name
        self.modules = modules ## change to some private name or in __node__ / __pipeline__
        node_inputs = []
        #node_inputs.append(dict(name='image',interface='IStr'))
        #node_inputs.append(dict(name='output',interface='IFileStr'))
        #node_inputs.append(dict(name='metadata',interface='IDict'))
        #node_inputs.append(dict(name='update', interface='ISequence', value=[]))
        #input_names = [n['name'] for n in node_inputs]  # list to keep the order
        
        # create inputs for the inputs of first module
        pl_inputs = []
        pl_names  = set()
        for m in modules:
            m_inputs = m.get_node_attribute('inputs')
            m_output = m.get_node_attribute('outputs')
            m_inputs = dict((mi['name'],mi) for mi in m_inputs)
            missing  = [name for name in m_inputs.keys() if name not in pl_names]       
            pl_inputs.extend([m_inputs[name] for name in missing])
            pl_names.update(missing)
            pl_names.update(mo['name'] for mo in m_output)
            
        pl_outputs = dict(name='pipeline_namespace', value=dict())
        
        node(name=name, outputs=pl_outputs, inputs=pl_inputs)(self)
    
    def __call__(self, *args, **kargs):
        inputs = self.get_node_attribute('inputs')
        in_names = [i['name'] for i in inputs]
        kargs.update(zip(in_names, args))
        
        nspace = dict((i['name'],i['value']) for i in inputs)
        nspace.update(kargs)
        
        verbose = nspace.get('verbose', False)
        
        # execute modules pipeline
        for module in self.modules:
            #update_module = module.name in update or 'all' in update
            mod_inputs = module.get_node_attribute('inputs')
            #mod_output = module.get_node_attribute('outputs')
            mod_inames = [i['name'] for i in mod_inputs]
            outputs = module.run(**nspace) #dict((name,nspace[name]) for name in mod_inames))
            nspace.update(outputs)
                
        return nspace
    
    ##def run(self, io_mode='default', force_update=[], verbose=False, **kargs):
    ##    pass

