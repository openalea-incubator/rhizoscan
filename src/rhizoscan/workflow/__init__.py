""" 
Some stuff to add annotation to functions and allow

todo? 
change the general node system to follow the func_annotation as in PEP3107
  - node renamed to annotate
  - rename outputs by 'return'
  - set input directly as entry
        => what about conflict with other annotated name!!!
  - annotate decorator only add `func_annotation`to functions (to method.im_func!)
  - add annotate_class: 
        => annotate __init__ but with class name
        => (&find annotated methods?)
  - and annotate_method (or use annotated class parsing?)
  - ...
  
OR

change node to simply annotate function(...) and use get/set _functions_
  - don't attach get/set/copy to annotated function
  - keep get/set/... as static
      => rename get/set as g/set_attribute - in use: node.get_attribute(....)
  - add class_node decorator:
      => annotate class __init__ but with class `name`
      => look for annotated method (which would only be their im_func)
           rename [class_name].--- => what about oa finding it?
           what about declaration which is called by node? => make a method_node decorator without?
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

from rhizoscan.storage import create_entry as _create_entry

# default printing functionality
def _print_state(verbose, msg):
    if verbose: print '  ', msg
def _print_error(verbose, msg):
    if verbose: print '  \033[31m*** %s *** \033[30m' % repr(msg)



class node(object):
    """
    Class that can be used as a decorator to declare workflow nodes.
    
    A node is a function decorated with the following methods:
    
      - `get_node_attribute(name, default=None)`
      - `set_node_attribute(name, value)`
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
        
        self.kwargs = kwargs
        
    # call creates a pointer to the functions and attached (key)arguments to it
    def __call__(self,f):
        # attache alea parameter to decorated function
        print f, type(f)
        setattr(f,'__node__', dict())   
        f.run                = _MethodType(self.run,  f)
        f.copy               = _MethodType(self.copy, f)
        f.set_node_attribute = _MethodType(self.set_node_attribute, f)
        f.get_node_attribute = _MethodType(self.get_node_attribute, f)
        
        for key,value in self.kwargs.iteritems():
            f.set_node_attribute(key,value)
            
        for declare_node in _workflows:
            declare_node(f)
            
        return f
            
    @staticmethod
    def run(function, entry=None, io_mode=True, verbose=False, **kargs):
        """
        Call the node function with optional output IO functionality
        
        The function arguments should be given as key-arguments
        
        If the function is a "savable" node: its `io_mode` is not False
        If prevously computed data can be loaded, and 'io_mode' is not 'update', 
        retrieve it using the load functions. If it does not work, call this 
        node's function.
        
        :Input:
          - entry:
                A storage entry (see storage module) or a url (string)
                to/from where is stored/loaded the function's outputs
          - io_mode:
                if False, do not attempt to load data and don't save output:
                          run is the same as calling the function directly.
                if True, try to load `data` before computation and save outputs
                if 'update', save outputs (do not try to load).
                *** if `entry` is None, False is implied ***
          - verbose:
                If True, print current process stage at run time
          
          **kargs:
                the arguments of the decorated function, as key-arguments
                
        :Outputs:
            Outputs of the node function, as a dictionary.
        """
        func_name  = function.get_node_attribute('name')
        serializer = function.get_node_attribute('serializer')
        
        if io_mode is False or entry is None:
            return node._run(function=function, verbose=verbose, **kargs)
        
        # try to load the data
        import os
        data = None
        entry = _create_entry(entry, serializer=serializer)
        if io_mode is True and entry.exist():
            try:
                data = node.format_outputs(function,entry.load())
            except:
                _print_error(verbose, 'Error while loading data %s' % func_name)
                
        # compute the data
        if data is None:
            data = node._run(function=function, verbose=verbose, **kargs)
            # save computed data
            if io_mode is True:
                entry.save(data)
        elif verbose:
            start_txt = 'loading ' + func_name + ' output from: '
            start_len = len(start_txt)
            url = entry.url
            if len(url)>80-start_len:
                end_txt = '...' + url[-(77-start_len):]
            else:
                end_txt = url
            _print_state(verbose, start_txt + end_txt)
            
        return data
        
    @staticmethod
    def _run(function, verbose=False, **kargs):
        """
        Call the node function, and return the function outputs as a dictionary
        
        The function's argument should be given as key-arguments:
        
        >>> @node('out1','out2')
        >>> def fct(a,b=1):
        >>>     return a+b, a*b
        >>>
        >>> out = fct.run(a=6, b=7)
        >>> print out
        >>> # {'out1':13,'out2':42}
        """
        _print_state(verbose, 'running '+ function.get_node_attribute('name', '"unnamed function"'))
        param = function.get_node_attribute('inputs', [])
        kargs_key = kargs.keys()
        
        required = [p['name'] for p in param if p.get('required',False)]
        missing = [req_input for req_input in required if req_input not in kargs_key]
        if len(missing):
            raise TypeError("Required argmument '" + "','".join(missing) + "' not found")
        
        param = dict([(p['name'],p['value']) for p in param])
        param.update((name,kargs[name]) for name in param.keys() if name in kargs_key)
        
        return node.format_outputs(function,function(**param))
        
    @staticmethod
    def format_outputs(node, outputs):
        """
        Make a dictionary from `outputs` and the name of the node outputs
        
        if outputs is a dictionary-like object, return it
        """
        ##if all(map(hasattr,[outputs]*4, ['keys','values','update','__getitem__'])):
        ##    return outputs
            
        ##todo: raise error if outputs length != node outputs length ?
        out_name = [o['name'] for o in node.get_node_attribute('outputs')]
        if len(out_name)==1 or not hasattr(outputs,'__iter__'):
            return {out_name[0]:outputs}
        else:
            return dict(zip(out_name, outputs))
    
    @staticmethod
    def set_node_attribute(node, attribute, value):
        if attribute=='inputs':
            # assert inputs value format:
            #  - a dictionary which contain at least 'name' & 'value'
            #  - if no values is given, a 'required' flag is set
            val = value
            value = []
            for i,d in enumerate(val):
                if isinstance(d,basestring):
                    d = dict(name=d, value=None, required=True)
                else:
                    d.setdefault('name', 'in'+str(i+1)) # just in case...
                    if not d.has_key('value'):
                        d['value']    = True
                        d['required'] = True
                    d.setdefault('value',None)
                value.append(d)
        node.__node__[attribute] = value
        
    @staticmethod
    def _get_node_function(node, ntype=None):
        """ return type of `node`, and the function to call """
        if ntype is None: ntype = node.get_node_attribute('type')
        if   ntype=='function': return node
        elif ntype=='method':   return node.im_func
        elif ntype=='class':    return node.__init__
        elif ntype=='object':   return node.__call__
        
    @staticmethod
    def get_node_attribute(node, attribute=None, default=None):
        """
        Return the required `attribute` from `node`
        Or the whole attribute dictionary of the node if `attribute` is None
        """
        if attribute is None:
            node.get_node_attribute('name')
            node.get_node_attribute('inputs')
            node.get_node_attribute('outputs')
            node.get_node_attribute('doc')
            return node.__node__
            
        if not node.__node__.has_key(attribute):
            if attribute=='type':
                import types
                print node, type(node)
                if isinstance(node,types.ClassType):
                    node.set_node_attribute(attribute,'class') 
                elif isinstance(node,types.FunctionType):
                    node.set_node_attribute(attribute,'function')
                elif isinstance(node,types.MethodType):
                    node.set_node_attribute(attribute,'method')
                elif hasattr(node,'__call__'):
                    node.set_node_attribute(attribute,'object') # i.e. callable obj
                else:
                    raise TypeError('Unrecognized type of node ' + str(node))
                
            elif attribute=='name': 
                node.set_node_attribute(attribute,getattr(node,'__name__',default))
                
            elif attribute=='outputs':
                # if not outputs, set it to one name 'None'
                node.set_node_attribute('outputs',dict(name='None'))
                
            elif attribute=='doc':
                # if node doesn't have 'doc', take the function's doc
                from inspect import getdoc
                doc = getdoc(function)
                if node_type=='class' and node.__doc__ is not None:
                    doc = node.__doc__ + '\nConstructor:\n' + doc
                node.set_node_attribute('doc',doc if doc else '')
                
                
            elif attribute=='inputs':
                from inspect import getargspec, ismethod
                argspec = getargspec(function)
                names = argspec.args
                value = argspec.defaults if argspec.defaults is not None else ()
                noval = len(names)-len(value)
                value = [None]*noval + list(value)
                                                                  
                if node_type in ['class', '__call__']:
                    names = names[1:]
                    value = value[1:]
                    noval -= 1
                
                value = [dict(name=n, value=v) for n,v in zip(names,value)]
                for i in range(noval):
                    value[i]['required'] = True
                node.set_node_attribute('inputs',value)
                
        return node.__node__.get(attribute,default)

    @staticmethod
    def copy(fct_node, **kargs):
        """
        Make a copy of the node `fct_node`, with optional change of node param.
        
        If the `fct_node` is a function, it makes a new function object (which
        links to the same function content) that has an independant set of node
        parameters.
        
        Use `**kargs` to give replacement values for the node parameters
        """
        f = fct_node
        
        # copy function instance
        if isinstance(fct_node,_FunctionType):
            g = _FunctionType(f.func_code, f.func_globals, name=f.func_name, argdefs=f.func_defaults, closure=f.func_closure)
            g.__dict__.update(f.__dict__)
        else:
            from copy import copy
            g = copy(fct_node)
        
        # copy function's methods
        from inspect import ismethod
        for name in [name for name,attrib in g.__dict__.iteritems() if ismethod(attrib) and attrib.im_self is fct_node]:
            setattr(g,name, _MethodType(getattr(g,name).im_func, g))
            
        # copy/replace node attribute
        g.__node__ = g.__node__.copy()
        for key,value in kargs.iteritems():
            g.set_node_attribute(key,value)
        
        return g

class class_node(node):
    def __call__(self, cls):
        node.__call__(self,cls)
        ##...


class pipeline(object):
    """
    Decorator to make a Pipeline from a function::
    
      @pipeline(--kargs--)
      def some_pipeline(): pass
 
    is the same as::

      def some_pipeline(): pass
      some_pipeline = Pipeline(some_pipeline, --kargs--)
      
    Note that (for now) the function is not use
    
    See `Pipeline` documentation
    """
    def __init__(self, nodes):
        self.nodes = nodes
    def __call__(self, function):
        return Pipeline(function=function, nodes=self.nodes)
        
class Pipeline(object):
    """
    Simple pipeline workflow: execute a sequence of `workflow.node` 
                                        
    A `Pipeline` is made to execute a sequence of `node` in a common namespace
    The nodes inputs and outputs are thus map with each other w.r.t their name. 
    If this is not suitable with original node IO names, node copies can be 
    passed with alternative name using the nodes `copy(...)` method.
    """
    def __init__(self, function, nodes):
        """
        function: the decorated function
        """
        pl_name = function.func_name
        self.__name__ = pl_name
        self.__module__ = function.__module__
        self.__pipeline__ = nodes
        
        pl_inputs = []
        ns_names  = set()
        for n in nodes:
            n_inputs = n.get_node_attribute('inputs')
            n_output = n.get_node_attribute('outputs')
            n_inputs = dict((ni['name'],ni) for ni in n_inputs)
            missing  = [name for name in n_inputs.keys() if name not in ns_names]       
            pl_inputs.extend([n_inputs[name] for name in missing])
            ns_names.update(missing)
            ns_names.update(no['name'] for no in n_output)
            
        pl_outputs = [dict(name='pipeline_namespace', value=dict())]
        
        node(name=pl_name, outputs=pl_outputs, inputs=pl_inputs)(self)
        del self.run #! remove node's run to have Pipeline.run again... 
    
    def __call__(self, *args, **kargs):
        """
        Call iteratively the pipeline nodes
        
        ##todo:
            - call self.run(node='all',namespace=None, **kargs) after mergin args into kargs
            - return output with name from self.get_node_attribute(...)
        """
        inputs = self.get_node_attribute('inputs')
        in_names = [i['name'] for i in inputs]
        kargs.update(zip(in_names, args))
        
        nspace = dict((i['name'],i['value']) for i in inputs)
        nspace.update(kargs)
        
        verbose = nspace.get('verbose', False)
        
        # execute nodes pipeline
        for node in self.__pipeline__:
            node_inputs = node.get_node_attribute('inputs')
            node_inames = [i['name'] for i in node_inputs]
            outputs = node.run(**nspace)
            nspace.update(outputs)
                
        return nspace
    
    def run(self, compute='missing', update=True, namespace=None, stored_data=[], **kargs):
        """
        Run the pipeline nodes to fill `namespace` with te nodes outputs
        
        :Inputs:
          - `compute`:
              - If 'all', compute all pipeline nodes
              - If 'missing' (default) compute the nodes for which the outputs
                are missing in `namespace`. -Those are always computed-
              - If a list of node name: compute those nodes.
          - `update`:
              - If False, compute only the nodes given by the `compute` argument
              - If True, compute also the nodes for which some of the inputs 
                were updated by previous nodes.
          - `namespace`:
              - If None, initiate run with an empty namespace dictionary
              - Otherwise, it should be an object with a dictionary interface
                (i.e. implements `keys`, `update` and `__getitem__`) and is used
                to store/retrieve inputs and outputs of the pipeline nodes.
          - `stored_data`:
              List of data name that are to be stored by `namespace`. In this
              case, `namespace` should implement `set(key,value,store)` method
              (such as datastructure.Mapping) which will be called with 
              `store=True` for the given data name.
              
          - **kargs:
              data and parameters which are added to `namespace` before starting 
              the computation and thus are passed to the nodes run method.
              
        :Output:
          Return the updated `namespace`
        """
        if namespace is None:
            namespace = dict()
            
        namespace.update(kargs)
        
        for i in self.get_node_attribute('inputs'):
            if not i.get('required',False):
                namespace.setdefault(i['name'],i['value']) 
        
        # execute nodes in pipeline
        for node in self._nodes_to_compute(compute, update, namespace):
            in_names = set(i['name'] for i in node.get_node_attribute('inputs'))
            in_names.add('verbose')
            outputs = node.run(**dict((name,namespace[name]) for name in in_names if name in namespace))
            if len(stored_data)==0:
                namespace.update(outputs)
            else:
                for name,value in outputs.iteritems():
                    namespace.set(name,value,store=name in stored_data)
                
        return namespace
        
    def _nodes_to_compute(self,compute, update, namespace):
        """
        Find the list of nodes to compute w.r.t run method arguments
        
        compute:   same values as the run method `compute` argument
        udpate:    same values as the run method `update` argument
        namespace: initial namespace the pipeline will run with. Same as the
                   run method `namespace` argument to which kargs has been added
        """
        if compute=='all':     
            nodes = self.__pipeline__
        else:
            # check missing outputs names of pipeline nodes
            nodes = []
            names  = set(namespace.keys())
            if compute=='missing': compute=[]
            for node in self.__pipeline__:             
                node_output = [o['name'] for o in node.get_node_attribute('outputs')]
                                                         
                if not all([output in names for output in node_output]) \
                or node.get_node_attribute('name') in compute:
                    nodes.append(node)
                    if update:
                        names.difference_update(node_output)
        
        return nodes
