""" 
Some stuff to add annotation to functions and allow

todo? 
change the general node system to follow the func_annotation as in PEP3107?
  - node renamed to annotate
  - rename outputs by 'return'
  - set input directly as attribute
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

from rhizoscan.tool import _property

# default printing functionality
def _print_state(verbose, msg):
    if verbose: print '  ', msg
def _print_error(verbose, msg):
    if verbose: print '  \033[31m*** %s *** \033[30m' % repr(msg)



class node(object):
    """
    Decorate function with the `__node__` attribute
    
    A node attribute of a function can be access using the following methods:
    
      - `node.get_attribute(fct, name, default=None)`
      - `node.set_attribute(fct, name, value)`
      - `node.run(...)` function call with additional abilities (see `run` doc)
      - `node.copy()` make an independant copy of the node (see doc)

    Decoration data are stored in `__node__` dictionary, but the `get_attribute`
    and `set_attribute` accessors should be used instead of calling it directly
    because they manage automatic attributes formatting and filling. 
      
    The decorator attach any key-value pairs to the given function. However
    specific entries are expected and, if not provided, default value are set by
    the `get/set_attribute` functions. Those entries are: 
    
      - `name`:
            The name of the node. 
            [default: the function `__name__` attribute]
      - `inputs`: 
            A list of dictionaries, each entry being for one of the function`s 
            input. These dictionnay should at least have the key `name`.
            [default: names and default values of the function's API]
      - `outputs`:
            A list of dictionaries, each entry being for one of the function`s 
            outputs.These dictionnay should at least have the key `name`.
            [default: one output named `None`]
      - `doc`:
            A description of the function (as a string). 
            [default: function `__doc__` attribute]
      
    :Notes:
      - At least the names of outputs of the decorated function should be given
        as it cannot be infered automatically (see "outputs arguments" below)
      - the `set_attributes` copies the given attribute as it is given. It only
        assert the format for `inputs'.
      - the `get_attribute` return the required attribute if it exist. Otherwise
        if the attribute is one of the special listed above, default values are
        return **and set**. For other attribute, it return and set `default`.
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
            # non-key arguments are outputs
            kwargs['outputs'] = list(args)
        self.kwargs = kwargs
        
    # call creates a pointer to the functions and attached (key)arguments to it
    def __call__(self,f):
        # attache alea parameter to decorated function
        for key,value in self.kwargs.iteritems():
            node.set_attribute(f, key,value)
            
        for declare_node in _workflows:
            declare_node(f)
            
        return f
        
    @staticmethod
    def is_update_required(function, namespace):
        """
        Return true if one or more of `function` outputs is missing in
        dictionary `namespace`.
        
        `function`is expected to be a decorated `node`
        `namespace` should have `__contain__` method, such as dict object.
        """
        return not all(o['name'] in namespace for o in node.get_attribute(function, 'outputs'))
            
    @staticmethod
    def get_input_arg(function, namespace):
        """
        Return `function` inputs taken from `namespace`, as a dictionary.
        
        :Inputs:
          - `function` is expected to be a decorated `node`
          - `namespace` should be a dictionary or implement `get(key,default)` 
             and `has_key`
        
        :Outputs: 
          - the function inputs as a dictionary taken from `namespace` or, if it
            does not contain required key, takes the default value from 
            `function` node.
          - a list of input names that marked as 'required' in `function` and 
            are missing in `namespace`. 
        """
        inputs  = node.get_attribute(function, 'inputs')
        missing = [i['name'] for i in inputs if i['required'] and not namespace.has_key(i['name'])]
        inputs  = dict((i['name'],namespace.get(i['name'], i['value'])) for i in inputs)
        
        return inputs, missing
        
    @staticmethod
    def run(function, namespace=None, stored_data=[], update=None, verbose=False, **kargs):
        """
        Call `function` with optional output IO functionality
        
        `function` is expected to be a decorated `node`. 
        
        :Input:
          - `function`:
              A function which is expected to be a decorated `node`.
              Otherwise, it is decorated automatically.
          - `namespace`:
              If not None, it should be an object with a dictionary interface   
              (implements `update`, `__contain__`, `get` and `has_key`) and is 
              used to retrieve inputs and store outputs of `function` node.
          - `stored_data`:
              List of data name that are to be stored by `namespace`. In this
              case, `namespace` should implement `set(key,value,store)` method
              (such as datastructure.Mapping) which will be called with 
              `store=True` for the given data name.
          - `update`:
              If None, run the function only if `namespace` does not contain
              the names of `function` outputs.
              Otherwise, always run the function.
          - `verbose`:
                If True, prints current process stage at run time
          
          **kargs:
                the arguments of the decorated function, as key-arguments.
                They overwrite those in `namespace`, if provided.
                
        :Outputs:
            Return updated namespace
        """
        func_name  = node.get_attribute(function, 'name')
        
        if namespace is None: namespace = dict()
        namespace.update(kargs) 
        if update    is None: update = node.is_update_required(function, namespace)
                                         
        # compute the data
        if update:
            _print_state(verbose, 'Running function '+func_name)
            fct = node.get_function(function) # for special node types
            inputs,missing = node.get_input_arg(function, namespace)
            if len(missing):
                raise TypeError(func_name + ": Required argmument '" + "','".join(missing) + "' not found")
            outputs = fct(**inputs)
            outputs = node.format_outputs(function, outputs)
        else:
            outputs = dict((o['name'],namespace.get(o['name'])) for o in node.get_attribute(function, 'outputs'))
        
        if len(stored_data):   ## ... this should be done another way ... 
            for oname, ovalue in outputs.iteritems():
                namespace.set(oname,ovalue,store=oname in stored_data)
            if hasattr(namespace,'dump') and getattr(namespace,'__file_object__',None):
                namespace.dump()
        else:
            namespace.update(outputs)
        return namespace
        
    @staticmethod
    def _run(function, verbose=False, **kargs):
        """
        ##DEPRECATED
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
    def format_outputs(function, outputs):
        """
        Make a dictionary from `outputs` and the name of the node outputs
        
        if outputs is a dictionary-like object, return it
        """
        ##if all(map(hasattr,[outputs]*4, ['keys','values','update','__getitem__'])):
        ##    return outputs
            
        ##todo: raise error if outputs length != node outputs length ?
        out_name = [o['name'] for o in node.get_attribute(function, 'outputs')]
        if len(out_name)==1 or not hasattr(outputs,'__iter__'):
            return {out_name[0]:outputs}
        else:
            return dict(zip(out_name, outputs))
    
    @staticmethod
    def set_attribute(function, attribute, value):
        """
        Set `function` node `attribute` to `value`
        
        Do some automatic formating if attribute = 'inputs' or 'outputs'
        """
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
                    d.setdefault('required',not d.has_key('value'))
                    d.setdefault('value', None)
                value.append(d)
        elif attribute=='outputs':
            val = value
            value = []
            for i,d in enumerate(val):
                if isinstance(d,basestring):
                    d = dict(name=d, value=None)
                else:
                    d.setdefault('name', 'in'+str(i+1)) # just in case...
                    d.setdefault('value', None)
                value.append(d)
        if not hasattr(function, '__node__'): function.__node__ = dict()
        function.__node__[attribute] = value
        
    @staticmethod
    def get_function(function, ntype=None):
        """ return the function to call detpending on the type of `function` """
        if ntype is None: ntype = node.get_attribute(function,'type')
        if   ntype=='function': return function
        elif ntype=='method':   return function.im_func
        elif ntype=='class':    return function.__init__
        elif ntype=='object':   return function.__call__
        
    @staticmethod
    def get_attribute(function, attribute=None, default=None):
        """                                     
        Return the required `attribute` from `function` node
        Or the whole node attribute dictionary if `attribute` is None
        
        If attribute is missing in `function` set it as 'default' and return it
        
        Also, if attribute is 'name', 'inputs', 'outputs' or 'doc' and is 
        missing in `function`, provide automatic values. 
        """
        if attribute is None:
            node.get_attribute(function, 'name')
            node.get_attribute(function, 'inputs')
            node.get_attribute(function, 'outputs')
            node.get_attribute(function, 'doc')
            return function.__node__
        
        if not hasattr(function, '__node__'): 
            function.__node__ = dict()
            
        if not function.__node__.has_key(attribute):
            if attribute=='type':
                import types
                if isinstance(function,types.ClassType):
                    node.set_attribute(function, attribute,'class') 
                elif isinstance(function,types.FunctionType):
                    node.set_attribute(function, attribute,'function')
                elif isinstance(function,types.MethodType):
                    node.set_attribute(function, attribute,'method')
                elif hasattr(function,'__call__'):
                    node.set_attribute(function, attribute,'object') # i.e. callable obj
                else:
                    raise TypeError('Unrecognized type of node ' + str(node))
                
            elif attribute=='name': 
                node.set_attribute(function, attribute,getattr(function,'__name__',default))
                
            elif attribute=='outputs':
                # if not outputs, set it to one name 'None'
                node.set_attribute(function, 'outputs',dict(name='None'))
                
            elif attribute=='doc':
                # if node doesn't have 'doc', take the function's doc
                from inspect import getdoc
                fct = node.get_function(function)
                doc = getdoc(fct)
                if node.get_attribute(function,'type')=='class' and function.__doc__ is not None:
                    doc = function.__doc__ + '\nConstructor:\n' + doc
                node.set_attribute(function, 'doc',doc if doc else '')
                
            elif attribute=='inputs':
                from inspect import getargspec, ismethod
                fct = node.get_function(function)
                argspec = getargspec(fct)
                names = argspec.args
                value = argspec.defaults if argspec.defaults is not None else ()
                noval = len(names)-len(value)
                value = [None]*noval + list(value)

                if node.get_attribute(function,'type') in ['class', '__call__']:
                    names = names[1:]
                    value = value[1:]
                    noval -= 1
                
                value = [dict(name=n, value=v) for n,v in zip(names,value)]
                for i in range(noval):
                    value[i]['required'] = True
                node.set_attribute(function, 'inputs',value)
                
        return function.__node__.get(attribute,default)

    @staticmethod
    def set_input_attribute(function, input_name, **attributes):
        """
        Set `attributes` in the input node attributes of `function` with name `input_name` 
        """
        inputs = node.get_attribute(function, 'inputs')
        in_names = dict((i['name'],k) for k,i in enumerate(inputs))
        
        if not input_name in in_names:
            raise KeyError("The given function does not have input with name %s" % input_name)
        else:
            k = in_names[input_name]
            inputs[k].update(attributes)
            node.set_attribute(function, 'inputs', inputs)
        
    @staticmethod
    def copy(function, **kargs):
        """
        Make a copy of the node `function`, with optional change of node attribute
        
        If `function` is a function, it makes a new function object (which
        links to the same function content) that has an independant set of node
        parameters.
        
        `**kargs` can be used to replace values of node attributes::
        
            @rwf.node('x_square')
            def square(a): return a**2
            
            x_square = node.copy(square, inputs=['x'])
        """
        f = function
        
        # copy function instance
        ntype = node.get_attribute(f,'type')
        if ntype=='function':
            g = _FunctionType(f.func_code, f.func_globals, name=f.func_name, argdefs=f.func_defaults, closure=f.func_closure)
            g.__dict__.update(f.__dict__)
        else:
            from copy import copy
            g = copy(function)
        
        # copy function's methods
        from inspect import ismethod
        for name in [name for name,attrib in g.__dict__.iteritems() if ismethod(attrib) and attrib.im_self is f]:
            setattr(g,name, _MethodType(getattr(g,name).im_func, g))
            
        # copy/replace node attribute
        g.__node__ = g.__node__.copy()
        for key,value in kargs.iteritems():
            node.set_attribute(g,key,value)
        
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
      
    Note that (for now) the function is never called
    
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
            n_inputs = node.get_attribute(n,'inputs')
            n_output = node.get_attribute(n,'outputs')
            n_inputs = dict((ni['name'],ni) for ni in n_inputs)
            missing  = [name for name in n_inputs.keys() if name not in ns_names]       
            pl_inputs.extend([n_inputs[name] for name in missing])
            ns_names.update(missing)
            ns_names.update(no['name'] for no in n_output)
            
        pl_outputs = [dict(name='pipeline_namespace', value=dict())]
        
        node(name=pl_name, outputs=pl_outputs, inputs=pl_inputs)(self)
        ##del self.run #! remove node's run to have Pipeline.run again... 
    
    def __call__(self, *args, **kargs):
        """
        ---OUTDATED---
        Call iteratively the pipeline nodes
        
        ##todo:
            - call self.run(node='all',namespace=None, **kargs) after mergin args into kargs
            - return output with name from node.get_attribute(...)
        """
        inputs = node.get_attribute(self,'inputs')
        
        if len(args):
            in_names = [i['name'] for i in inputs]
            kargs.update(zip(in_names, args))
        
        return self.run(compute='all', namespace=kargs) # return whole namespace (??)
        
    @_property
    def inputs(self):
        return node.get_attribute(self,'inputs')
    @_property
    def outputs(self):
        return node.get_attribute(self,'outputs')
    
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
        
        for i in node.get_attribute(self,'inputs'):
            if not i.get('required',False):
                namespace.setdefault(i['name'],i['value']) 
        
        # execute nodes in pipeline
        for n in self._nodes_to_compute(compute, update, namespace):
            node.run(n, namespace=namespace, stored_data=stored_data, update=1)
                
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
            names  = dict((k,None) for k in namespace.keys())
            if compute=='missing': compute=[]
            for n in self.__pipeline__:
                if node.get_attribute(n,'name') in compute or node.is_update_required(n, names):
                    nodes.append(n)
                    if update:
                        for o in node.get_attribute(n,'outputs'):
                            names.pop(o['name'], None)
                            
        return nodes
    
    def __repr__(self):
        def full_name(x): return x.__module__ + '.' + x.__name__
        cls = full_name(self.__class__)
        fct = ','.join([full_name(f) for f in self.__pipeline__])
        return cls + '([' + fct + '])'
