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
    _node_parser.append(openalea.attributes_parser)
except:
    pass

from types import MethodType   as _MethodType
from types import FunctionType as _FunctionType

from rhizoscan.misc.decorators import _property

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
      - `node.call(...)` function call with additional abilities (see `run` doc)
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
        
    # creates a pointer to the functions and attached (key)arguments to it
    def __call__(self,f):
        # attache alea parameter to decorated function
        for key,value in self.kwargs.iteritems():
            node.set_attribute(f, key,value)
            
        for declare_node in _workflows:
            declare_node(f)
            
        return f
        
    @staticmethod
    def is_update_required(function, namespace, test_input=False):
        """
        Return true if one or more of `function` outputs is missing in
        dictionary `namespace`.
        
        `function`:   a `node` function
        `namespace`:  dict like object (should implement `__contain__`)
        `test_input`: if True, update is required if an input is missing 
        """
        update = not all(o['name'] in namespace for o in node.get_outputs(function))
        if test_input:
            update |= not all(i['name'] in namespace for i in node.get_inputs(function))
        return update
            
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
        inputs  = node.get_inputs(function)
        missing = [i['name'] for i in inputs if i['required'] and not namespace.has_key(i['name'])]
        inputs  = dict((i['name'],namespace.get(i['name'], i['value'])) for i in inputs)
        
        return inputs, missing
        
    @staticmethod
    def run(function, namespace=None, store=[], update=None, verbose=False, **kargs):
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
          - `store`:
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
            Return the function output as a **dictionary**
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
            outputs = dict((o['name'],namespace.get(o['name'])) for o in node.get_outputs(function))
        
        if len(store):   ## ... this should be done another way ... 
            for oname, ovalue in outputs.iteritems():
                namespace.set(oname,ovalue,store=oname in store)
            if hasattr(namespace,'dump') and getattr(namespace,'__file_object__',None):
                namespace.dump()
        else:
            namespace.update(outputs)
            
        return dict((out['name'],namespace[out['name']]) for out in node.get_outputs(function))
        
    @staticmethod
    def format_outputs(function, outputs):
        """
        Make a dictionary from `outputs` and the name of the node outputs
        
        if outputs is a dictionary-like object, return it
        """
        ##if all(map(hasattr,[outputs]*4, ['keys','values','update','__getitem__'])):
        ##    return outputs
            
        ##todo: raise error if outputs length != node outputs length ?
        out_name = [o['name'] for o in node.get_outputs(function)]
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
        """ return the function to call depending on the type of `function` """
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
    def get_inputs(function, default=None):
        """ returns the `function` inputs """
        return node.get_attribute(function, attribute='inputs', default=default)
    @staticmethod
    def get_outputs(function, default=None):
        """ returns the `function` outputs """
        return node.get_attribute(function, attribute='outputs', default=default)
        
    @staticmethod
    def set_input_attribute(function, input_name, **attributes):
        """
        Set `attributes` in the input node attributes of `function` with name `input_name` 
        """
        inputs = node.get_inputs(function)
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
    def __init__(self, nodes, outputs=None):
        self.nodes   = nodes
        self.outputs = outputs
    def __call__(self, function):
        return Pipeline(function=function, nodes=self.nodes, outputs=self.outputs)
        
class Pipeline(object):
    """
    Simple pipeline workflow: execute a sequence of `workflow.node` 
                                        
    A `Pipeline` is made to execute a sequence of `node` in a common namespace
    The nodes inputs and outputs are thus map with each other w.r.t their name. 
    If this is not suitable with original node IO names, node copies can be 
    passed with alternative name using the nodes `copy(...)` method.
    """
    def __init__(self, function, nodes, outputs=None):
        """
        function: the decorated function  ##to remove? (used only be decorator)
        outputs: the outputs of the pipeline. If None, use outputs of last node.
        
        TODO: check unicity of data names (input&output)
        """
        pl_name = function.func_name
        self.__name__ = pl_name
        self.__module__ = function.__module__
        self.__pipeline__ = nodes
        
        inputs = []
        ns_names  = set()
        for n in nodes:
            n_inputs = node.get_inputs(n)
            n_output = node.get_outputs(n)
            n_inputs = dict((ni['name'],ni) for ni in n_inputs)
            missing  = [name for name in n_inputs.keys() if name not in ns_names]       
            inputs.extend([n_inputs[name] for name in missing])
            ns_names.update(missing)
            ns_names.update(no['name'] for no in n_output)
            
        if outputs is None:
            outputs = node.get_outputs(nodes[-1])
        ##pl_outputs = [dict(name='pipeline_namespace', value=dict())]
        
        node(name=pl_name, outputs=outputs, inputs=inputs)(self)
        ##del self.run #! remove node's run to have Pipeline.run again... 
    
    get_inputs  = node.get_inputs
    get_outputs = node.get_outputs
    
    def __call__(self, *args, **kargs):
        """
        ---OUTDATED---
        Call iteratively the pipeline nodes
        """
        inputs = node.get_inputs(self)
        
        if len(args):
            in_names = [i['name'] for i in inputs]
            #todo: assert arg names not in kargs? 
            kargs.update(zip(in_names, args))
        
        outputs = self.run(compute='all', namespace=kargs)
        
        # return suitable outputs
        outputs = [ns[out['name']] for out in self.get_outputs()]
        if len(outputs)==1: return outputs[0]
        else:               return outputs
        
    def run(self, namespace={}, outdated=[], outputs=None, store=[], **kargs):
        """
        Run the "necessary" nodes of the pipeline
        
        :Inputs:
          - `namespace`:
              - If None, initiate run with an empty namespace dictionary
              A dictionary-like object (i.e. implements `keys`, `update` and []) 
              which is used to store and get inputs and outputs of all the nodes
              
          - outdated:
             list of data names (string) that are outdated.
             
          - outputs:
             list of the names (string) expected outputs.
             None means to return the default pipeline outputs.
             
          - `store`:
              List of data name that are to be stored by `namespace`. In this
              case, `namespace` should implement `set(key,value,store)` method
              (such as datastructure.Mapping) which will be called with 
              `store=True` for the given data name.
              
          - **kargs:
              data and parameters which are added to `namespace` before starting 
              the computation and thus are passed to the nodes run method.
              
        :Output:
          Return the **dictionary** of this pipeline outputs
          
        :Called nodes:
          By default only the nodes necessary to compute the pipeline outputs 
          using the variables given in `namespace` are called. For example, if
          those outputs are present in `namespace`, then no node is called.
          The `outdated` (and `outputs`) argument can be used to enforce the 
          recomputation of some data.
          
          In practive, the list of nodes to be called is obtained with the 
          method `nodes_to_call` with `namespace`, `outdated` and `outputs` 
          given as arguments.
          
          See the `node_to_call` documentation for more details.
        """
        # update namespace with kargs and default values
        namespace.update(kargs)
        
        for i in self.get_inputs():
            if not i.get('required',False):
                namespace.setdefault(i['name'],i['value']) 

        # execute nodes in pipeline
        to_call = self.nodes_to_call(namespace=namespace, outdated=outdated, outputs=outputs)
        for nod in to_call:
            if namespace.get('verbose',False):
                print 'running:', nod.__name__
            node.run(nod, namespace=namespace, store=store, update=1)
                
        if outputs is None:
            outputs = (out['name'] for out in self.get_outputs())
        return dict((out,namespace[out]) for out in outputs)
        
    def nodes_to_call(self, namespace={}, outdated=[], outputs=None):
        """
        Return the nodes that needs to be computed w.r.t to given arguments
        
        This function is used by `run` to select which pipeline nodes needs
        to be computed given the same arguments.
        
        By default, this function returns all the nodes that should be run in 
        order to get the pipeline outputs and namespace (missing) items.
        
        If `outdated` is not empty, the respective data as well as all their 
        'descendent' are treated as outdated. Descendant are the output of nodes
        that have them or intermediate descendant as input. 
        Thus a data listed in `outdated` has the same effect as having them 
        **and** their descendant missing in `namespace`. However, depending on 
        the pipeline outputs, they might not be (re)computed.
        
        If not None, `outputs` indicates the name of alternative pipeline output 
        
        :Inputs:
          - namespace:
             dict-like object that the pipeline is executed into
          - outdated:
             list of data names (string) that are outdated.
          - outputs:
             list of the names (string) expected outputs of `Pipeline.run`.
             None means the default pipeline outputs.
             
        :Outputs:
          The list of pipeline node to be computed
        """
        # "propagate" outdated data
        if len(outdated):
            outdated = set(outdated)
            for nod in self.__pipeline__:
                nod_input_name = set(i['name'] for i in node.get_inputs(nod))
                if nod_input_name.intersection(outdated):
                    outdated.update(o['name'] for o in node.get_outputs(nod))
                    
        # valid namespace item
        ns_names = set(namespace.keys()).difference(outdated)

        # find output that need to be computed
        #   i.e. which are not in namespace, or are outdated
        if outputs is None:
            outputs = (o['name'] for o in self.get_outputs())
        data_to_compute = set(outputs).difference(ns_names)
        
        # Construct a dict of (data-name, node that output it)
        #   note: nodes should not share any output name
        node_of = {}
        for nod in self.__pipeline__:
            node_of.update((o['name'],nod) for o in node.get_outputs(nod))
            
        # make the list of nodes to compute
        node_to_computes = []
        for nod in self.__pipeline__[::-1]:
            nod_outputs = [o['name'] for o in node.get_outputs(nod)]
            
            if len(data_to_compute.intersection(nod_outputs)):
                node_to_computes.append(nod)
                data_to_compute.difference_update(nod_outputs)
                missing = set(i['name'] for i in node.get_inputs(nod))
                missing.difference_update(ns_names)
                data_to_compute.update(missing)
        
        return node_to_computes[::-1]
                
    
    def __repr__(self):
        def full_name(x): return x.__module__ + '.' + x.__name__
        cls = full_name(self.__class__)
        fct = ','.join([full_name(f) for f in self.__pipeline__])
        return cls + '([' + fct + '])'
