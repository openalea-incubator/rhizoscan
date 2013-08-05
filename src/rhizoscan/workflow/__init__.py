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

from types import MethodType as _MethodType

class node:
    """
    Class that can be used as a decorator to declare workflow nodes.
    
    A node is a function decorated with additional information in the attribute
    called `__node__`. Any key-value pairs can be attached, but the following 
    are expected, or extracted automatically from the function definition:
    
      - `name`:
            The name of the node. 
            By default, it use the function `__name__`
      - `inputs`: 
            A list of dictionaries, each entry being for one of the function`s 
            input. These dictionnay should at least have the key `name`.
            By default, the names of the function inputs are taken.
      - `outputs`:
            A list of dictionaries, each entry being for one of the function`s 
            outputs.These dictionnay should at least have the key `name`.
            *** the outputs should be given because they cannot be infered ***
      - `doc`:
            A description of the function (as a string). 
            By default, the function documentation is taken.
      
    :Notes:
      - At call, the decorator only attach the given information to the decorated 
        function with minimal processing.
      - At least the names of outputs of the decorated function should be given
        as it cannot be infered automatically.
      - if openalea is installed (if `workflow.openalea` can be imported), then
        additional processing is done for the declared nodes in order to be 
        used as openalea nodes.
    
    :Special arguments for outputs:
      By default, the `node` decorator expect key-arguments, which are attached
      to the declared function. Hoever, if unnamed arguments are given, they are 
      considered to be the outputs descriptors:
      
      1. dictionaries that describes outputs  - ex: `{'name':'out1'},{'name':'out2'}`
      2. the names (strings) of the outputs   - ex: `'out1', 'out2'`
      
    If any of these are used, they overwrite the `outputs` key-args, if given.
    
    :Example:
    
        >>> # to simply give outputs names, use 2.
        >>> @node('x2','y2')
        >>> def f(x,y):  return x**2, y**2
        >>> 
        >>> # to give the outputs names and iterfaces (for openalea), use 1.
        >>> @aleanode({'name':'x2','interface':'IFloat'},{'name':'y2','interface':'IFloat'})
        >>> def g(x,y):  return float(x)**2, float(y)**2
        >>> 
        >>> # rename inputs and give the name to 2 outputs
        >>> @aleanode(inputs=[{'name':'a'},{'name':'b'}],outputs=[{'name':'a2'},{'name':'b2'}])
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
        setattr(f,'__node__', self.kwargs)
        ##f.node_attributes = _MethodType(node_attributes, f)
        
        for declare_node in _workflows:
            declare_node(f)
            
        return f    

def node_attributes(function):
    """
    Update and return `function.__node__` (filling missing entries)
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
        value = (None,)*(len(names)-len(value)) + value
        
        node['inputs'] = [dict(name=n, value=v) for n,v in zip(names,value)]
    
    if not node.has_key('outputs'):
        node['outputs'] = [dict(name='None')]

    # if node doesn't have 'doc', take the function's doc
    if not node.has_key('doc'):
        node['doc'] = inspect.getdoc(function)
            
    for parser in _node_parser:
        parser(function)
    
    return node

