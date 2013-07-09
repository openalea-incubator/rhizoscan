"""
This module provides tools to easily create openalea nodes from python functions

It contains in particular:

  - the `aleanode` decorator to *mark* a python function as an openalea node
  - the `wrap_package` and `wrap_module` functions that parse packages and 
    modules for *marked* openalea nodes and create the `wralea` repository
  - the `clean_wralea_package` that remove generated `wralea` repository
  
It also include the overloaded classes with corrections and added abilities:

  - `PFuncNode` that overload `openalea.core.node.FuncNode` 
  - `FuncFactory` that overload `openalea.core.node.NodeFactory`
"""

from __future__ import absolute_import

__this_module__ = __name__

try:        
    from openalea.core import NodeFactory                
except: 
    print '\033[31m *** failed to import openalea.core.NodeFactory ***\033[30m'
    NodeFactory = object
    import warnings
    warnings.warn("OpenAlea.core.NodeFactory could not be found. Aleanode won't be defined", ImportWarning)
    
try:        
    from openalea.core.node import FuncNode   
except: 
    print '\033[31m *** failed to import openalea.core.node.FuncNode ***\033[30m'
    FuncNode = object
    import warnings
    warnings.warn("OpenAlea.core.node.FuncNode could not be found. Aleanode won't be defined", ImportWarning)
    
import os, sys, imp, pkgutil


# print color
R = '\033[31m'  # red
G = '\033[32m'  # green
B = '\033[34m'  # blue
K = '\033[30m'  # black

__all__ = ['aleanode', '_aleanode',
           'wrap_module', 'wrap_package', 'create_wrap_folder', 'clean_wralea_package', 
           'Factory','FuncFactory', 'PFuncNode', 'find_interface', 
           'function_node_attributs', 'node_attribute', 'load_module']


class aleanode:
    """
    Class that can be used as a decorator to declare openalea nodes.
    
    This constructor automatically find name of input (the name of fonctions arguments),
    all the function reference needed for contructing the Factory, and its documentation
    
    The main information that is missing are the names and number of outputs 
    
    Arguments of aleanode call can be either:
      1. key arguments of Factory constructor - ex: `outputs=[{'name':'out1'},{...`
      2. dictionaries that are the outputs    - ex: `{'name':'out1'},{'name':'out2'}`
      3. the names (strings) of the outputs   - ex: `'out1', 'out2'`
      
    If any of the 2nd or 3rd (or both) input types are used, then they overwrite 
    the 'outputs' key-arguments.
    
    :Example:
    
        >>> # for simply giving outputs names, use 3.
        >>> @aleanode('x2','y2')
        >>> def f(x,y):  return x**2, y**2
        >>> 
        >>> # for giving the outputs names and iterfaces (ie. openalea types), use 2.
        >>> @aleanode({'name':'x2','interface':'IFloat'},{'name':'y2','interface':'IFloat'})
        >>> def g(x,y):  return float(x)**2, float(y)**2
        >>> 
        >>> # rename inputs and give the name to 2 outputs
        >>> @aleanode(inputs=[{'name':'a'},{'name':'b'}],outputs=[{'name':'a2'},{'name':'b2'}])
        >>> def h(x,y):
        >>>     return x**2, y**2
        
    :Note about installation:
    
      - Declaring openalea nodes using this decorator does not directly create 
        the node. It is the `wrap_package()` function that parse packages and 
        modules for all declared nodes, and create/update wralea file structure.
        
      - For packages, the `__init__.py` file must contains either declared
        aleanode or at least an empty `_aleanodes_` list, in order for the 
        parser to look further into its modules and subpackages.
        
      - The setup.py files of openalea package should be calling wrap_package 
        and declaring the suitable entry points at package installation
    
    :See also: wrap_package
    """
    # creation of the aleanode object - store given (key)arguments
    def __init__(self, *args, **kwargs):
        if len(args):
            # if some arguments are string, covert them the 'name' dictionary valid as output
            args = list(args)
            for i,v in enumerate(args):
                if isinstance(v,basestring): args[i] = {'name':v}
            kwargs['outputs'] = args
        self.kwargs = kwargs
        
    # call creates a pointer to the functions and attached (key)arguments to it
    def __call__(self,f):
        # attache alea parameter to decorated function
        setattr(f,'_aleanode_', self.kwargs)
        
        # add decorated function to the _aleanodes_ attribute of the function module
        mod = sys.modules[f.__module__]
        if not hasattr(mod,'_aleanodes_'):
            mod._aleanodes_ = [f]
        else:
            mod._aleanodes_.append(f)
        return f
        

_aleanode = aleanode

class PFuncNode(FuncNode):
    """ FuncNode that allow automatic update of caption """
                                                                
    def __init__(self, auto_caption=None, *args, **kargs):
        """
        
        :Input:
            same input as FuncNode: 
                `inputs`, `outputs` and `func`
                
            auto_caption: 
              - if None, no automatic update of node caption when called
              - Otherwise:
                  - 'all': automatically add all inputs to caption when called
                  - a number 'n': add only the nth inputs to caption
        """
        FuncNode.__init__(self, *args,**kargs)
        
        if hasattr(self.func,'set_wrapper_node'):
            self.func.set_wrapper_node(self)
        
        if auto_caption is True: auto_caption = slice(None)
        self._auto_caption = auto_caption

    def __call__(self, inputs = ()):
        """ Call function """
        if(self.func):
            if hasattr(self.func, '_aleanode_') and self.func._aleanode_.has_key('auto_caption'):
                self.set_caption(str(inputs[self.func._aleanode_['auto_caption']]))
            return self.func(*inputs)
                                           

##todo: how to instantiate home made FuncNode (eg. that optionally auto update caption)
#   - subclass NodeFactory with upadted instantiate
#   - subclass NodeFactory with upadted __init__ that make a suitable .classobj...
#       => seems complex: module.['ndoeclass_name'] must behave like a Node class
#   - ...
import types
class FuncFactory(NodeFactory):
    """
    Subclass of openalea.core.NodeFactory that instanciate a `PFuncNode`
    
    :BUG: 
        Also overload get_module_node because the NodeFactory method is 
        re-importing the python modules each time a node is instanciated
    """
    def instantiate(self, call_stack=[]):
        """
        Returns a PFuncNode instance.
        
        call_stack: 
            the list of NodeFactory id in call stack (to avoir infinite recursion)
        """

        # The module contains the node implementation.
        module = self.get_node_module()
        classobj = reduce(lambda m,n: getattr(m,n,None),[module]+self.nodeclass_name.split('.'))

        if classobj is None:
            raise Exception("Cannot instantiate '" + \
                self.nodeclass_name + "' from " + str(module))

        # Check inputs and outputs
        if(self.inputs is None):
            sign = sgn.Signature(classobj)
            self.inputs = sign.get_parameters()
        if(self.outputs is None):
            self.outputs = (dict(name="out", interface=None), )


        # Check and Instantiate if we have a functor class
        if((type(classobj) == types.TypeType)
           or (type(classobj) == types.ClassType)):

            _classobj = classobj()
            if callable(_classobj):
                classobj = _classobj

        node = PFuncNode(auto_caption=True, inputs=self.inputs, outputs=self.outputs, func=classobj)
                                                                            
        # Properties
        try:
            node.factory = self
            node.lazy = self.lazy
            if(not node.caption):
                node.set_caption(self.name)
                
            ##name in self.hidden:
            ##    node.set_port_hidden(self, name, state)

            node.delay = self.delay
        except:
            pass

        # to script
        if self.toscriptclass_name is not None :
            node._to_script_func = module.__dict__.get(self.toscriptclass_name, None)

        return node

    def instantiate_widget(self, node=None, parent=None,
                                 edit=False, autonomous=False):
        """ Return the corresponding widget initialised with node """

        # Code Editor
        if(edit):
            from openalea.visualea.code_editor import get_editor
            w = get_editor()(parent)
            try:
                w.edit_module(self.get_node_module(), self.nodeclass_name)
            except Exception, e:
                # Unable to load the module
                # Try to retrieve the file and open the file in an editor
                src_path = self.get_node_file()
                print "instantiate widget exception:", e
                if src_path:
                    w.edit_file(src_path)
            return w

        # Node Widget
        if(node == None):
            node = self.instantiate()

        modulename = self.widgetmodule_name
        if(not modulename):
            modulename = self.nodemodule_name

        # if no widget declared, we create a default one
        if(not modulename or not self.widgetclass_name):

            from openalea.visualea.node_widget import DefaultNodeWidget
            return DefaultNodeWidget(node, parent, autonomous)

        else:
            # load module
            module = load_module(modulename)
            
            ##(file, pathname, desc) = imp.find_module(modulename, 
            ##    self.search_path + sys.path)
            ##
            ##sys.path.append(os.path.dirname(pathname))
            ##module = imp.load_module(modulename, file, pathname, desc)
            ##sys.path.pop()
            ##if(file):
            ##    file.close()

            widgetclass = module.__dict__[self.widgetclass_name]
            return widgetclass(node, parent)

    def get_node_module(self):
        """
        Return the (loaded) python module object
        """
        # *** same behavior as the original openalea code ***
        if not (self.nodemodule_name):
            # By default use __builtin__ module
            import __builtin__
            return __builtin__


        # Test if the module is already in loaded
        if(self.nodemodule_path and self.module_cache
           and not hasattr(self.module_cache, 'oa_invalidate')):
            return self.module_cache


        # *** CHANGES: use __import__ function (see load_module() below) *** 
        ##try:
        nodemodule = load_module(self.nodemodule_name,self.search_path)  # see function def below
        self.module_cache = nodemodule
        self.nodemodule_path = nodemodule.__file__
        return nodemodule
        #except:
        #    sys.path = sav_path
        #    return Factory.get_node_module(self)

Factory = FuncFactory

##class _OLD_Factory(NodeFactory):
##    """
##    Override the `openalea.core.Factory` class, just to override its get_node_module()
##    and get_node_file() methods. 
##    
##    It allows nodes with module name pointing on package modules (i.e. containing dot,
##    as in 'somePackage.someModule'). In particular, this is necessary to have node
##    pointing to modules that are using relative import.
##    
##    ## core.Factory is now updated and this class is no more useful
##    ##   => still some bug on module reloading: see FuncFactory
##    """
##    def get_node_module(self):
##        """
##        Return the (loaded) python module object
##        """
##        # *** same behavior as the original openalea code ***
##        if not (self.nodemodule_name):
##            # By default use __builtin__ module
##            import __builtin__
##            return __builtin__
##
##
##        # Test if the module is already in loaded
##        if(self.nodemodule_path and self.module_cache
##           and not hasattr(self.module_cache, 'oa_invalidate')):
##            return self.module_cache
##
##
##        # *** CHANGES: use __import__ function (see load_module() below) *** 
##        ##try:
##        nodemodule = load_module(self.nodemodule_name,self.search_path)  # see function def below
##        self.module_cache = nodemodule
##        self.nodemodule_path = nodemodule.__file__
##        return nodemodule
##        #except:
##        #    sys.path = sav_path
##        #    return Factory.get_node_module(self)
##        
##        
##    def get_node_file(self):
##        """
##        Return the path of the python module.
##        
##        Override openalea.core.Factory.get_node_file() method to overcome the 
##        limitation of node pointing to packageless module (no dot in module_name)
##        
##        Try to find the file directly by searching the path or, if it don't work,
##        load all parent packages of node module. This second approach is (at least)
##        necessary if a package has more than one path.
##        
##        :Note:
##        Might get the wrong file if the path exist, but some of the module/package 
##        in its parent package actually point to something different (but that 
##        situation would be quite wicked...)
##        ex: pack/mod.py exist but pack/__init__.py contains:
##            import [any-other-module-but-mod] as mod
##        """
##
##        if(self.nodemodule_path):
##            # if already know, return it
##            return self.nodemodule_path
##            
##        elif(self.nodemodule_name):
##            # find module path
##            mod = self.nodemodule_name.split('.')
##            pkg = mod[:-1]
##            mod = mod[-1]
##            
##            # Try first to find the file by searching the path directly: load nothing
##            # -----------------------------------------------------------------------
##            path = self.search_path + sys.path
##            
##            for pk in pkg:
##                # parse the path to find packages
##                # iteratively add package folder name to path, and keep only those that exist 
##                path  = filter(os.path.exists, [os.path.join(p,pk) for p in path])
##                
##            # path to module .py file 
##            path = filter(os.path.exists, [os.path.join(p,mod+'.py') for p in path]+[os.path.join(p,mod,'__init__.py') for p in path])
##
##            # found ya !
##            if len(path):
##                self.nodemodule_path = path[0]
##                return self.nodemodule_path
##                
##            
##            # Otherwise, try to find the file by searching packages and modules
##            # -----------------------------------------------------------------
##            #
##            # If the nodemodule_name is composed (contains '.') then, it loads 
##            # the intermediate packages as I can't find a way to follow descendance
##            # without loading packages: 
##            #   equivalent to import [modname-before-last-'.']
##            #   then look for attribute or package module  [modname-after-last-'.']
##            #
##            # Otherwise, does exactly the same as openalea Factory get_node_file()
##            path = self.search_path + sys.path
##            if len(pkg):
##                modPkg = __import__('.'.join(pkg),globals(),locals(),[''],-1)
##                if hasattr(modPkg,mod):
##                    pathname = modPkg.__getattribute__(mod).__file__
##                else:
##                    (file, pathname, desc) = imp.find_module(mod,pkg.__path__)
##                    if file: file.close()
##            else:
##                (file, pathname, desc) = imp.find_module(mod, path)
##                if(file): file.close()
##
##            self.nodemodule_path = pathname
##
##            return self.nodemodule_path
##               
               
def find_interface(value):
    """ Find the suitable openalea interface for the value type
    
    Current implemented mapping are:
    
        =======  ===========
         type     interface
        =======  ===========
        `int`    `IInt`
        `float`  `IFloat`
        `bool`   `IBool`
        `str`    `IStr`
        `dict`   `IDict`
        `list`   `ISequence`
        `tuple`  `ITuple`
        =======  ===========
    """
    vtype = type(value)
    if   vtype is int:   return 'IInt'
    elif vtype is float: return 'IFloat'
    elif vtype is bool:  return 'IBool'
    elif vtype is str:   return 'IStr'
    elif vtype is dict:  return 'IDict'
    elif vtype is list:  return 'ISequence'
    elif vtype is tuple: return 'ITuple'
    else:                return None


def wrap_module(module, wrapper_path, attrib={}, search_path = [], test=False, verbose=True):
    """
    Retrieve from given module all declared openalea nodes (i.e. functions listed
    in variable `_aleanodes_`). Then generate a wrapper folder that reference them,
    at given `wrapper_path`
    
    `module` can be a loaded module, or a string with the name of the module as
    it is given to import (i.e. containing package dependancies separated by `.`)
    If the module (or parent package) is not in the `sys.path`, additional 
    directories can be provided in argument `search_path`
    
    :See also: `wrap_package`, `create_wrap_folder`
    """
    if isinstance(module,type(sys)):
        modulename = module.__name__
    else:
        modulename = module                           
        module = load_module(modulename, search_path)      
    
    if not hasattr(module,'_aleanodes_'):                         
        return None                                      
                                                            
                                                  
    # create wrapper folder and get wrapper file:
    # -------------------------------------------
    wrapper_file, start_txt, end_txt = create_wrap_folder(wrapper_path)
                                           
                                          
    # Write generated code
    # --------------------                                           
    text = []                                                                      
    ## add comments

    text.append("from " + __this_module__ + " import Factory\n")
    text.append("if '__all__' not in locals(): __all__ = []\n\n")

    # write node attributs
    keys =  ['__name__','__version__','__license__','__author__','__institutes__','__description__','__url__','__editable__','__icon__','__alias__']
    maxL = max([len(k) for k in keys])                        
    for key in keys:
        text.append((key+'\t').expandtabs(maxL+1) + '= ' + repr(attrib.get(key,'')) + '\n')
        
    text.append('\n\n')
    
    
    # for all nodes: test if required, then write suitable code
    for fct in module._aleanodes_:
        node = function_node_attributs(fct, modulename=modulename, search_path=search_path, 
                                            test=test,  verbose=verbose)

        # set arguments order:
        # all base_arg (but last: 'function'), followed by all others
        base_arg =  ['name', 'nodeclass', 'nodemodule', 'search_path', 'function']
        keys = base_arg[:-1] + list(set(node.keys()) - set(base_arg))
        
        
        # write suitable code in file
        var_name = node['name'].replace('.','_')
        text.append(var_name + " = Factory(")      
        for key in keys:
            text.append(key + '=' + repr(node[key]) + ",\n                ")
        text.append(")\n")
        text.append("__all__.append('" + var_name + "')\n\n")


    # concatenate text string lists & write it in wraper file
    # -------------------------------------------------------
    text = ''.join(start_txt + text + end_txt)
    wrap = open(wrapper_file, 'w')
    wrap.write(text)
    wrap.close()
    
    return module._aleanodes_

def function_node_attributs(fct, modulename='', search_path=[], test=False, verbose=False):
    """
    Extract node attributs from function `fct` *labeled* by an `aleanode` decorator
    
    If `test`, create a FactoryNode from the result. If it fails:
      - and `test='raise'` raise the error
      - otherwise, return `None`
    """
    node = fct._aleanode_.copy()
    
    # set suitable reference to the function the node is refering to
    node.setdefault('name',       fct.__name__)
    node.setdefault('nodemodule', modulename)
    node.setdefault('nodeclass',  fct.__name__)
    node.setdefault('search_path',search_path)   ## restrict to path of this module (package) ?

    # in case, fct is not a function but is still callable
    if not hasattr(fct,'__code__') and hasattr(fct,'__call__'): 
        fct = fct.__call__
    
    # if node doesn't have 'inputs', infer it from function
    if not node.has_key('inputs'):
        c = fct.__code__
        in_name = c.co_varnames[:c.co_argcount]
        in_numb = c.co_argcount
        
        import types
        if type(fct)==types.MethodType:
            in_name = in_name[1:]
            in_numb = in_numb-1
            
        def_val = () if fct.__defaults__ is None else fct.__defaults__
        def_val = (None,)*(in_numb - len(def_val)) + def_val
        node['inputs'] = tuple([dict(name=name, value=val, interface=find_interface(val)) for name,val in zip(in_name,def_val)])

    if verbose:
        print "   -> '%s' (%s.%s)" % (node['name'],node['nodemodule'], node['nodeclass'])
    
    # appened function doc to node "description"
    if fct.__doc__: doc = fct.__doc__
    else:           doc = ''
    node['description'] = node.get('description','') + (fct.__doc__ if fct.__doc__ else '') 
    
    # Assert the generated factory:
    # make a Factory from given arguments and load the function from it,
    if test=='raise':
        f = Factory(**node)
        f.get_node_module().__getattribute__(f.nodeclass_name)
    elif test:
        try:
            f = Factory(**node)
            f.get_node_module().__getattribute__(f.nodeclass_name)
        except Exception as e:
            if verbose:
                print "\033[31m Error creating OA node from function: " + fct.__name__ + '\033[30m'
            return None
            
    return node
    
def wrap_package(pkg, pkg_attrib={}, wrap_name=None, wrap_path=None, entry_points=None, entry_name=None, verbose=True):
    """
    Recursively wralea packages for `pkg` and its modules/subpackages
    
    This functions find declared openalea nodes (from `pkg._aleanodes_`) and
    create a parallel depository to contain openalea wrapper modules.
    This repository starts in the folder given by `wrap_path` (see below) or, 
    if None, in a folder created next to the package folder with the same name
    appended by `_wralea`
    
    :Inputs:                                                                 
        - pkg:
            a python package
        
        - pkg_attrib:
            a dictionary that can contain package attributes:: 
            
              __version__        # default:   '0.0.1'
              __license__        # default:   'CeCILL-C'
              __author__         # default:   'OpenAlea consortium'
              __institutes__     # default:   'INRIA/CIRAD/INRA'
              __description__    # default:   ''
              __url__            # default:   'http://openalea.gforge.inria.fr'
              __editable__       # default:   'False'
              __icon__           # default:   ''
              __alias__          # default:   []
                
            These attributes take the value, if provided, following the order
              1) from the package or module treated, if it contains an attributes with same name
              2) from its parent package
              3) from pkg_attrib
              4) or the default written above

        - wrap_name:  
            the name of the wrapper package. By default, take the name
            of the input package to which `_wralea` is appended
            
        - wrap_path:
            Indicate the path to which the wrapper repository starts. 
            By default, use the path of the input package
            
        - entry_name:
            By default, the openalea entry corresponding to the wrapped 
            packaged will have the same name as the package. 
            Otherwise, it can start with the given `entry_name`
            
        - points:
            An optional list to which all wrapped module are appened
                    
        - verbose:
            - if True, or 1, (default) print a list of all module wrapped
            - if 2, also print the list of nodes found in all modules

    :Outputs:
        A list of entry point string `entry-name = wrapped-package` as it is 
        used by the default setup.py installer of openalea packages.           

    :Warning:
        Packages that don't contain aleanode directly (in the __init__.py file),
        must at least contain an empty `__aleanodes__` list for this function to
        parse subpackages and modules:
        
          ** An empty `_aleanodes_` list induces the recursive parsing **
    """
    add_entry = lambda entry,wrap: entry_points.append('%-40s = %s' %(entry,wrap))

    pkg_name = pkg.__name__
    pkg_path = os.path.dirname(pkg.__file__)
    
    # manage input arguments
    if not wrap_name:    wrap_name = pkg_name + "_wralea" 
    if not wrap_path:    wrap_path = pkg_path[:-len(pkg_name)]
    if not entry_name:   entry_name = pkg_name
    if not entry_points: entry_points = []
    
    
    # function that return a package path from base path and package name
    join = lambda path,pack: os.path.join(path,pack.replace('.',os.path.sep))
    
    # wrap package __init__ module
    attrib = node_attribute(pkg, entry_name=entry_name, parent_attrib=pkg_attrib)
    nodes  = wrap_module(pkg, attrib=attrib,wrapper_path=join(wrap_path,wrap_name), verbose=False)
    
    # if it's not a valid aleanode container (i.e. it has no attribute _aleanodes_)
    # quit wrapping recursion
    if nodes is None: return entry_points
    else:             add_entry(entry_name,wrap_name)

    # print which package is treated
    if verbose: 
        print R + pkg_name + K + '   wrapped in:   '+ G + join(wrap_path,wrap_name) + K
    if verbose>1 and len(nodes):
        print ' >  ' + ' ; '.join([fct.__name__ for fct in nodes])

    # treat all modules and subpackage
    for importer, name, ispkg in pkgutil.iter_modules([pkg_path]):
        module   = __import__(pkg_name + '.' + name, globals(), locals(),[''])

        mod_name  = wrap_name  + '.' + name
        mod_entry = entry_name + '.' + name
        
        if ispkg: 
            # wrap subpackage
            entry_points = wrap_package(module,  pkg_attrib  = attrib, 
                         wrap_name  = mod_name,  wrap_path   = wrap_path, 
                         entry_name = mod_entry, entry_points= entry_points,
                         verbose=verbose)
        else:
            # wrap module
            mod_attrib = node_attribute(module, entry_name=mod_entry, parent_attrib=attrib)
            nodes = wrap_module(module,attrib=mod_attrib, wrapper_path=join(wrap_path,mod_name), verbose=False)
            
            if nodes:
                add_entry(mod_entry,mod_name)
                if verbose: 
                    print B + module.__name__ + K + '  wrapped to   ' + G + mod_name + K
                if verbose>1 and len(nodes):
                    print ' >  ' + ' ; '.join([fct.__name__ for fct in nodes])

    return entry_points

def clean_wralea_package(wrap_pkg):
    """
    Remove all wralea created by wrap_package and wrap_module

    From wrap_pkg path, iterate through folder hierarchy and:
      - remove generated content from `__wralea__.py` files
      - remove `*.pyc` files
      - if `__wralea__.py` file is empty and folder don't contains other file 
        then `__init__.py` and `__wralea__.py`, delete the folder
    
    Return the list of subpackage and modules that has content not generated
    by wrap_package.
    
    If `wrap_pkg` does not exist, return None
    """
    # files left
    file_left = []
    
    if isinstance(wrap_pkg,basestring):
        try:
            pkg = __import__(wrap_pkg, fromlist=[''])
        except ImportError:
            return None
    else:
        pkg = wrap_pkg
    pkg_name = pkg.__name__
    pkg_path = os.path.dirname(pkg.__file__)
    
    # function that return a package path from base path and package name
    join = lambda path,pack: os.path.join(path,pack.replace('.',os.path.sep))
    
    # process all modules and subpackage of pkg
    for importer, name, ispkg in pkgutil.iter_modules([pkg_path]):
        module   = __import__(pkg_name + '.' + name, globals(), locals(),[''])

        mod_name  = pkg_name  + '.' + name
        
        if ispkg:
            # clean subpackage (that's also initial modules !)
            sub_left = clean_wralea_package(module)
            file_left.extend(sub_left)
        #else:
            # sub modules are not generated and will be detected below

    # clean package __wralea__
    wralea_file = os.path.join(pkg_path,'__wralea__.py')
    if os.path.exists(wralea_file):
        text = read_wralea(wralea_file)
        if any(map(len,text)):
            text = ''.join(text[0] + text[1])
            wrap = open(wralea_file, 'w')
            wrap.write(text)
            wrap.close()
        else:
            os.remove(wralea_file)

    # clean package __init__
    if os.path.exists(pkg.__file__):
        init = open(pkg.__file__,'r')
        text = init.readlines()
        init.close()
        if len(text)==0:
            os.remove(pkg.__file__)
    
    # check for files left, and remove directory if empty
    files = os.listdir(pkg_path)
    pyc_f = filter(lambda f: f.endswith('.pyc'),files)
    for pf in pyc_f:
        os.remove(os.path.join(pkg_path,pf))
    sys_f = filter(lambda f: f.startswith('.'),files)
    sys_f = filter(lambda f: f not in ['.svn','.git'],sys_f)
    for sf in sys_f:
        os.remove(os.path.join(pkg_path,sf))
    files = os.listdir(pkg_path)
    if len(files)==0:
        # this should not append if left is not empty
        os.rmdir(pkg_path)
    else:
        file_left.extend([os.path.join(pkg_path,f) for f in files])

    return file_left

    

def node_attribute(module, entry_name=None, parent_attrib={}):
    """
    Make openalea node attributs for the given module.
    
    :Inputs:
        - `entry_name`: 
            The entry_name attribute. 
            If not given (None), use the module `__name__` attribute 
    
        - `parent_attrib`:
            Dictionary of the parent package attribute.
            Its content is used when the module does not contains some attribute
    
    :Note:
        An attribute name is choosen: 
          - first from the module attribute if it exist,
          - then in `parent_attrib` if it exist
          - or a default one is taken
          
        Call this function with a basic module to see the default attributs::
        
            >>> make_attrib(os)
    """
    # use list of (key,value) pairs instead of a dictionnary to preserve order
    
    # set default values
    default = {}
    default['__version__'    ] = '0.0.1'                          
    default['__license__'    ] = 'CeCILL-C'                       
    default['__author__'     ] = 'OpenAlea consortium'
    default['__institutes__' ] = 'INRIA/CIRAD/INRA'
    default['__description__'] = ''
    default['__url__'        ] = 'http://openalea.gforge.inria.fr'
    default['__editable__'   ] = 'False'                           
    default['__icon__'       ] = ''
    default['__alias__'      ] = []

    # create attribu dictionary
    #   selected in the order:
    #   value of module attributes => parent attrib value => default value
    attrib = {}
    attrib['__name__'] = entry_name if entry_name is not None else module.__name__
    for key in default.keys():
        attrib[key] = module.__dict__.get(key, parent_attrib.get(key, default[key] ))

    # assert icon path is absolute 
    if len(attrib['__icon__'])>0:
        if not os.path.isabs(attrib['__icon__']):
            exists = os.path.exists
            join = os.path.join
            dirname = os.path.dirname
            icon = attrib['__icon__']
            
            if exists(join(dirname(module.__file__),icon)):
                attrib['__icon__'] = join(dirname(module.__file__),icon)
            else:
                top_pkg = sys.modules[module.__name__.split('.')[0]]
                attrib['__icon__'] = join(dirname(top_pkg.__file__),'icon',icon)
                
    return attrib
    
# start and end lines of generated code
_GEN_TEXT_START = "## ***** automatically generated code **** ##\n"
_GEN_TEXT_END   = "## ******** end of generated code ******** ##\n"
    
def create_wrap_folder(path):
    """
    Create a python wrapper folder for wralea package
    
    A wrap folder contains an empty `__init__.py` and `__wralea__.py`
    
    If any of these files already exist, do not change it
    
    If __wralea__.py exist, read it and retrieve all contents before and after 
    the *generated code* lines::
    
      ## ***** automatically generated code **** ##
                    ...
      ## ******** end of generated code ******** ##
        
    :Input:
      - path: the directory to the wralea folder to create
        
    :Outputs:
      - the filename of the __wralea__.py file
      - the text found before (and including) the "generated code" part (*)
      - the text found after  (and including) the "generated code" part (*)
        
      (*) stored as lists of string
    """
    
    # create folder
    if len(path) and not os.path.exists(path): os.makedirs(path)
    
    # create __init__.py
    file = os.path.join(path,'__init__.py')
    if not os.path.exists(file):
        f = open(file,'w')
        f.close()
        
    # create __wralea__.py
    filename = os.path.join(path,'__wralea__.py')
    if os.path.exists(filename):
        text_start, text_end = read_wralea(filename)
        text_start.append(_GEN_TEXT_START)
        text_end.insert(0,_GEN_TEXT_END)
    else:
        f = open(filename,'w')
        f.close()
        text_start = [_GEN_TEXT_START]
        text_end   = [_GEN_TEXT_END]
    
    return filename, text_start, text_end

def read_wralea(filename):
    """
    Read a `__wralea__.py` file and return all text before and after generated code 
    """
    wrap = open(filename,'r')
    text = wrap.readlines()
    wrap.close()
    
    # find if it contains a generated code already
    # and store preceeding code (or whole file) in text_start
    start = [t.startswith(_GEN_TEXT_START) for t in text]
    if True in start:
        start = start.index(True)
        text_start = text[:start]
        text = text[start:]
    else:
        text_start = text
        text = []

    # find if remaining text contains text_end flag
    # if it does, store following code in text_end
    end   = [t.endswith(_GEN_TEXT_END) for t in text]
    if True in end:
        end = end.index(True) 
        text_end = text[end+1:]
    else:
        text_end = []

    return text_start, text_end

def load_module(module_name, search_path=[]):
    """
    load `module_name`
    
    :Inputs:
      - `module_name`
          string with the name of the module as it is given to import 
          (i.e. containing package dependancies separated by '.')
      - `search_path`
          Additional search directories, given as a list of string, to look for 
          `module_name` (and its parent package) if it is in not in `sys.path`.
    """
    if len(search_path)>0:
        syspath  = sys.path
        sys.path = search_path + sys.path
        
    module = __import__(module_name,globals(),{}, [''], -1)
    
    if len(search_path)>0:
        sys.path = syspath

    return module

