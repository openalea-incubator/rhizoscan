"""
Generic and unclassified tools

.. currentmodule:: rhizoscan.tool

"""

import subprocess                           # used by jedit
import inspect                              # used by jedit and print functionnalities
import time                                 # used by tic and toc
#import functools

__all__ = ['_property','class_or_instance_method', 'static_or_instance_method', 'jedit','tic','toc','static_set','static_get']

# decorators
# ----------
import new
import types
class _property(property): pass  # property decorator without the property doc

class class_or_instance_method(object):
    """
    Decorator that makes a method act as either class or instance method 
    depending on the call. 
    
    :Example:
        >>> class A:
        >>>     @class_or_instance_method
        >>>     def func(cls_or_self):
        >>>         print cls_or_self
        >>> 
        >>> A.func()    # =>  print (class) A
        >>> A().func()  # =>  print A instance
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        inst_cls = owner if instance is None else instance
        ##return functools.partial(self.func, inst_cls)
        ##return new.instancemethod(self.func,inst_cls,owner)
        return types.MethodType(self.func,inst_cls)
        
class static_or_instance_method(object):
    """
    Decorator that makes a method act as either a static or instance method
    depending on the call.
    
    :Example:
        >>> class A:
        >>>     @static_or_instance_method
        >>>     def func(self_or_value):
        >>>         print self_or_value
        >>> 
        >>> A().func()  # =>  print A instance
        >>> A.func(42)  # =>  print 42
    """
    ##TODO: static_or_instance doc: decorator example
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        if instance is None: 
            func = self.func
        else:
            func = types.MethodType(self.func,instance)
            #func = new.instancemethod(self.func,instance,instance.__class__)
            #func = functools.partial(self.func,instance)
            #func.__doc__ = self.func.__doc__ 
        return func
        
# memory size of objects
# ----------------------
def sizeof(obj, ids=None):
    """
    Simple (recursive) estimation of the memory used by `obj` (in bytes)
    
    This function manages standard python types (dict,list,tuple), instance of
    (new type) classes and numpy arrays.
    
    `ids` is a set of object ids which should not be counted. It is used 
    internally to avoid counting twice the same object.
    
    :Note 1: Two objects can share content, which biais their size count.
    In order to get exclusive size count of objects `a` and `b`, do::
    
        ids = set()
        a_size = sizeof(a,ids)  # ids of all parsed objects are added in ids
        b_size = sizeof(b,ids)  # size of objects not already counted in a_size
        
    :Note
    """
    import sys
    
    if ids is None: ids = set()
    if id(obj) in ids: return 0
    
    # count the size of current obj
    if hasattr(obj,'nbytes') and isinstance(obj.nbytes,int): # for numpy arrays 
        s = obj.nbytes 
    else:
        s = sys.getsizeof(obj)
    ids.add(id(obj))
    
    # get all referenced objects
    subobj = []
    if isinstance(obj,dict):
        subobj.extend(obj.values())
    if hasattr(obj,'__dict__'):      
        subobj.extend(obj.__dict__.values())
    if hasattr(obj,'__slots__'):      
        subobj.extend(filter(None,(getattr(obj,attr,None) for attr in obj.__slots__)))
    if isinstance(obj,(list,tuple)): 
        subobj.extend(obj)

    # (recursively) add size of referenced objects 
    for o in subobj:
        s += sizeof(o,ids)
        
    return s

# tool to open file, or source of modules and functions, with jedit
# -----------------------------------------------------------------
def jedit(file=''):
    """
    jedit(input)  : open input with jedit
    
    input can be a filename, a module or a function
    """
    
    if not isinstance(file,basestring):
        if inspect.isbuiltin(file):
            print "Cannot open builtin file: " + str(file)
            return
            
        try:
            line = inspect.getsourcelines(file)[-1]
            file = inspect.getsourcefile(file)
        except:
            print "Unrecognized input arguments: " + str(file)
            return
    else:
        line = 0
    subprocess.Popen(['jedit','-reuseview',file,'+line:'+str(line)])
        


# special printing functionalities
# --------------------------------
def printMessage(msg,stack=False,header=''):
    """
    print a string. Allows to interspect all written print
        printMessage(msg, stack=False)
    if stack is True, also print the module, function and line where it has been called
    """
    if stack:
        s = inspect.stack()[1]
        print "%s(%s.%s,l%s): %s " % (header,s[1],s[3],s[2],_message2string_(msg))
    else:
        print header, _message2string_(msg)

def printWarning(msg,stack=True):
    """
    print a warning message in blue with (if stack) the module, 
    function name and line where the function has been called
    """
    printMessage(msg=msg,stack=stack,header='\033[94m *** Warning')

def printError(msg,stack=True):
    """
    print a error message in red with (if stack) the module, 
    function name and line where the function has been called
    """
    printMessage(msg=msg,stack=stack,header='\033[91m *** Error')

def printDebug(msg=''):
    s = inspect.stack()[1]
    printMessage(msg=msg,stack=True,header='\033[94m ')

def _message2string_(msg):
    if isinstance(msg,basestring): return msg
    elif isinstance(msg,tuple):
        return " ".join(map(str,x))
    else:
        raise TypeError('unrocognized message type')


# tic and toc functionality  (use the static variable functionality, see below)
# -------------------------
def tic(flag='default'):
    """
    set initialization time for 'flag'
    """
    static_set('tic&toc_static:'+ flag,time.time())
    
def toc(flag='default',verbose=True):
    """
    depending on verbose, it prints or returns the time 
    speend since the last call to tic (with same flag)
    """
    t2 = time.time()
    t1 = static_get('tic&toc_static:' + flag)
    if verbose:
        printMessage('time elapsed: ' + str(t2-t1))
    else:
        return t2-t1

def timeit(fct,*args,**kwargs):
    tic('timeit')
    if kwargs.has_key('iter_fct'): 
        iter_fct = kwargs['iter_fct']
        kwargs.pop('iter_fct')
    else:
        iter_fct = 100
    for i in xrange(iter_fct):
        fct(*args,**kwargs);
    t = toc('timeit',verbose=False)
    print 'average time over %d iterations: %f' % (iter_fct, t/iter_fct)

# provide static variable functionality
# -------------------------------------
def static_set(name,value):
    data = __static_data__()
    data[name] = value
    return value
    
def static_get(name):
    data = __static_data__()
    if data.has_key(name):
        return data[name]
    else:
        raise KeyError(name + ' is not an existing static key')
        
def __static_data__(data={}):
    # return (and store) a list of static data
    return data
    



