import subprocess                           # used by jedit
import inspect                              # used by jedit and print functionnalities
import time                                 # used by tic and toc

__all__ = ['_property','class_or_instance_method', 'static_or_instance_method', 'jedit','tic','toc','static_set','static_get']

# decorators
# ----------
#import functools
import new
class _property(property): pass  # property decorator without doc

class class_or_instance_method(object):
    """
    Decorator that makes a method act as either class or instance method 
    depending on the call. 
    
    Example:
        class A:
            @class_or_instance_method
            def func(cls_or_self):
                print cls_or_self
        
        class B(A): pass
        
        B.func()    =>  print B
        B().func()  =>  print B instance
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        inst_cls = owner if instance is None else instance
        #return functools.partial(self.func, inst_cls)
        return new.instancemethod(self.func,inst_cls,owner)
        
class static_or_instance_method(object):
    """
    Decorator that makes a method act as either a static or instance method
    depending on the call.
    """
    ##TODO: static_or_instance doc: decorator example
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        if instance is None: 
            func = self.func
        else:
            func = new.instancemethod(self.func,instance,instance.__class__)
            #func = functools.partial(self.func,instance)
            #func.__doc__ = self.func.__doc__ 
        return func

# system and path utilities
# -------------------------
def which(program):
    """ 
    Equivalent to the unix which command 
    code taken from http://stackoverflow.com/questions/377017
    """ 
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
    
def abspath(path, base_dir=None):
    """
    Return the absolute path, with given default base directory 
    
    Similar to os.path.abspath, but if the given *path* is not absolute, prepend 
    it with *base_dir* instead of the current directory.
    
    If *base_dir* is None, use current directory (same behavior as os.path.abspath)
    """
    import os
    from os.path import isabs, join, normpath 
    if not isabs(path):
        if base_dir is None: 
            if isinstance(path, unicode):
                base_dir = os.getcwdu()
            else:
                base_dir = os.getcwd()
        path = join(base_dir, path)
    return normpath(path)

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
    



