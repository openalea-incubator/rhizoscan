"""
Generic and unclassified tools

.. currentmodule:: rhizoscan.misc

"""

__all__ = ['jedit','tic','toc','sizeof']

      
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
    import subprocess
    import inspect
    
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
        


# practical printing functionalities
# ----------------------------------
def printMessage(msg,stack=False,header='', color=None):
    """
    print a string. Allows to interspect all written print
        printMessage(msg, stack=False)
    if stack is True, also print the module, function and line where it has been called
    """
    # color system: see http://stackoverflow.com/a/4332587/1206998
    color_map = dict(black=30,red=31,green=32,yellow=33,blue=34,purple=35,cyan=36)
    ESCAPE = '%s[' % chr(27)
    RESET = '%s0m' % ESCAPE
    FORMAT = '1;%dm'
        
    if color:
        print ESCAPE + (FORMAT % color_map[color])
        
    if stack:
        import inspect
        s = inspect.stack()[stack]
        print "%s(%s.%s,l%s):\n  %s " % (header,s[1],s[3],s[2],_message2string_(msg)),
    else:
        print header, _message2string_(msg),
    
    if color:
        print RESET

def printWarning(msg,stack=True):
    """
    print a warning message in blue with (if stack) the module, 
    function name and line where the function has been called
    """
    printMessage(msg=msg,stack=stack*2,header='Warning',color='blue')

def printError(msg,stack=True):
    """
    print a error message in red with (if stack) the module, 
    function name and line where the function has been called
    """
    printMessage(msg=msg,stack=stack*2,header='Error', color='red')

def printDebug(msg=''):
    import inspect
    s = inspect.stack()[1]
    printMessage(msg=msg,stack=2,header='Debug',color='yellow')

def _message2string_(msg):
    if isinstance(msg,basestring): return msg
    elif isinstance(msg,tuple):
        return " ".join(map(str,x))
    else:
        raise TypeError('unrocognized message type')


# tic and toc functionality
# -------------------------
_tictoc = {}
def tic(flag='default'):
    """
    set initialization time for 'flag'
    """
    import time
    _tictoc[flag] = time.time()
    
def toc(flag='default',verbose=False):
    """
    depending on verbose, it prints or returns the time 
    speend since the last call to tic (with same flag)
    """
    import time
    dt = time.time() - _tictoc.get(flag,0)
    if verbose:
        printMessage('time elapsed: ' + str(dt))
    return dt


# misc
# ----
def argsort(x):
    """ return the indices that sort list `x` """
    return sorted(range(len(x)),key=x.__getitem__)
