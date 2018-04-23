"""
Some tools to process path, file name etc...
"""


import string
import re
import os


# system and path utilities
# -------------------------
def which(program):
    """ 
    Equivalent to the unix which command 
    code taken from http://stackoverflow.com/questions/377017
    """ 
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
    from os.path import isabs, join, normpath 
    if not isabs(path):
        if base_dir is None: 
            if isinstance(path, unicode):
                base_dir = os.getcwdu()
            else:
                base_dir = os.getcwd()
        path = join(base_dir, path)
    return normpath(path)

def assert_directory(filename):
    """
    Create firectory of `filename`if it does not exist
    """
    d = os.path.dirname(filename)
    if len(d) and not os.path.exists(d):
        os.makedirs(d)

# parse strings using a format-type of pattern
# --------------------------------------------
_def_re   = '.+'
_int_re   = '[0-9]+'
_float_re = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' # http://www.regular-expressions.info/floatingpoint.html

_spec_char = '[\^$.|?*+()'
    
def format_parse(text, pattern):
    """
    Scan `text` using the string.format-type `pattern`
    
    If `text` is not a string but iterable return a list of parsed elements
    
    All format-like pattern cannot be process:
      - variable name cannot repeat (even unspecified ones s.t. '{}_{0}')
      - alignment is not taken into account
      - only the following variable types are recognized:
           'd' look for and returns an integer
           'f' look for and returns a  float
           
    Examples::
        
        res = parse('the depth is -42.13', 'the {name} is {value:f}')
        print res
        print type(res['value'])
        # {'name': 'depth', 'value': -42.13}
        # <type 'float'>
        
        print 'the {name} is {value:f}'.format(**res)
        # 'the depth is -42.130000'
              
        # Ex2: without given variable name and and invalid item (2nd)
        versions = ['Version 1.4.0', 'Version 3,1,6', 'Version 0.1.0']
        v = parse(versions, 'Version {:d}.{:d}.{:d}')
        # v=[{0: 1, 1: 4, 2: 0}, None, {0: 0, 1: 1, 2: 0}]
    """
    # convert pattern to suitable regular expression & variable name
    v_int = 0   # available integer variable name for unnamed variable 
    cur_g = 0   # indices of current regexp group name 
    n_map = {}  # map variable name (keys) to regexp group name (values)
    v_cvt = {}  # (optional) type conversion function attached to variable name
    rpattern = '^'    # stores to regexp pattern related to format pattern        
              
    for txt,vname, spec, conv in string.Formatter().parse(pattern):
        # process variable name
        if len(vname)==0:
            vname = v_int
            v_int += 1
        if vname not in n_map:
            gname = '_'+str(cur_g)
            n_map[vname] = gname
            cur_g += 1                   
        else:    
            gname = n_map[vname]
        
        # process type of required variables 
        if   'd' in spec: vtype = _int_re;   v_cvt[vname] = int
        elif 'f' in spec: vtype = _float_re; v_cvt[vname] = float
        else:             vtype = _def_re;
        
        # check for regexp special characters in txt (add '\' before)
        txt = ''.join(map(lambda c: '\\'+c if c in _spec_char else c, txt))
        
        rpattern += txt + '(?P<'+gname+'>' + vtype +')'
                                               
    rpattern += '$'
        
    # replace dictionary key from regexp group-name to the variable-name 
    def map_result(match):
        if match is None: return None
        match = match.groupdict()
        match = dict((vname, match[gname]) for vname,gname in n_map.iteritems())
        for vname, value in match.iteritems():
            if vname in v_cvt:
                match[vname] = v_cvt[vname](value)
        return match
    
    # parse pattern
    if isinstance(text,basestring):
        match = re.search(rpattern, text)
        match = map_result(match)
    else:
        comp  = re.compile(rpattern)
        match = map(comp.search, text)
        match = map(map_result, match)
            
    return match


