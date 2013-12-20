from .results import get_word_files

def _find_end(dataset_name, word_group):
    """ find two consecutive result files that have the same header
    
    This method use a divide-and-conquer approach, but it is not suitable:
    header can be the temporarily the same
    
    Use last_header_diff instead
    """
    word_files = get_word_files(dataset_name,word_group=word_group)
    word_num   = len(word_files)
    
    print 'number of words', word_num
    
    i = word_num-2
    step = i/2
    while i>0 and i<word_num-1 and step>1:
        print i, step
        h1 = _read_header(word_files[i])
        h2 = _read_header(word_files[i+1])
        
        if h1==h2:
            i -= step
        else:
            i += step
        step /=2
        
    return i, word_files[min(i,word_num-1)]
    
def _read_header(filename):
    """
    Return the first 9 lines of file `filename`
    """
    with open(filename) as f:
        h = []
        for i in range(9):
            line = f.readline()
            if i in [1,2,4,5,6,7,8]:
                h.append(line)
    return h
    
def last_header_diff(dataset_name,word_group, rm=False):
    """
    Find the last result file with header different that the previous one
    
    If `rm`, remove all following files
    """
    print 'Load result file list of', dataset_name, word_group
    word_files = get_word_files(dataset_name,word_group=word_group)
    word_num   = len(word_files)
    
    h1 = _read_header(word_files[-1])
    print 'Find last file with diff header in %d result file list' % word_num
    for i in xrange(word_num-2,-1,-1):
        h0 = _read_header(word_files[i])
        if h0<>h1: break
        h1 = h0
        
    if rm:
        proceed = raw_input('removing files (yes to proceed)?')
        if proceed=='yes':
            import os
            for j in xrange(i+2,wod_num-1):
                os.remove(word_files[j])
        
    return i+1, word_files[i+1]


