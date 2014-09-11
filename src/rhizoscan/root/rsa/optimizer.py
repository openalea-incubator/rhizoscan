"""
Path merging optimization
"""

def get_optimizer(name):
    if name=='gradient':
        return gradient
    if name=='parse_all':
        return parse_all
    else:
        raise TypeError('optimizer unrecognized: '+ str(name))

def gradient(builder, update_model, constraint=True, **kargs):
    """
    Simple gradient descent optimization 
    """
    _count = 0
    while True:
        value  = builder.value()
        merges = builder.possible_merges(constraint=constraint)
        
        if len(merges):
            merges_value = dict((key,builder.evaluate_merge(merge))
                                    for key,merge in merges.iteritems())
            best = min(merges_value, key=merges_value.get)
            if merges_value[best]<value:
                _count += 1
                merge = merges[best]
                print '%4d:'% _count, '>>', best, len(merges), min(merges_value.values()) ##
                builder = builder.apply_merge(merge=merge, merge_key=best, update=update_model)
                continue
        break
        
    print 'merge number:', _count ##

    return builder
    
 
def _find_next(builder, queue, done, max_value, constraint):
    """ Private method used by optimization algorithm
    
    It updates `queue` with states reachable from `builder`
    
    return list of added state (i.e. queue key)
    """
    done[builder.state()] = (builder.value(),builder)
    
    merges = builder.possible_merges(constraint=constraint)
    added  = []
    
    best = ''
    
    for key,merge in merges.iteritems():
        m_state = merge['state']
        
        if m_state in queue or m_state in done:
            continue
            
        m_value = builder.evaluate_merge(merge)
        
        if m_value<max_value:
            queue[m_state] = dict(builder=builder, value=m_value, 
                                          key=key, merge=merge)
            added.append(m_state)
            if m_value<best:
                best = m_value
            
    return best, added

 
def parse_all(builder, update_model, coef=1, front_width=10, max_iter=1000, constraint=True, verbose=False):
    """ parse all (decreasing) transformation """
    
    # Parse all reachable state, once
    #  - apply iteratively all merge in queue
    #  - store results in 
    #  - all *new* rechable states are stored in 'queue'
    
    queue = {}
    done  = {}
    
    best_value = 'undef' # any float < str
    count = 0
    
    if coef=='best': 
        front_width=1
        coef=1
    
    def find_next(builder, best_value):
        max_value = coef*builder.value()
        best, states = _find_next(builder, queue, done, max_value, constraint)
        if best<best_value:
            best_value = best
        return best_value, states
        

    # init
    best_value, added = find_next(builder, best_value)

    if verbose:
        print 'Optimization start value:', builder.value()##

    # iter
    while len(queue) and count<max_iter:
        sort_queue = sorted(((q['value'],st,q) for st,q in queue.iteritems()))
        
        if front_width and len(queue)>front_width:
            to_process = [st for v,st,q in sort_queue[:front_width]]
            queue = dict((st,queue[st]) for st in to_process)
        else:
            to_process = queue.keys()
        
        for key in to_process:
            processed = queue.pop(key)##sort_queue[0][1])
            
            builder = processed['builder']
            key     = processed['key']
            merge   = processed['merge']
            
            if verbose>1:
                print '%4d (queue:%2d) - best %.8f >>' % (count,len(queue),best_value), key, 'last:', builder.model.history[-1:]
            
            builder = builder.apply_merge(merge_key=key, merge=merge, update=update_model)
            best_value, added = find_next(builder, best_value)
            
        count += 1
    
    
    # return best
    best = dict((state,value) for state,(value,builder) in done.iteritems())
    best = min(best, key=best.get)
    
    if verbose:
        print '  >>> iter: %d, best value: %.12f' % (count,done[best][0])
    
    return done[best][1]
