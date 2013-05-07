import numpy as _np
import scipy as _sp
import scipy.stats   as _st
import scipy.ndimage as _nd

from ..ndarray.filter import normalize as _normalize


"""
## TODO:
    > cluster_... should be 'cluster' with the only ability to cluster over a set of distrib
    > & make a quick EM_clustering methods
    > add the beta distribution
        - make a hellinger distance function between beta distrib
    > make a creater_distrib (or multivariate_fit) function, with list of 'normal' or 'beta'
      and input data to fit them over (i.e. euqlaly spaced, evenly distributed...)
        - return stats.distrib obj with added method (new.newinstance) 'distance' and 'fit'
        - create a function that add suitable method to a stats.distrib obj
            
    > rewrite the EM (gmm1D) in a general E/M iterative process
        - add a weight param (np.hist can take it directly)
        - take as input initialized distrib such as returned  by the create_distrib
        - make a quick call eg: dist0=['normal','beta']
        - define what method/arguments should be contained by distrib, sp.stats not necessary 
        - what about the "log" in the maximizing the log-likelihood stuff ?
    > make soft EM instead of hard (optional?):
        - use (normalized?) 'expectation' instead of 'cluster'
     
    ...     
     - make a general EM (not just for normal distribution)              
        - make a general (multi)"fit" function            (*)
            . that take a weight parameter array
            . and the distribution(s) to fit
        - make a general "clustering"                     (*)
            . compute the responsabilities   => see "cluster"
            . return either hard  (argmax) or soft (normalized?) clustering
        - make   general expectation / maximization steps (*)
            . expectation: 
                estimate which distrib the individual are in (soft)
                i.e. Z the unobserved latent variable (?) 
            . maximization: 
                estimate best distrib parameters knowing Z => see "fit"
        - make a general hellinger distance function      (*)
        - add other distribution in all (*):  
            . beta

 (*) for all allowable distribution
"""

def gmm1d(data, weights=None,classes=2, max_iter=100, threshold=10**-10, bins='unique', init="linspace",verbose=False, plot=False, logplot=False):
    """
    Apply fit a 1d gaussian mixture model on input data
    
    Input:
    ------
        data:      any numeric array-like object 
        weight:    weight of each data element (None: elements are equivalent)
                   Must have same shape as 'data'
        classes:   the number of gaussian distribution to fit
        max_iter:  maximum number of iteration for the maximization
        threshold: suffisent Hellinger distance between updated normals for the
                   algorithm to stop.
        bins:      method to reduce data size:
                    - 'unique': use all possible value (exact)
                    - an integer 'n': approximate the given data by a set of 'n'
                          regularly spaced numbers
                    - a set of bins such as given to numpy.histogram used to 
                          cluster all data values
        init:    method to produce initial clustering, either:
                    - linspace - cluster between evenly spaced values 
                    - quantile - cluster with same number of elements
                          
        verbose: print the iteration number and current maximum Hellinger   (*)
                 distance between last and new estimated normals
        plot:    plot estimated normals on top of the histogram of data     (*)
        logplot: use a logarithmic scale for plotting                       (*)
        
        (*) require matplotlib
        
    Output:
    -------
        the list of estimated normals as scipy.stats.norm objects
        the list of their relative weight
    """
    from itertools import izip
    
    data = _np.asanyarray(data)
    
    # compress data    
    #Dev note: np.histogram (1) vs bincount(digitize(...)) 
    #   > quick comparison: (1) simpler and faster for large enough array than (2)  
    if bins=='unique': 
        weight, bins = _np.histogram(data,_np.hstack((_np.unique(data),_np.inf)),weights=weights)
        value = bins[:-1]
    else:
        weight, bins = _np.histogram(data,bins,weights=weights)
        value  = (bins[:-1] + bins[1:])/2.
        valid  = weight!=0
        weight = weight[valid]
        value  = value [valid]
        
    # normalize weights
    weight = weight / _np.sum(weight,dtype=float)  
    
    # initialize plotting
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(plot)
        plt.clf()
        if weights is None: w=None
        else: w = weights.flat
        minh = plt.hist(data.flat,100,weights=w,normed=True,log=logplot)[0]
        if logplot: minh = 10**_np.floor(_np.log(minh[minh>0].min())/_np.log(10))
        else:       minh = 0

    # compute the normal distribution of all clusters
    def normal(cluster):
        w    = _nd.sum(weight      ,                    labels=cluster, index=ind)
        mean = _nd.sum(weight*value,                    labels=cluster, index=ind) / w
        std  = _nd.sum(weight*(value-mean[cluster])**2, labels=cluster, index=ind) / w  ## biased std
        return [_st.norm(loc=m, scale=max(s,2**-15)) for (m,s) in izip(mean,std**0.5)]    
        
    # make an initial clustering which all have (approx) the same number of 
    # elements separated by evenly spaced quartiles
    if init=='quantile':
        cs      = _np.cumsum(weight)
        bins    = _np.zeros(classes+1)
        bins[:] = _np.argmin(abs(cs[:,None] - _np.linspace(0,1,classes+1)[None,:]),axis=0)
        
        bins[1:-1] = value[bins[1:-1].astype(int)]
        bins[ 0]   = value.min()
        bins[-1]   = value.max()+10**-10        ## eps constants
    else:
        bins = _np.linspace(value.min(),value.max()+10**-10,classes+1)
        
    cluster    = _np.digitize(value,bins)-1
    
    # initialize mask (threshold at mean value) and normals
    ind     = range(classes)
    normals = normal(cluster)
    
    # iterative likelikehood maximization
    i = 0  # current iteration
    t = 1  # current Hellinger distance (for threshold stopping criterion)
    
    resp = _np.zeros((classes,len(value)))
    while i<max_iter and t>threshold:
        i += 1
        
        # total weight of each class
        w = _nd.sum(weight,labels=cluster, index=ind)    
        
        # responsability (probability) of elements to be part of the each class
        for c,n in enumerate(normals):
            resp[c] = n.pdf(value) * w[c]

        ##resp = _normalize(resp, method='taxicab', axis=0)
        # update classes
        cluster = _np.argmax(resp,axis=0)
        new_n = normal(cluster)
        
        # max hellinger distance between old and new normal 
        t = max(map(normal_distance,normals,new_n))
        normals = new_n
        
        if plot:
            plt.figure(plot)
            for (n,wi) in zip(normals,w):
                pdf = _np.maximum(wi*n.pdf(value), minh)
                plt.plot(value, pdf,'k')
            
        if verbose:
            print 'i: %d, max d=%f' % (i,t)
            
    if max_iter==0:
        w = _nd.sum(weight,labels=cluster, index=ind)
        
    if plot:
        plt.figure(plot)
        pdf = _np.zeros(value.shape)#[:] = 0
        for (n,wi) in zip(normals,w):
            pdf += _np.maximum(wi*n.pdf(value), minh)
        plt.plot(value, pdf,'r')
        
    return normals, w
    

def normal_distance(n1,n2):
    """
    Return the Hellinger distance between 2 normal distribution n1 and n2
    
    n1 & n2 should be scipy.stats.norm objects
    """
    m1 = n1.mean()
    s1 = n1.std()
    m2 = n2.mean()
    s2 = n2.std()
    return 1 - ((2*s1*s2)/(s1**2+s2**2))**0.5 * _np.exp(-((m1-m2)**2) / (4*s1**2+s2**2))

def cluster_gmm1d(data, distribution=None, weights=None, classes=2, bins=256, threshold=10**-10):
    """
    Cluster data elements into the suitable distributions.
    
      1)  cluster_gmm1d(data, normals, weights)
            or
      2)  cluster_gmm1d(data, classes,  threshold=10**-10, bins=256)
    
    1) cluster data elements into one of the given normal distributions
    2) Same but estimated normals distribution on data before clustering 
    
    Input:
        data:    an array-like object of floating numbers 
        
        1) normals: list of normal distribution (scipy.stats.norm objects)
           weights: the weights of each normals distribution
           
        2) classes:   the number of classes to be estimated
           threshold: the stopping threshold for gmm1d (see gmm1d doc)
           bins:      number of bins to approximates data onto. see gmm1d for details
        
    Output:
        an integer array of same shape as data, where the value of each element
        is the indices of the normals it has been clustered into
    """
    if distribution is None:
        distribution, weights = gmm1d(data.ravel(), classes=classes, 
                                      bins=bins, threshold=threshold)
    
    E = expectation(data=data,distribution=distribution, weights=weights)
    return _np.argmax(E,axis=0)
    
def expectation(data, distribution, weights, normalize=False, output=None):
    """
    Return the expectation of each data element to be in the given 'distribution'
    
    Input:
    ------
        data: array-like 
        distribution: a list/tuple of scipy.stats frozen distribution
        weight: the relative weight of each distribution. 
        normalize: if True, return a normalized expectation. For each data 
                   element, the expectation over all distributions sum to one.
                   
    Output:
    -------
        The return matrix has shape Nx[D] where N is the number of distribution 
        and [D] the shape of input data 
    """
    if output is None: E = _np.empty((len(distribution),) + data.shape)
    else:              E = output
    for i,(n,w) in enumerate(zip(distribution,weights)):
        E[i] = w*n.pdf(data)

    if normalize:
        E /= _np.sum(E,axis=0)[_np.newaxis]
        
    return E
    
def outlier(data, maxiter=None):
    """
    Detect iteratively the outlier using Chauvenet's criterion
    
    Input:
    ------
        data:    any numeric array-like object
        maxiter: maximum number of iteration (None means no limite)
        
    Output:
    -------
        a boolean array of the detected outliers
    """
    if maxiter is None: maxiter = _np.inf
    
    n  = _st.norm(loc=data.mean(), scale=data.std())
    c = 0
    i = (n.pdf(data)*data.size)<0.5
    
    iter = 0
    inum = _np.sum(i)
    new  = inum>0
    while new and iter < maxiter:
        d = data[-i]
        n = _st.norm(loc=d.mean(), scale=d.std())
        i = (n.pdf(data)*d.size)<0.5
        
        icur = _np.sum(i)
        new  = icur > inum
        inum = icur
        iter = iter+1
        
    return i
    
