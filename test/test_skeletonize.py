"""
Test scikits-image skeletonize
"""

def test_skeletonize():
    import numpy as np
    from skimage.morphology import skeletonize
    
    m = ((np.arange(20*30)%7)).reshape(20,30)
    m1 = m>2
    m2 = (m>3)&(m<6)
    
    slices = [slice(1,-1)]*2
    assert (skeletonize(m1)[slices]==m2[slices]).all()