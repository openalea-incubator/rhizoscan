"""
Module to track images of RSA 
"""
import numpy as _np
from scipy import ndimage as _nd

from rhizoscan.workflow import node        as _node
from rhizoscan.workflow import pipeline    as _pipeline
from rhizoscan.opencv   import descriptors as _descriptors
from rhizoscan.image    import Image       as _Image
##from rhizoscan.geometry import translation as _translation


@_node('key_point','descriptor')
def detect_sift(image, mask=None, verbose=True):
    kp, desc = _descriptors.detect_sift(image)
    
    if desc.max()<256:
        desc = _Image(desc)
        desc.set_serializer(pil_format='PNG',ser_dtype='uint8',ser_scale=1,extension='.png')
    elif verbose:
        print '  descriptors cannot be serialized into png'
        
    return kp, desc


## tmp pipeline to process image up to sift detection
from rhizoscan.root.pipeline import load_image as _load_image
##from rhizoscan.root.pipeline import detect_petri_plate as _detect_petri_plate
@_pipeline([_load_image,detect_sift]) ## _detect_petri_plate, ---> use mask??
def _sift_pipeline():
    """ pipeline up to sift detection """
    pass


def transformation_sequence(ds, reference=0, release_image=True, verbose=False):
    """
    Compute the affine transformation for all item in `ds` list
    
    `ds` is a dataset list such as used in root.pipeline.dataset. However, their
    'key_point' and 'descriptor' attribute should be related.
    
    The transformation is computed using the content stored in 'key_point' and 
    'descriptor' attributes. If those are not found in some `ds` item, then 
    `detect_sift` is called given the item 'image' attribute.
    
    The transformation is done with respect to the `ds` item with index given by
    `reference` (i.e. `ds[reference]`). That means the transformation fit each
    item coordinate frame *into* the reference one:
        `position-in-reference = T * position-in-item`
    
    If image are used and if `release_image` is True, then the item 'image' 
    attribute is replace by its loader after use.
    """
    # check for key_point and descriptor in all ds elements, and compute missing  ## remove that?
    for d in ds:
        d.load()
        if not d.has_key('key_point') or not d.has_key('descriptor'):
            if verbose: 
                print 'compute sift on item', d.__key__
            kp, desc = detect_sift(d.image, verbose=verbose-1)
    
            if release_image:
                d.image = d.image.loader()
                
    # image tracking
    r = ds[reference]
    r_kp   = r.key_point
    r_desc = r.descriptor
    for i,d in enumerate(ds):
        if i==reference:
            d.image_transformation = _np.eye(3)
            continue
            
        d_kp   = d.key_point
        d_desc = d.descriptor
        
        if verbose: 
            print 'find affine transfrom on item', d.__key__
        T = _descriptors.affine_match(d_kp,d_desc, r_kp,r_desc, verbose=verbose-1)
        d.image_transform = T
        d.dump()
