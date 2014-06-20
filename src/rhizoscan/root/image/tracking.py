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


@_node('key_points','descriptors')
def detect_sift(image, verbose=True):
    kp, desc = _descriptors.detect_sift(image)
    
    if desc.max()<256:
        desc = _Image(desc)##.astype('uint8'))
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


def sequence_transformation(ds, reference=0, verbose=False):
    """
    Compute the affine transformation for all item in `ds` list
    
    `ds` is a dataset list such as used in root.pipeline.dataset containing the 
    attributes 'key_points' and 'descriptors' that store image descriptors 
    
    The transformation is done with respect to the `ds` item with index given by
    `reference` (i.e. `ds[reference]`). That means the transformation fit each
    item coordinate frame *into* the reference one:
        `matched-key-point-in-reference = T * matched-key-point-in-item`
    """
    r = ds[reference].copy().load()
    r_kp   = r.key_points
    r_desc = r.descriptors.astype(_np.float32, copy=False)
    for i,d in enumerate(ds):
        if i==reference:
            d.image_transform = _np.eye(3)
            d.dump()
            continue
            
        d = d.copy().load()
        d_kp   = d.key_points
        d_desc = d.descriptors.astype(_np.float32, copy=False)
        
        if verbose: 
            print 'find affine transfrom on item', d.__key__
        T,M = _descriptors.affine_match(r_kp,r_desc, d_kp,d_desc, verbose=verbose-1)
        d.image_transform = T
        d.dump()
