"""
The RhizoScan package is divided in a few subpackages. The higher level 
functionalities related to the analaysis of root system architecture from images
can be found in the :mod:`rhizoscan.root` Package.




.. _api-rhizoscan:

Rhizoscan packages API
----------------------
.. currentmodule:: rhizoscan

.. autosummary::
   :toctree: generated/

   root       - root structure analysis from images
   ndarray    - general nd-array stuff
   image      - tools and data structure related to images (2d gray-scale or 3d color arrays)
   workflow   - low-level data structures
   geometry   - linear algebra algorithm related to (homogeneous) geometry 
   gui        - graphical user interface basics


.. udpate: 09/05/13
"""

_aleanodes_ = []

__version__ = '0.1'
__author__  = 'Julien Diener'
__institutes__  = 'Virtual Plants, INRIA - CIRAD - INRA'
__description__ = 'package for image processing and analysis of root system architecture'
__url__ = 'https://sites.google.com/site/juliendiener/'
__editable__ = 'True'
__icon__  = 'root.png'##'om.png'
__alias__ = []


def get_data_path(local_path=None):
    import sys, os
    mpath = sys.modules[__name__].__file__
    ppath = os.path.sep.join(os.path.dirname(mpath).split(os.path.sep)[:-2])
    
    if local_path: return os.path.join(ppath,'test/data',local_path)
    else:          return os.path.join(ppath,'test/data')
    
