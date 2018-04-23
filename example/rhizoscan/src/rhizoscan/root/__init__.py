"""
Analysis of Root System Architecture from images

Tutorials:
----------
The :ref:`User Manual<rhizoscan_manual>` provides example of use of the end-user 
functionalities, such as:

  - :ref:`how to extract Root System Architecture from image using python<rhizoscan_script_tuto>`
  - :ref:`how to extract Root System Architecture from image with visualea<rhizoscan_visualea_tuto>`


Package content:
----------------
This package contains lots of in-development and development-stopped code which 
is probably of no interest.

In general, the analysis of roots from image is supposed to be done using 
ref:`image pipelines <rhizoscan_pipeline>` which calls iterativelly as set of 
functions (or *modules*) to:

 1) process the images
     * segmentation
     * detection of extra-contents (frame, seed and leaves)
     * extract the root pixel mask
     * convert the root mask to a graph
     
 2) from the root graph
     * extract an *axial tree graph*
     * export it to the `MTG`_ format
    

The :mod:`rhizoscan.root.image` package contains the image processing functionalities
 
The :mod:`rhizoscan.root.graph` module contains the graph functionalities 

The :mod:`rhizoscan.root.pipeline` package contains the pipelines



.. _MTG: http://openalea.gforge.inria.fr/doc/vplants/newmtg/doc/_build/html/contents.html
"""

_aleanodes_ = []           # openalea wrapped packages
__icon__    = 'root.png'   # icon of openalea package


