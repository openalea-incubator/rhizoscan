.. _rhizoscan_pipeline:

The image pipeline
==================

The analysis of root system from images is done using on of a set of *image pipelines*. Each pipeline comes as a all-in-one function that iteratively run the *processing modules* of  the pipeline.

These modules typically do each of the following tasks:
  1. frame detection
  2. image segmentation
  3. seeds detection
  4. convertion to graph
  5. extraction of the Root System Architecture
  

.. _arabidopsis-pipeline:

The arabidopsis pipeline
------------------------

This pipeline has been developed to analysis image of arabidopsis root system grown and imaged using a specific *experimental protocol*. It contains the following steps:

 1. Petri dish detection
      The root system have been growned and imaged in a squared Petri dish which is marked by four hand drawn curves, one on each corner, using a black pen.
 2. Image segmentation
      It follows the standard algorithm which first estimate and remove the lighting background using an overestimate of the maximum root radius, in pixels. It then separate root area from background using a simple expectation maximization (EM) algorithm.
 3. Leaves detections
     The analysis pipeline uses the detected leaves to determine automatically the starts of the root systems. This is done by image segmentation based on the leaves opacity being higher than the root axes.
 4. Convertion to graph
     This is the standard algorithm and doesn't need any parametrization
 5. Extraction of the Root System Architecture
     The Extraction of the RSA is done using a apriori model suitable for arabidopsis: it detect only one main root axes (the longest), all other being at least secondary.
     
.. note:: This pipeline can thus analyse root images if the following apply:

 - the Petri dish respect the frame detection protocol
 - leaves are more opaque than roots
 - there is one main axes (order 1), and it is the longuest

.. _arabidopsis-pipeline-API:

Arabidopsis pipeline API
++++++++++++++++++++++++

To use the arabidopsis pipeline from python, do::
    
    from rhizoscan.root.pipeline.arabidopsis import pipeline
    data = pipeline.run(**inputs_arguments)
    
With the following ``inputs_arguments``:
    :image:            ``(R)`` The image filename or a numpy array-like to  
                       analyse
    :output:           ``(R)`` The commun base of output file. Each module of 
                       the pipeline will save a file with path like *output_suffix*
    :update:           An optional list of the module name to recompute even if
                       previously computed data is accessible
    :metadata:         Optional dictionary-like structure with arbitrary field 
                       names (keys). The metadata is appended to the *'tree'* output data. Moreover all fields are added to the pipeline *namespace* and the metadata can be provide values for the pipeline inputs arguments. 
    :plant_number:     ``(D)`` Number of roots systems. default is ``1``
    :plate_width:      ``(D)`` The *real* size of the Petri plate side in the   
                       desired unit for output measurements (default ``120``).
    :leaf_height:      ``(D)`` A list of 2 numbers between 0 and 1 that reduces     
                       the search area for leaf with respect to the detected Petri plate. The default is ``[0.,0.2]``, meaning that the leaves appear in the 20% superior part of the plate.
    :root_max_radius:  ``(D)`` Overestimate of the maximum root radius 
                       **in pixels**. Anything between 1 and 3 times the real value is suitable.  
    :root_min_radius:  Estimate lower bound of root radius used by the leaf
                       detection algorithm. It is not a sensitive parameter.
                       Increasing it tend to increase the leaf area.
    :min_dimension:    Minimum size of root system in pixels (default ``50``).   
                       Anything less than this size is not analysed. 
    :smooth:           Initial smoothing of the input image before processing.
                       This value is the sigma parameter (in pixels) of a gaussian kernel (default ``1``). 
    :verbose:          If positive, print some intermediate computing state.  

``(R)`` Are required arguments, and ``(D)`` are arguments that depend on the image data and should be asserted and probably changed if default values are not suitable. For the others, the default values are generic, and should not need to be changed.
.. ``image`` and ``output`` ar required parameters. Moreover, ``plant_number``, ``plate width``, ``leaf_height`` and ``root_max_radius`` should be provided if the respective default values are not suitable.

The pipeline.run returns a dictionary of the pipeline *namespace* (``data`` in the above example). It is the set of variables used througout the pipeline which contains the given parameters and the computed data. Those are:
    :pmask:    (numpy arrays) The mask of the detected Petri plate
    :rmask:    (numpy arrays) The binary mask of the root axes
    :seed_map: (numpy arrays) The detected leaf area map
    :graph:    (RootGraph) The graph representing the root axes in ``rmask``
    :tree:     (RootAxialTree) The extracted axial tree representing the root 
               systems
    :bbox:     (tupel of slices) The bounding box of the detected Petri plate in 
               the original image. ``rmask`` and ``seed_map`` are for to this region.
    :px_ratio: (float) The size of 1 pixel in the designed measurement unit. It     
               is computed based on the ``plate_width`` and the detected plate area


See also
++++++++
  - tutorials:
      - :ref:`rhizoscan_script_tuto` 
      - :ref:`rhizoscan_visualea_tuto`
  - back to the :ref:`rhizoscan_manual` 

