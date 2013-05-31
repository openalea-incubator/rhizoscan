.. _rhizoscan_script_tuto:

Root image analysis with python
========================================

This tutorial provides a step by step description of how to do root images analysis with :ref:`the arabidopsis pipeline <arabidopsis-pipeline>` using the python programming language. A `minimal knowledge`_ of python is recommanded.

.. _minimal knowledge: http://docs.python.org/2/tutorial/introduction.html

.. contents:: Content of this tutorial
   :local:
   
.. section-numbering
    
Single image analysis
---------------------
To process one root image, you will needs:
  - to select an image (or image file name) to be analysed
  - to select an output path (the *file name base* of stored data)
  - to choose the suitable pipeline parameters

For this tutorial, we use an image provided with the package, and store the computed data in a relative path.

>>> # input image filename
>>> from rhizoscan import get_data_path
>>> image = get_data_path('pipeline/arabidopsis/J10/Photo_001.jpg')

>>> # file name base for stored data
>>> import tempfile, os
>>> outdir = tempfile.mkdtemp()                 # create a temporary directory
>>> output = os.path.join(outdir, 'Photo_001')

For this example, the default pipeline parameters are used, but for ``plant_number``: the input image contains 5 root systems. For details on the possible parameters, see :ref:`the API of the arabidopsis pipeline <arabidopsis-pipeline-API>`.

>>> from rhizoscan.root.pipeline.arabidopsis import pipeline
>>> data = pipeline.run(image=image,output=output, plant_number=5, verbose=1)
>>>    # this takes a couple of minutes to compute


The returned ``data`` variable is a dictionary of the pipeline namespace: it contains the modules computed data and the given parameters. See :ref:`the API of the arabidopsis pipeline <arabidopsis-pipeline-API>` for the list of its content.

.. note::
    The file name of the storage files will all start by the value of ``output`` and a suffix related to the stored data. E.g. the seed map image use the suffix "_seed.png", so ins our example a file ``[outdir]/Photo_001_seed.png`` will be created. If you want you can give another value to ``outdir`` but don't forget to create the directory if it doesn't exist!

Once you have finished with the computed data, don't forget to delete it: either manually using your OS file manager, or with python::

    import shutil
    shutil.rmtree(outdir)


Database analysis
-----------------

An :ref:`image database <image-database>` can be process easily. For example, using the testing databse of rhizoscan, this is done using the following::
    
    from rhizoscan import get_data_path
    from rhizoscan.root.pipeline import database
    from rhizoscan.root.pipeline.arabidopsis import pipeline
    
    db = get_data_path('pipeline/arabidopsis/database.ini')
    db, invalid, outdir = database.parse_image_db(db) 

    for elt in db:
        pipeline.run(elt)


.. todo:: To finish

      - what are hidden the paremeter => cf :ref:`pipeline api<arabidopsis-pipeline-API>`
      - how to get output data (ex 'tree')

Finally, if your don't need it anymore, remove the output directory used by the pipeline::
    
    import shutil
    shutil.rmtree(outdir)

Visualisation and measurements
------------------------------

.. Note:: Most of the following requires a matplotlib

.. todo:: split in the 2 previous parts? 

    plotting graph & tree
    exemple of getting some measurement from a tree: root.measurement


.. visualea: http://openalea.gforge.inria.fr/dokuwiki/doku.php?id=documentation:user:visual_programming

