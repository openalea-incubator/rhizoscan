.. _image-database:

Image Database
##############


Database file system
====================

The :ref:`rhizoscan <rhizoscan>` package provide a simple method to manage and process sets of images through a *database* mechanism. Here, a database is basically an unordered list of items with the following attributs: 
  :filename: the image file name
  :metadata: a (hierarchical) structure of descriptive parameters
  :output:   the file name base for all data computed from the orignal image

.. note:: this is not a real database, but it provides similar behaviors

An image database is made of:
  - a set of image files stored such that they can all be listed using a `shell globbing`_ pattern process by the `python glob`_ tool. For example the pattern ``*/*.jpg`` lists all ``.jpg`` images found in any subfolder for the current directory.
  - a ``.ini`` file that describes the database. In particular, it contains the ``globbing pattern`` mentioned above, and the metadata related to the images.

.. _shell globbing: http://en.wikipedia.org/wiki/Glob_(programming)
.. _python glob: http://docs.python.org/2/library/glob.html

To load such as databse with python, do:
  >>> from rhizoscan.root.pipeline import database
  >>> db, invalid, output = database.parse_image_db('openalea/rhizoscan/test/data/pipeline/arabidopsis/database.ini')

This function returns:
  :db: The database
  :invalid: the list of files that correspond to the globbing pattern but for which filename included metadata were not recognized
  :output: the relative path for storing computed data 

Database descriptor:
++++++++++++++++++++

To understand how to make a database `ini` file, let's look at the exemple in ``[rhisoscan-dir]/test/data/pipeline/arabidopsis/database.ini``:

.. include:: ../../../test/data/pipeline/arabidopsis/database.ini
    :literal:

With folder content::
    
    J10/
      + Photo_001.jpg
      + Photo_011.jpg
      + Photo_invalid.jpg
    J11/
      + Photo_001.jpg
      + Photo_011.jpg
    database.ini
    
The ``ini`` file contains four parts:

  **PARSING** 
    Describe which files to process and which metadata to attach to them. 
    
    The ``pattern`` field indicates what files should be processed, by replacing brackets by ``*``, it gives the file globbing pattern ``*/Photo_*.jpg``. Then the brackets indicates how to use what is replaced by the ``*``:
        - the 1st is a string (str) and should be stored as the ``age`` metadata
        - the 2nd is an integer (int) and is stored as the ``id`` metadata
    
    The brackets is always a pair ``metadata_name:parameter_type`` where ``parameter_type`` should be either:
        - a python type (``int``, ``float``, ``str``, ...)
        - ``$``: which means to use the content the fields in the ini file that has the respective name (see :ref:`database-label`)
        - ``date``: identified as a date, it requires ``PARSING`` to have a ``date`` field (see :ref:`database-datetype` for details)
        
    If a detected file has a parameter in its file name that does not respect the given data type, it is not added to the database but returned in the ``invalid`` output.
        
    The ``ini`` file of this exemple also provide a ``group`` field to attach metadata to group of images with respect to there position in their respective folder. See :ref:`database-group` section for details
    
    
  **metadata**
    default parameters-values to attach to **all detected files**.
    
  **A & B**
    metadata set that can be attach to the group of files with respective label. Here these are metadata to attach to the images of group ``A`` and ``B`` respectively.
  

.. _database-group:
    
grouping database files
-----------------------
A simple way to attach specific metadata to a group of files is to group them by their position (sorted by file name) in the folder they are stored in. This is usefull if images are given in folder such that images sharing metadata appears in a specific order. 

The grouping mechanism is obtained using the ``PARSING`` keyword ``group``. In the example above, it indicates that:
  - from the image ``0`` (i.e. the 1st), files are of the group ``A``
  - from the image ``1`` (i.e. the 2nd), files are of the group ``B``
  
In this exemple there are four images in the database (and one invalid files), which stored by pairs in 2 folders. The first group ``A`` contains the first image of each folder, and the group ``B`` the second.


.. _database-label:

adding metadata to labeled file name 
------------------------------------

If image file or folder name contains specific labels (i.e. word), it an be used to attach related metadata to the respective database elements using the ``$`` parameter type.

.. _second-example:

With the following file structure::

    2012_03_21/
      + GEN01/
          + nitrate_001.jpg
          + nitrate_010.jpg
      + GEN02/
          + nitrate_001.jpg
          + nitrate_010.jpg
          
    2012_03_22/
      + GEN01/
          + nitrate_001.jpg
          + nitrate_010.jpg
      + GEN02/
          + nitrate_001.jpg
          + nitrate_010.jpg
    database.ini

Using the ``ini`` file::
    
    [PARSING]
    pattern=[date:date]/[genotype:$]/nitrate_[nitrate:int].jpg
    date=%Y_%m_%d
    
    [metadata]
    xp=example
    
    [GEN01]
    name=genotype number 1
    gene=GEN01
    
    [GEN02]
    name=genotype number 2
    gene=GEN02

Then, each database element will have a ``genotype`` metadata with the content of field ``GEN01`` or ``GEN02`` respectively.

.. note:: If ``[$:$]`` is used then all the fields of the relative metadata group (``GEN01`` or ``GEN02``) are appended directly at the metadata base of the db element. Here, each element will have the respective ``name`` and ``gene`` metadata.


.. _database-datetype:

storing date metadata in file name
----------------------------------

As in :ref:`the example above <second-example>`, a date type of metadata can be given in the file or folder name. In this case, a ``date`` field should be present in ``PARSING`` that describe how the date is written, with one of the `time format <http://docs.python.org/2/library/time.html>`_ (see the listing of the function ``strftime`` for details)


.. _database-operation:

Database operations
===================
.. todo:: database operation

>>> from rhizoscan.root.pipeline import database
>>> database.filter(db, key=None, value=None, metadata=True)
>>> database.cluster(db, key, metadata=True)
>>> database.get_metadata(db)

