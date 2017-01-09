==============================
Installation on Conda (Ubuntu)
==============================

.. contents::

1. Download miniconda
---------------------

See : http://conda.pydata.org/miniconda.html


2. Create your own virtual environment
--------------------------------------

.. code:: shell

    conda create --name rhizoscan python

    source activate rhizoscan


3. Install conda dependencies
-----------------------------

.. code:: shell

    conda install sphinx jupyter nose coverage anaconda-client
    conda install numpy scipy matplotlib scikit-image opencv pil pillow
    conda install -c openalea openalea.mtg openalea.vpltk openalea.visualea openalea.core

3.1 Download & install TreeEditor
.................................

.. code:: shell

    git clone https://github.com/VirtualPlants/treeeditor
    cd treeeditor
    python setup.py develop

3.2 Download & install RSML-conversion-tools
............................................

.. code:: shell

    git clone https://github.com/RootSystemML/RSML-conversion-tools
    cd RSML-conversion-tools/python/rsml
    python setup.py develop

4. Install & test Rhizoscan
---------------------------

.. code:: shell

    git clone https://github.com/VirtualPlants/rhizoscan
    cd rhizoscan
    python setup.py develop
    cd test 
    nosetests

