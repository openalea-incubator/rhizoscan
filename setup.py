# -*- coding: utf-8 -*-
__revision__ = "$Id: $"

import sys
import os

from setuptools import setup, find_packages



#The metainfo files must contains 'metainfo.ini'
# version, release, project, name, namespace, pkg_name,
# description, long_description,
# authors, authors_email, url and license
# * version is 0.8.0 and release 0.8
# * project must be in [openalea, vplants, alinea]
# * name is the full name (e.g., VPlants.RhizoScan) whereas pkg_name is only 'rhizoscan'

# name will determine the name of the egg, as well as the name of 
# the pakage directory under Python/lib/site-packages). It is also 
# the one to use in setup script of other packages to declare a dependency to this package)
# (The version number is used by deploy to detect UPDATES)
import ConfigParser
conf = ConfigParser.RawConfigParser()
conf.read('metainfo.ini')
metadata = dict([(key,conf.get('metainfo',key)) for key in conf.options('metainfo')])


print '*** Installing the following package: ***'
for key,value in metadata.items():
    key = str(key)
    print '\t', key+':\t', value

# Packages list, namespace and root directory of packages
namespace = metadata['namespace']

pkg_root_dir = 'src'
packages     = [ pkg for pkg in find_packages(pkg_root_dir)]
top_pkgs     = [ pkg for pkg in packages if  len(pkg.split('.')) < 2]
package_dir  = dict( [('',pkg_root_dir)] + [(namespace + "." + pkg, pkg_root_dir + "/" + pkg) for pkg in top_pkgs] )


# setup dependencies stuff
setup_requires   = metadata.get('setup_requires', '')
install_requires = metadata.get('install_requires', '')
dependency_links = ['http://openalea.gforge.inria.fr/pi']

## generate openalea wrapper and entry points
#  ------------------------------------------
#X try:
sys.path.insert(0,os.path.abspath('src'))
import rhizoscan
from rhizoscan.workflow.openalea import wrap_package
entry =  wrap_package(rhizoscan,entry_name='rhizoscan',verbose=0)
print '\n wralea entry found:\n' + '\n'.join(entry) + '\n'
#X except ImportError, e:
#X     print 'Failed to build wralea nodes. Please rerun python setup.py'
#X     entry = None
#X     raise e
#X 
entry_points = {}
if entry:
    entry_points['wralea'] = entry

# List of top level wralea packages (directories with __wralea__.py) 
#wralea_entry_points = ['%s = %s'%(pkg,namespace + '.' + pkg) for pkg in top_pkgs]

# dependencies to other eggs
install_requires = []



setup(
    name            = metadata['name'],
    version         = metadata['version'],
    description     = metadata['description'],
    long_description= metadata['long_description'],
    author          = metadata['authors'],
    author_email    = metadata['authors_email'],
    url             = metadata['url'],
    license         = metadata['license'],
    keywords        = metadata.get('url',''),	

    # package installation
    packages=    packages,	
    package_dir= package_dir,
    
    ## ---- include openaleaPlus module in egg ---- 
    ##py_modules= ['openaleaPlus'],
    #  --------------------------------------------
    
    # Namespace packages creation by deploy
    #namespace_packages = [namespace],
    create_namespaces = False,
    zip_safe= False,

    # Dependencies
    setup_requires = setup_requires,
    install_requires = install_requires,
    dependency_links = dependency_links,


    # Eventually include data in your package
    # (flowing is to include all versioned files other than .py)
    # include_package_data = True,
    # (you can provide an exclusion dictionary named exclude_package_data to remove parasites).
    # alternatively to global inclusion, list the file to include   
    package_data = {'' : ['*.png'],},

    # postinstall_scripts = ['',],

    # Declare scripts and wralea as entry_points (extensions) of your package 
    entry_points = entry_points,
    )


