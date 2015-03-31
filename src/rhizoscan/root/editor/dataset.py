"""
manage dataset in RootEditor
"""
from rhizoscan.root.pipeline import dataset
from rhizoscan.datastructure import Data as _Data


class Dataset(_Data,list):
    pass
