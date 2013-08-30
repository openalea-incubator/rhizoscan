"""
Implement serialization functionalities for Data class and subclass
"""

import cPickle as cPickle
import os as _os


class File(object):
    """
    One-file storage
    
    
    """
    def __init__(self, filename, serializer='cPickle'):
        self.filename   = _os.abspath(filename)
        if serializer=='cPickle':
            serializer = _cPickle
        self._serializer = serializer

    
class FileMap(object):
    """
    A FileMap is a simple map-storage with the hard-disc
    
    A map-storage is a storage that save/load data w.r.t an identifier.
    The iterface is done using [], i.e. through `__getitem__` and `__setitem__`
    
    The FileMap use a string.format pattern to relate identifiers to filenames 
    """
    def __init__(self, file_format, serializer='cPickle'):
        """
        Create a map-storage on hard-disc filesystem following `file_format`
        
        `file_format` should be a string that can be used with the python 
        format functionality (eg. "directory/data_{}.data") of simply a string
        to which is append the name of data to be saved: file_format+"{}"
        
        Object "name" is stored/retrieved from `file_format.format(name)'  
        """
        if not '{' in file_format or not '}' in file_format:
            file_format += "{}"
        self.file_format = file_format
        
        if serializer=='cPickle':
            serializer = _cPickle
        self._serializer = serializer 
        
    def __setattr__(self, name, data):
        """ stores `data` to file with key value `name` """
        e = self.get_entry(name,mode='w')
        ## serialize and save
        e.close()
        ## make data_loader and stores is in self.__dict__
        
    def __getitem__(self,name):
        """ retrieve data from FileStorage (i.e. load the file) """
        ## todo
        pass

    def get_entry(self, name, mode='r'):
        return open(file_format.format(name), mode=mode)

