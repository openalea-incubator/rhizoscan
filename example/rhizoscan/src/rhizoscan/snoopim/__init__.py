"""
access function to snoopim input & output
"""

def get_dataset_path(name, *args):
    """ return the path of dataset `name`
    
    If *args, return the path to the sub-folder listed in *args
    
    Using environment variable "DATASET_PATH", this function basically returns:
      $DATASET/`name`/args[0]/args[1]...
    """
    import os
    
    dataset_path = os.environ['SNOOPIM_DB_PATH']
    
    return os.path.join(dataset_path, name, *args)
    
