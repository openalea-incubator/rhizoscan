"""
manage access to the images of snoopim dataset
"""
import scipy as sp
from . import get_dataset_path

def get_thumbnails(dataset_name, image_name):
    filename = get_dataset_path(dataset_name,'thumbnails',image_name)
    ratio    = get_dataset_path(dataset_name,'thumbnails','ratio.txt')
    
    with open(ratio) as f:
        for line in f:
            if image_name in line:
                ratio = float(line.split(':')[1])
                break
    return sp.misc.imread(filename), ratio

