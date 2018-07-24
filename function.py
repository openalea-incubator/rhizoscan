from rhizoscan import get_data_path
from rhizoscan.root.pipeline import load_image, detect_petri_plate, compute_graph, compute_tree
from rhizoscan.root.pipeline.arabidopsis import segment_image, detect_leaves
from rhizoscan.root.graph.mtg import tree_to_mtg


from skimage import  io
from skimage import filters
import skimage.measure
import numpy as np
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from sklearn.cluster import KMeans


from skimage.io import imread, call_plugin, plugin_info
from skimage.util.dtype import convert

from cv2 import erode as _erode
from sklearn.cluster import KMeans as _Kmeans

import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow import node as _node         # declare workflow nodes
from rhizoscan.workflow import pipeline as _pipeline # declare workflow pipeline

from rhizoscan.ndarray.measurements import clean_label as _clean_label
from rhizoscan.image                import Image       as _Image

#################### load image ##########################################

def load_image(filename, normalize=True):
    img = _Image(filename, dtype='f', color='gray')
    if normalize:
        img,op = normalize_image(img)
        img.__serializer__.post_op = op
    return img
	 
######################Detect Features ###################################
"""

pmask, px_scale, hull = detect_petri_plate(image,border_width=35, plate_size=140, fg_smooth=1)

Instead  of border_width = 25 (Julien), I change to 35 and  plate_size = 120 (Julien), i change to 140.

NB : we keep the same code

"""
################### Image segmentation ###################################

def segment(image, pmask=None):
	
	image = image.astype(_np.uint8)
	
	rmask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,101,25)
		
	label_image = skimage.measure.label(rmask)
	regions     = skimage.measure.regionprops(label_image)

	

	region = max(regions,key = lambda x: x.area )
	rmask = label_image.copy()

	rmask[rmask != region.label] = 0

	rmask[rmask > 0 ] = 255 

	rmask[5000: ,:] = 0 
	rmask[:660 , :] = 0
	rmask[: , :750] = 0
	rmask[: ,5500:] = 0 
	
	return rmask
	
######################## detect_leave #####################################


def detect_leav(rmask):
	"""
	input rmask is of the segmentation
	
	output : the big component connexe
	
	"""
	kernel = np.ones((4,4))

	image = cv2.morphologyEx(rmask.astype(np.uint8),cv2.MORPH_OPEN, kernel, iterations = 10)
    
	image[5000: ,:] = 0 
	image[:660 , :] = 0
	image[: , :750] = 0
	image[: ,5500:] = 0 

	for i in range(5):
		label_image1 = skimage.measure.label(np.array(image))
		regions1     = skimage.measure.regionprops(label_image1)

	region1 = max(regions1,key = lambda x: x.area)
	image = label_image1.copy()
	
	plant_number = 5
	x, y = _np.where(image > 0)

	pts = list()
	for xx, yy in zip(x, y):
		pts.append((xx, yy))

	pts = _np.array(pts, dtype=float)

	kmeans = _Kmeans(n_clusters=plant_number).fit(pts)
	label = kmeans.labels_

	pts = pts.astype(int)

	for (x, y), label in zip(list(pts), label):
		image[x, y] = int(label) + 1
		
	return image
	
	
	
