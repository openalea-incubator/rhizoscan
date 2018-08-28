# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       author : Lafoi Lessy
# ==============================================================================
import skimage.measure
import numpy as np
import cv2
from sklearn.cluster import KMeans
import numpy

from rhizoscan import get_data_path
# ==============================================================================

def _load_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = image.astype(numpy.uint8)
    return image

def load_my_image(root_type="small"):
    """
    root_type = "small", "medium" or "big"
    Parameters
    ----------
    root_type

    Returns
    -------
    GRAYSCALE image on uint8/ format
    """
    filename = get_data_path('pipeline/{}_root.jpg'.format(root_type))
    return _load_image(filename)


# ==============================================================================

def remove_petri_plate(image, bbox=(5000, 660, 750, 5500)):

    image[bbox[0]:, :] = 0
    image[:bbox[1], :] = 0
    image[:, :bbox[2]] = 0
    image[:, bbox[3]:] = 0

    return image

# ==============================================================================


def segment_root(image, bbox=(5000, 660, 750, 5500), plant_number=5):

    rmask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 101, 25)

    label_image = skimage.measure.label(rmask)
    regions = skimage.measure.regionprops(label_image)

    # kept the biggest component
    region = max(regions, key=lambda x: x.area)

    rmask = label_image.copy()
    rmask[rmask != region.label] = 0
    rmask[rmask > 0] = 255

    rmask = remove_petri_plate(rmask, bbox=bbox)

    return rmask.astype(numpy.bool)


def segment_leaf(root_mask, plant_number=5, open_iteration=10):
    # Kept biggest connected component
    kernel = np.ones((4, 4))
    open_root_mask = cv2.morphologyEx(root_mask.astype(np.uint8),
                                      cv2.MORPH_OPEN,
                                      kernel,
                                      iterations=open_iteration)

    x, y = numpy.where(open_root_mask > 0)
    pts = list()
    for xx, yy in zip(x, y):
        pts.append((xx, yy))
    pts = numpy.array(pts, dtype=float)

    kmeans = KMeans(n_clusters=plant_number).fit(
        pts[:, 1].reshape(pts.shape[0], 1))
    
    label = kmeans.labels_

    pts = pts.astype(int)
    for (x, y), label in zip(list(pts), label):
        open_root_mask[x, y] = int(label) + 1

    return open_root_mask.astype(numpy.uint8)


def segment_root_and_leaf(image, bbox=(5000, 660, 750, 5500), plant_number=5,
                          open_iteration=10):
    """

    Parameters
    ----------
    image
    bbox
    plant_number

    Returns
    -------

    """
    root_mask = segment_root(image,
                             bbox=bbox,
                             plant_number=plant_number)

    leaf_mask = segment_leaf(root_mask,
                             plant_number=plant_number,
                             open_iteration=open_iteration)

    return root_mask, leaf_mask

# ==============================================================================
