from __future__ import absolute_import
from matplotlib import pyplot as plt
from IPython.display import Image
#RhizoScan Import
from rhizoscan import get_data_path
from rhizoscan.root.pipeline import load_image, detect_petri_plate, compute_graph, compute_tree
from rhizoscan.root.pipeline.arabidopsis import segment_image, detect_leaves, _detect_leaves
from rhizoscan.root.graph.mtg import tree_to_mtg
from rhizoscan.root.image.seed import _cluster_seed

from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_opening, binary_closing
import numpy as np
from skimage.io import imsave
from path import Path

def rhizo_image(image_filename, dx_seed = 15, nb_dilatation=3, nb_closing=2, min_tip_length=15, root_max_radius=3):
    
    fn, ext = Path(image_filename).splitext()
    
    #get_ipython().magic(u'pylab notebook')

    #Image(image_filename) #affichage de l image telle qu elle est 
    
    #Ici on va charger et sauvegarder l image ouverte sur python
    Image = load_image(image_filename)
    #imshow(Image)


    #Toute cette partie du code sert a segmenter l image
    rmask, bbox = segment_image(Image, root_max_radius=root_max_radius, min_dimension=300)

    #On creer les labels et on sauvegarde cette seconde image
    label_image, nb_label= label(rmask, return_num=True)
    regions = regionprops(label_image)
    #imshow(label_image)
    #mettre un save
    fn2 = fn+'_stage1'
    #imsave(fn2+ext, label_image)

    reg = regions[0]
    regions = [reg for reg in regions if reg.area > 50]

    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(label_image) #pas utile pour la suite 

    #creation de la seed map
    seed_map = np.zeros(rmask.shape, dtype=np.uint8)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        seed_map[minr:minr+dx_seed, minc:maxc+1] = label_image[minr:minr+dx_seed, minc:maxc+1]
            
    #dilatation et closing de rmask
    for i in range(nb_dilatation):
        rmask = binary_dilation(rmask)
    for i in range(nb_closing):
        rmask = binary_closing(rmask)

    fn3 = fn+'_stage2'
    imsave(fn3+ext, rmask)

    ##Creation du graphe
    graph = compute_graph(rmask, seed_map=seed_map, bbox=bbox, verbose=False)
    graph.plot(linewidth=4)
    #ajouter un save pour sauvegarder l'image du graphe
    fn4 = fn+'_stage3'
    plt.savefig(fn4+ext, dpi=300)

    ##Creation de l'arbre
    tree = compute_tree(graph=graph, px_scale=5, min_length=min_tip_length)
    tree.plot(linewidth=4)
    #ajouter un save pour sauvegarder l'image du graphe

    fn5 = fn+'_stage4'
    plt.savefig(fn5+ext, dpi=300)








