#!/usr/bin/env python
# coding: utf-8

# # RhizoScan Pipeline

# This pipeline is a step by step pipeline

# ## Common import

# In[ ]:


from __future__ import absolute_import


# In[ ]:


get_ipython().magic(u'pylab notebook')
from matplotlib import pyplot as plt
from IPython.display import Image
 


# ## RhizoScan Import

# In[ ]:


from rhizoscan import get_data_path
from rhizoscan.root.pipeline import load_image, detect_petri_plate, compute_graph, compute_tree
from rhizoscan.root.pipeline.arabidopsis import segment_image, detect_leaves, _detect_leaves
from rhizoscan.root.graph.mtg import tree_to_mtg
from rhizoscan.root.image.seed import _cluster_seed


# ## RSA Image

# In[ ]:


image_filename ='https://github.com/fortfe/rhizoscan/blob/fort_scripts/example/Seq%208_Boite%2000002_graphe.JPG'
#image_filename = ''
Image(image_filename)


# ### Load Image

# In[ ]:


Image = load_image(image_filename)
imshow(Image)


# ### Detect features (Petri plate)

# ### Image Segmentation

# In[ ]:


rmask, bbox = segment_image(Image, root_max_radius=10, min_dimension=500)


# ### Detect leaves and seed

# In[ ]:


from skimage.measure import label, regionprops
#from skimage.color import label2rgb

label_image, nb_label= label(rmask, return_num=True)
regions = regionprops(label_image)
imshow(label_image)


# In[ ]:


reg = regions[0]
regions = [reg for reg in regions if reg.area > 50]
print(len(regions))


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(label_image)


# In[ ]:


import matplotlib.patches as mpatches


for region in regions:
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
fig


# In[ ]:


import numpy as np
seed_map = np.zeros(rmask.shape, dtype=np.uint8)

for region in regions:
    minr, minc, maxr, maxc = region.bbox
    seed_map[minr, minc:maxc+1] = label_image[minr, minc:maxc+1]
            
    
    


# In[ ]:


imshow(seed_map)


# ### Compute the graph corresponding to the RSA

# In[ ]:


graph = compute_graph(rmask, seed_map=seed_map, bbox=bbox, verbose=True)
graph.plot(linewidth=4)


# In[ ]:


get_ipython().set_next_input(u'graph = compute_graph');get_ipython().magic(u'pinfo2 compute_graph')


# In[ ]:


graph = compute_graph


# ### Extract a tree from the graph

# In[ ]:


get_ipython().magic(u'pinfo2 compute_tree')


# ### Save the RSA into an MTG

# In[ ]:


tree = compute_tree(graph=graph, px_scale=10, min_length=150)
tree.plot(linewidth=4)


# In[ ]:


g = rsa = tree_to_mtg(tree)
g.display()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




