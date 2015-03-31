from rhizoscan.root.pipeline import load_image, detect_petri_plate, compute_graph, compute_tree
from rhizoscan.root.pipeline.arabidopsis import segment_image, detect_leaves
from rhizoscan.root.graph.mtg import tree_to_mtg

from matplotlib import pyplot as plt

image = load_image(image_filename)
#plt.imshow(image);

pmask, px_scale, hull = detect_petri_plate(image,border_width=25, plate_size=120, fg_smooth=1)
plt.imshow(pmask);

rmask, bbox = segment_image(image,pmask,root_max_radius=5)
#plt.imshow(rmask);


seed_map = detect_leaves(rmask, image, bbox, plant_number=2, leaf_bbox=[0,0,1,.4])
#plt.imshow(seed_map);
#plt.imshow(seed_map+rmask);

graph = compute_graph(rmask,seed_map,bbox)
#graph.plot()

tree = compute_tree(graph, px_scale=px_scale)
#tree.plot()

#rsa = tree_to_mtg(tree)
