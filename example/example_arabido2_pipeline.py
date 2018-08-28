# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
import matplotlib.pyplot as plt

import rhizoscan.root.pipeline.arabido2 as rzs_arabido2
import rhizoscan.root.pipeline as rzs_pipeline
import rhizoscan.root.graph.mtg as rzs_mtg
# ==============================================================================


def show_image(image, color=None):
    plt.imshow(image, cmap=color)
    plt.show()


def main():

    image = rzs_arabido2.load_my_image("big")
    show_image(image, color="gray")

    # ==========================================================================
    # Segmentation root and leaf
    plant_number = 5
    bbox = (5000, 660, 750, 5500)
    open_iteration = 10 # 7 for small and medium root

    root_mask, leaf_mask = rzs_arabido2.segment_root_and_leaf(
        image, bbox=bbox, plant_number=plant_number, open_iteration=open_iteration)
    show_image(root_mask + leaf_mask)

    # ==========================================================================
    # Compute the graph corresponding to the RSA
    graph = rzs_pipeline.compute_graph(root_mask, leaf_mask)
    graph.plot(linewidth=1)
    plt.show()

    # ==========================================================================
    # Extract a tree from the graph
    px_scale = 0.0307937766585
    tree = rzs_pipeline.compute_tree(graph, px_scale=px_scale)
    tree.plot(linewidth=1)
    plt.show()
    # ==========================================================================
    # Save the RSA into an MTG
    g = rsa = rzs_mtg.tree_to_mtg(tree)
    g.display()

if __name__ == "__main__":
    main()
