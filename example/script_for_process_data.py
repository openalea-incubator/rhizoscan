# ==============================================================================
import cv2
import matplotlib.pyplot

from rhizoscan.root.pipeline import (
    detect_petri_plate,
    compute_graph,
    compute_tree)

from rhizoscan.root.pipeline.arabidopsis import (
    segment_image,
    detect_leaves_with_kmeans)

from rhizoscan.root.graph.mtg import (
    tree_to_mtg,
    RSMLSerializer)

# ==============================================================================


def show_image(img, description=""):
    print description
    matplotlib.pyplot.imshow(img)
    matplotlib.pyplot.show()


def run(image_filename, have_petri_plate=False, verbose=False):

    # ==========================================================================
    # Load image

    image = cv2.imread(image_filename, flags=cv2.IMREAD_GRAYSCALE)

    if verbose:
        show_image(image, "original image")

    # ==========================================================================
    # Detect features (Petri plate)

    border_width = 250  # Measure in pixel
    plate_size = 120    # Useful just for pixel scale
    fg_smooth = 1       # Dono

    if have_petri_plate:
        pmask, px_scale, hull = detect_petri_plate(image,
                                                   border_width=border_width,
                                                   plate_size=plate_size,
                                                   fg_smooth=fg_smooth)

        if verbose:
            show_image(pmask, "mask petri plate")
            print "Pixel scale :", px_scale
    else:
        pmask = None
        px_scale = None

    # ==========================================================================
    # Image Segmentation

    root_max_radius = 13

    rmask, bbox = segment_image(image,
                                pmask=pmask,
                                root_max_radius=root_max_radius,
                                min_dimension=50,
                                smooth=2,
                                verbose=True)

    if verbose:
        show_image(rmask, "segment mask")

    # ==========================================================================
    # Detect leaves and seed

    root_min_radius = 3
    plant_number = 5

    seed_map = detect_leaves_with_kmeans(rmask,
                                         erode_iteration=0,
                                         bounding_box=[0.05, 0.06, 0.90, 0.10],
                                         plant_number=plant_number)

    if verbose:
        show_image(seed_map + rmask, "seed_map+rmask")

    # ==========================================================================
    # Compute the graph corresponding to the RSA

    graph = compute_graph(rmask, seed_map, bbox=bbox, verbose=False)

    if verbose:
        graph.plot()
        matplotlib.pyplot.show()

    # ==========================================================================
    # Extract a tree from the graph

    tree = compute_tree(graph, px_scale=px_scale, min_length=15)

    if verbose:
        tree.plot()
        matplotlib.pyplot.show()

    # ==========================================================================
    # Save the RSA into an MTG

    mtg = tree_to_mtg(tree)

    rsml_serializer = RSMLSerializer()
    rsml_serializer.dump(mtg, image_filename + '.rsml')


def main():

    images_filename = ['img/T1.jpg',
                       'img/T2.jpg',
                       'img/T3.jpg',
                       'img/T4.jpg',
                       'img/T5.jpg',
                       'img/T6.jpg',
                       'img/T7.jpg',
                       'img/T8.jpg',
                       'img/T9.jpg']

    for filename in images_filename:
        run(filename, verbose=True)

if __name__ == "__main__":
    main()


