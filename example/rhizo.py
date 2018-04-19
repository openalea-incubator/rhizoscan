from rhizoscan.root.pipeline import *
from openalea.core.path import path
from time import strptime

root_dir = path('../share/test_root_DB').abspath()

#X def date(dir_name):
#X     s = dir_name.basename()
#X     return strptime(s, '%Y_%m_%d_%Hh%M')
#X 
#X dirs = list(root_dir.walkdirs())
#X d1, d2 = dirs[:2]
#X day1 = date(d1)
#X day2 = date(d2)
#X days = sorted(map(date, dirs))
#X 
# Simple silly step
images = list(root_dir.walkfiles('*.jpg'))

# Version
trees = []
failure = []
for img in images:
    try:
        tree = image_pipeline_no_frame(img)
        trees.append(tree)
    except:
        # mode debug
        failure.append(img)


# Version 2
#trees = map(images, image_pipeline_no_frame(img))

# Version 3 
#trees = pmap(images, image_pipeline_no_frame(img), nb_cpu=4)


