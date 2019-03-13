"""

"""
from path import Path
from collections import OrderedDict

# Parent directory containing the image directories
# In my case Philippe_image_4_boites

def my_cmp(x, y):
    "Sort the sequence directories by there number"
    index= lambda x: int(x.name.split('Sequence')[-1].strip())
    return cmp(index(x), index(y))


def box_number(filename):
    "Return the box number based on its filename."
    name = filename.name
    record = name.split('_')
    box = [t for t in record if t.startswith('Boite')][0]
    box_number = int(box.split()[-1])
    return box_number


class DataEnv(object):
    "Main class to manage the image data"
    def __init__(self, input_dir, output_dir,
                 img_pattern, simulate):
        self.input_dir = Path(input_dir).abspath()
        self.output_dir = Path(output_dir).abspath()
        self.img_pattern = img_pattern
        self.simulate = simulate

        # These directories have the following semantic.
        # Sequence* are directories that contain full size images.
        # Visualization Seq* contain compressed images.
        self.sequence_dir_name = 'Seq*'
        self.visu_dir_name = 'Visu*'

        self.status= self.check()


    def check(self):
        status = 0
        if not self.input_dir.exists():
            print('WARNING : The input directory %s does not exist'%self.input_dir)
            status = 1

        if self.output_dir.exists():
            print('WARNING : The output directory %s already exist'%self.output_dir)
            print('WARNING : Move or Remove it and launch the program again')
            status = 1

        seqs = self.input_dir.glob(self.sequence_dir_name)
        if not seqs:
            print('WARNING : No Sequence directories in %s'%self.input_dir)
            status = 1

        visus = self.input_dir.glob(self.visu_dir_name)
        if not visus:
            print('WARNING : No Visualization directories in %s'%self.input_dir)
            status = 1

        return status


    def process(self):
        """ Create the output directories.
        And sort the images in the good directory.
        """

        run = not self.simulate

        # Create the output directory
        if not self.output_dir:
            print("mkdir %s"%str(self.output_dir))
            if run:
                self.output_dir.makedirs()

        s_dirs = self.get_seq_dirs(self.sequence_dir_name)
        s_dict = self.reordering(s_dirs)
        self.save_images(s_dict, target_name='Boite')

        v_dirs = self.get_seq_dirs(self.visu_dir_name)
        v_dict = self.reordering(v_dirs)
        self.save_images(v_dict, target_name='Visualisation_Boite')


    def get_seq_dirs(self, sequence_dir_name):
        """ Return the directories sorted by date.
        """
        image_dir = self.input_dir
        dirs = image_dir.glob(sequence_dir_name)

        dirs = sorted(dirs, cmp=my_cmp)
        print("List of image dirs: ",)
        print([x.name for x in dirs])

        return dirs


    def reordering(self, dirs):
        """ Group images by box rather than time or sequence."""

        boxes = sorted(list(set(box_number(f) for _d in dirs for f in _d.glob(self.img_pattern))))

        result = OrderedDict()
        for b in boxes:
            result[b] = []

        for _d in dirs:
            for f in _d.glob(self.img_pattern):
                result[box_number(f)].append(f)

        print(result)
        return result

    def save_images(self, images_by_boxes, target_name='Boite'):
        images = images_by_boxes
        print(images.keys())

        run = not self.simulate

        cwd = self.output_dir
        for b in images:
            dname = target_name+'_%03d'%(b)
            _d = cwd/dname
            print('mkdir %s'%(_d))
            if not _d.isdir():
                if run:
                    _d.makedirs()
                print('Mkdir ', _d)
            for f in images[b]:
                print('Move file %s to %s'%(f, _d/f.name))
                if run:
                    f.copyfile(_d/f.name)

        print('SUCESS !!!')

#dirs = get_seq_dirs()



def reordering(d):
    image_dir = d
    dirs = get_seq_dirs(d)
    boxes = sorted(list(set(box_number(f) for _d in dirs for f in _d.glob(img_pattern))))

    result = OrderedDict()
    for b in boxes:
        result[b] = []

    for _d in dirs:
        for f in _d.glob(img_pattern):
            result[box_number(f)].append(f)

    print(result)
    return result


def save_manips(input='Christophe', output='result'):
    _input = Path(input)
    manips = _input.glob('manip*')
    output = Path(output)

    if not output.isdir():
        output.makedirs()

    for manip in manips:
        mn = manip.name
        res_manip = output/mn
        if not res_manip.isdir():
            res_manip.makedirs()

        save_all(manip, output_dir=res_manip)


