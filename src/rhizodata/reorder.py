"""

"""
from path import Path
from collections import OrderedDict

# Parent directory containing the image directories
# In my case Philippe_image_4_boites

class DataEnv(object):
    "Main class to manage the image data"
    def __init__(self, input_dir, output_dir,
                 img_pattern, simulate):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_pattern = img_pattern
        self.simulate = simulate






def my_cmp(x, y):
    index= lambda x: int(x.name.split('Sequence')[-1].strip())
    return cmp(index(x), index(y))

def get_seq_dirs(d):
    """ Return the directories sorted by date.
    """
    image_dir = d
    dirs = d.glob(sequence_dir_name)

    dirs = sorted(dirs, cmp=my_cmp)
    print([x.name for x in dirs])

    return dirs

#dirs = get_seq_dirs()

def box_number(filename):
    name = filename.name
    record = name.split('_')
    box = [t for t in record if t.startswith('Boite')][0]
    box_number = int(box.split()[-1])
    return box_number

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


def save_all(d, output_dir=Path.getcwd()):
    image_dir = d
    images = reordering(d)
    print(images.keys())

    cwd = output_dir
    for b in images:
        dname = target_name+'_%03d'%(b)
        _d = cwd/dname
        print('mkdir %s'%(_d))
        if not _d.isdir():
            _d.mkdir()
            print('Mkdir ', _d)
        for f in images[b]:
            print('Move file %s to %s'%(f, _d/f.name))
            f.copyfile(_d/f.name)

    print('SUCESS !!!')


def save_manips(input='Christophe', output='result'):
    _input = Path(input)
    manips = _input.glob('manip*')
    output = Path(output)

    if not output.isdir():
        output.mkdir()

    for manip in manips:
        mn = manip.name
        res_manip = output/mn
        if not res_manip.isdir():
            res_manip.mkdir()

        save_all(manip, output_dir=res_manip)

if __name__ == '__main__':

    d = Path('.')
    l= d.glob('Phili*')
    l= d.glob('Christ*')
    directory_base = image_dir = l[0]

    img_pattern = 'Philippe*.jpg'
    img_pattern = 'ALEX*.jpg'

    # Either Sequence or Visualisation
    sequence_dir_name = 'Seq*'
    sequence_dir_name = 'Visu*'

    target_name = 'VisuBoite'

