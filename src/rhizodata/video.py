import cv2
import os
from path import Path

name = 'result/manip1/VisuBoite_005'
image_folder = name
video_name = name+'.mp4'

def my_cmp(x, y):
    def index(x):
        seq = x.name.split('_Seq')[1]
        boite = (seq.split('_Boite')[0]).strip()
        _id = int(boite)
        return _id
    return cmp(index(x), index(y))

def get_images(d):
    _dir = Path(d)
    images = [_dir/img for img in os.listdir(d) if img.endswith(".jpg")]
    images = sorted(images, cmp=my_cmp)
    return images


def video(images, video_name, ratio=10):
    print(images[0])
    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    dim = (int(width/ratio), int(height/ratio))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _video = cv2.VideoWriter(video_name, fourcc, 10., dim)

    for image in images:
        frame = cv2.imread(str(image))
        resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        _video.write(resized_frame)

    cv2.destroyAllWindows()
    _video.release()

def video_manips(d='result'):
    result = Path(d)
    manips = result.glob('manip*')

    for manip in manips:
        manip_name = str(manip.name)
        for box in manip.glob('VisuBoite_*'):
            if not box.isdir():
                continue
            video_name = 'result/%s_%s.mp4'%(manip_name, box.name)

            _images = get_images(box)
            #print _images
            video(images=_images, video_name=video_name)
            print('Generate ', video_name)

if __name__ == '__main__':


    images = get_images()
