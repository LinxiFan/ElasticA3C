"""
Image utils
"""
import os
import sys
import numpy as np
from PIL import Image
from collections import namedtuple
try:
    import cv2
except ImportError:
    print('Please install OpenCV first. `pip install opencv-python` or '
          '`yes | conda install -c https://conda.binstar.org/menpo opencv3` '
          'if you use Anaconda. Make sure `import cv2` works after installation.', 
          file=sys.stderr)
    sys.exit(1)

def is_PIL(img):
    return isinstance(img, Image.Image)


def is_np(img):
    return isinstance(img, np.ndarray)


def np2PIL(img):
    "Numpy array to PIL.Image"
    return Image.fromarray(img)


def PIL2np(img):
    "PIL.Image to numpy array"
    assert is_PIL(img)
    print(img.size)
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def load_img(filename, format='np'):
    assert os.path.exists(filename), 'image file {} does not exist'.format(filename)
    format = format.lower()
    if format == 'np':
        return cv2.imread(filename)
    elif format == 'pil':
        return Image.open(filename)
    else:
        raise ValueError('format must be either "np" or "PIL"')


def save_img(img, filename):
    "Save a numpy or PIL image to file"
    if isinstance(img, Image.Image):
        img.save(filename)
    else:
        cv2.imwrite(filename, img)


def get_np_img(obj):
    if isinstance(obj, str):
        return load_img(obj, 'np')
    elif is_PIL(obj):
        return PIL2np(obj)
    elif is_np(obj):
        return obj
    else:
        raise ValueError('{} must be string (filename), ndarray, or PIL.Image'
                         .format(obj))
        
"Bounding box"
Box = namedtuple('Box', ['x', 'y', 'w', 'h'])

        
def crop(img, box):
    """
    Crop a numpy image with bounding box (x, y, w, h)
    """
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def draw_rect(img, box):
    "Draw a red bounding box"
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)


def draw_text(img, text, box, color='bw'):
    """
    FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_PLAIN
    FONT_HERSHEY_SCRIPT_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_TRIPLEX
    FONT_ITALIC
    """
    x, y, w, h = box
    font = cv2.FONT_HERSHEY_DUPLEX
    region = crop(img, box)
    if color == 'bw':
        brightness = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
        if brightness > 127:
            font_color = (0,0,0)
        else:
            font_color = (255,255,255)
    elif color == 'color':
        mean_bg = np.round(np.mean(region, axis=(0, 1)))
        font_color = tuple(map(int, np.array((255,255,255)) - mean_bg))
    else:
        font_color = (255, 0, 0) # blue

    cv2.putText(img, text, (x, y+h), font, 1, font_color, 2)
    

def disp(img, pause=True):
    "Display an image"
    save_img(img, '_temp.png')
    os.system('open _temp.png')
    if pause:
        input('continue ...')
        