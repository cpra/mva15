
'''
Python2 API for the PCB DSLR dataset.
~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology
'''

import cv2
import numpy as np

import os
import os.path
import argparse


class Annot:

    '''
    An annotated PCB component.
    '''

    def __init__(self, rect, scale, text=''):
        '''
        Constructor.
        rect: region of the component specified by ((cx, cy), (dx, dy), angle) as in OpenCV.
        scale: scale factor (affects size_pixels() and size_cm2()).
        text: optional label text
        '''

        self.rect = rect
        self._scale = scale
        self.text = text

    def __repr__(self):
        '''
        Returns a string representation.
        '''

        return 'Annotation {} "{}"'.format(self.rect, self.text)

    def size_pixels(self, scaled=True):
        '''
        Returns the size of the component in pixels.
        scaled: whether to regard the scale factor.
        '''

        sz = self.rect[1][0]*self.rect[1][1]
        if not scaled:
            sz /= self._scale

        return sz

    def size_cm2(self, scaled=True):
        '''
        Returns the size of the component in cm^2.
        scaled: whether to regard the scale factor.
        '''

        sz = (self.rect[1][0]/87.4) * (self.rect[1][1]/87.4)
        if not scaled:
            sz /= self._scale

        return sz

    def aspect(self):
        '''
        Returns the aspect ratio (larger side length / smaller side length).
        '''

        return max(self.rect[1][0], self.rect[1][1]) / min(self.rect[1][0], self.rect[1][1])


class PCB:

    '''
    A printed circuit board.
    '''

    def __init__(self, root, scale=1):
        '''
        Constructor.
        root: root directory path.
        scale: scale factor (1 = original size).
        '''

        if not os.path.isdir(root):
            raise Exception('Path "{}" is not a directory'.format(root))

        if scale <= 0 or scale > 2:
            raise Exception('Scale must be > 0 and <= 2')

        self._root = root
        self._scale = scale
        self._recordings = [int(os.path.splitext(p)[0][3:]) for p in os.listdir(root) if p.startswith('rec') and p.endswith('.jpg') and 'mask' not in p]
        self._cache_cropinfo = {}

    def __repr__(self):
        '''
        Returns a string representation.
        '''

        return 'PCB {} ({} recordings)'.format(self.id(), len(self._recordings))

    def id(self):
        '''
        Returns the PCB ID.
        '''

        return int(os.path.splitext(os.path.basename(self._root))[0][3:])

    def recordings(self):
        '''
        Returns a list of IDs of all available recordings.
        '''

        return self._recordings

    def image(self, rec=1):
        '''
        Returns the image of the specified recording.
        rec: desired recording (see recordings()).
        '''

        if rec not in self._recordings:
            raise Exception('Recording {} does not exist for this PCB'.format(rec))

        im = cv2.imread(os.path.join(self._root, 'rec{}.jpg'.format(rec)), cv2.IMREAD_UNCHANGED)
        if im.size == 0:
            raise Exception('Could not load the image')

        if self._scale != 1:
            im = cv2.resize(im, (0, 0), im, self._scale, self._scale)

        return im

    def mask(self, rec=1):
        '''
        Returns the mask of the specified recording.
        rec: desired recording (see recordings()).
        '''

        if rec not in self._recordings:
            raise Exception('Recording {} does not exist for this PCB'.format(rec))

        im = cv2.imread(os.path.join(self._root, 'rec{}-mask.png'.format(rec)), cv2.IMREAD_GRAYSCALE)
        if im.size == 0:
            raise Exception('Could not load the mask')

        if self._scale != 1:
            im = cv2.resize(im, (0, 0), im, self._scale, self._scale)

        return im

    def image_masked(self, rec=1):
        '''
        Returns the image of the specified recording, masked by the corresponding mask and cropped to remove background.
        rec: desired recording (see recordings()).
        '''

        im = self.image(rec)
        mask = self.mask(rec)

        im[mask == 0, :] = 0
        ci = self._cropinfo(rec)

        return im[ci[1]:ci[1]+ci[3], ci[0]:ci[0]+ci[2]]

    def ics(self, rec=1, cropped=False, size=(0, 0), aspect=(0, 0)):
        '''
        Returns a list of IC chips as a list of Annot objects.
        rec: desired recording (see recordings()).
        cropped: whether to return coordinates for cropped images (see image_masked()).
        size: (min, max) size of returned ICs in cm^2, disregarding the scale factor (0 = all).
        aspect: (min, max) aspect ratio of returned ICs (0 = all).
        '''

        if rec not in self._recordings:
            raise Exception('Recording {} does not exist for this PCB'.format(rec))

        fpath = os.path.join(self._root, 'rec{}-annot.txt'.format(rec))
        if not os.path.isfile(fpath):
            raise Exception('"{}" is not a file'.format(fpath))

        lines = None
        with open(fpath) as f:
            lines = [l.strip().split() for l in f.readlines()]

        ret = []
        for l in lines:
            l = [x.strip() for x in l]
            if len(l) < 5:
                raise Exception('Failed to parse line "{}"'.format(l))

            rect = [float(s) for s in l[:5]]
            text = '' if len(l) == 5 else ' '.join(l[5:])

            sz = (rect[2]/87.4, rect[3]/87.4)
            asp = max(sz[0], sz[1]) / min(sz[0], sz[1])
            sz = sz[0]*sz[1]

            if size[0] > 0 and sz < size[0]:
                continue

            if size[1] > 0 and sz > size[1]:
                continue

            if aspect[0] > 0 and asp < aspect[0]:
                continue

            if aspect[1] > 0 and asp > aspect[1]:
                continue

            if self._scale != 1:
                rect[0] *= self._scale
                rect[1] *= self._scale
                rect[2] *= self._scale
                rect[3] *= self._scale

            if cropped:
                ci = self._cropinfo(rec)
                rect[0] -= ci[0]
                rect[1] -= ci[1]

            ret.append(Annot((tuple(rect[0:2]), tuple(rect[2:4]), rect[4]), self._scale, text))

        return ret

    def _cropinfo(self, rec):
        '''
        Return (and cache) information for auto cropping a PCB image.
        rec: desired recording (see recordings()).
        '''

        if rec in self._cache_cropinfo:
            return self._cache_cropinfo[rec]

        im = self.mask(rec)
        cnt, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(cnt) > 1:  # use largest region if there are multiple
            def cntsz(c):
                rr = cv2.minAreaRect(c)
                return rr[1][0]*rr[1][1]

            cnt = sorted(cnt, key=cntsz, reverse=True)

        cx, cy, cw, ch = cv2.boundingRect(cnt[0])
        self._cache_cropinfo[rec] = (cx, cy, cw, ch)

        return self._cache_cropinfo[rec]


class PCBDataset:

    '''
    A PCB dataset.
    '''

    def __init__(self, root):
        '''
        Constructor.
        root: root path to the dataset.
        '''

        if not os.path.isdir(root):
            raise Exception('Path "{}" is not a directory'.format(root))

        self._root = root

        self._pcb_paths = {}
        for p in [os.path.join(root, f) for f in os.listdir(root) if f.startswith('pcb') and os.path.isdir(os.path.join(root, f))]:
            id = int(os.path.splitext(os.path.basename(p))[0][3:])
            self._pcb_paths[id] = p

    def num_pcbs(self):
        '''
        Returns the number of PCBs in the dataset.
        '''

        return len(self._pcb_paths)

    def pcb_ids(self):
        '''
        Returns a sorted list of IDs of all PCBs in the dataset.
        '''

        return sorted(list(self._pcb_paths.keys()))

    def pcb(self, id, scale=1):
        '''
        Returns the PCB with the given ID as a PCB object.
        id: PCB id (see pcb_ids()).
        scale: scale factor (1 = original size).
        '''

        if id not in self._pcb_paths:
            raise Exception('Unknown PCB ID')

        return PCB(self._pcb_paths[id], scale)

    def pcbs(self, scale=1):
        '''
        Generator function for returning all PCBs in the dataset as PCB objects.
        scale: scale factor (1 = original size).
        '''

        for id in self._pcb_paths:
            yield PCB(self._pcb_paths[id], scale)


# a simple visualizer to demonstrate the API

if __name__ == "__main__":
    # parse args

    def minmax(s):
        try:
            mi, ma = map(float, s.split(','))
            return mi, ma
        except:
            raise argparse.ArgumentTypeError('min/max syntax is min,max')

    parser = argparse.ArgumentParser(description='Visualize a dataset image')
    parser.add_argument('--root', type=str, dest='root', required=True, help='Path to the dataset')
    parser.add_argument('--pcb', type=int, dest='pcb', default=1, help='ID of the PCB to show')
    parser.add_argument('--rec', type=int, dest='rec', default=1, help='Recording to show')
    parser.add_argument('--scale', type=float, dest='scale', default=1, help='Scale factor')
    parser.add_argument('--icsz', type=minmax, dest='icsz', default='0,0', help='(min, max) size of returned ICs in cm^2 (0 = no restriction)')
    parser.add_argument('--icas', type=minmax, dest='icas', default='0,0', help='(min, max) aspect ratio of returned ICs (0 = no restriction)')
    args = parser.parse_args()

    # load data and show it

    db = PCBDataset(args.root)
    print('Dataset contains images of {} PCBs'.format(db.num_pcbs()))
    print(' {}'.format(db.pcb_ids()))

    pcb = db.pcb(args.pcb, args.scale)
    print('Loaded PCB {}, available recordings: {}'.format(args.pcb, pcb.recordings()))

    ics = pcb.ics(args.rec, True, args.icsz, args.icas)
    print('PCB contains {} ICs'.format(len(ics)))

    img = pcb.image_masked(args.rec)

    for ic in ics:
        bp = cv2.cv.BoxPoints(ic.rect)
        bp = np.int0(bp)
        cv2.drawContours(img, [bp], 0, (0, 255, 0), 2)

    cv2.imshow('PCB', img)
    cv2.waitKey(0)
