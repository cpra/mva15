
'''
Annotate a PCB image.
~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology
'''

import cv2
import numpy as np
from PyQt4 import QtGui, QtCore

import sys
import argparse
import os.path


class Annot:

    def __init__(self, rect, text=''):
        if rect is None:
            rect = ((0, 0), (0, 0), 0)

        self.rect = rect
        self.text = text

    def __repr__(self):
        return 'Annotation {} "{}"'.format(self.rect, self.text)


def parse_annotation_file(path):
    lines = None
    with open(path) as f:
        lines = [l.strip().split() for l in f.readlines()]

    ret = []
    for l in lines:
        l = [x.strip() for x in l]
        if len(l) < 5:
            raise Exception('Failed to parse line "{}"'.format(l))

        rect = [float(s) for s in l[:5]]
        text = '' if len(l) == 5 else ' '.join(l[5:])
        ret.append(Annot((tuple(rect[0:2]), tuple(rect[2:4]), rect[4]), text))

    return ret


def write_annotation_file(data, to):
    with open(to, 'w') as f:
        for d in data:
            f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f} {}\n'.format(d.rect[0][0], d.rect[0][1], d.rect[1][0], d.rect[1][1], d.rect[2], d.text))


def mat_to_qimage(mat):
    tmp = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    h, w, bpc = tmp.shape
    return QtGui.QImage(tmp.data, w, h, bpc*w, QtGui.QImage.Format_RGB888)


class MouseLabel(QtGui.QLabel):

    def __init(self, parent):
        QtGui.QLabel.__init__(self, parent)

    def mouseReleaseEvent(self, ev):
        self.emit(QtCore.SIGNAL('clicked(int, int, int)'), ev.x(), ev.y(), ev.button())


class Window(QtGui.QWidget):

    def __init__(self, image_path, image, annot_path, annot):
        self.image_path = image_path
        self.image = image.copy()
        self.annot_path = annot_path
        self.annot = annot

        self.annot_coords = [[0, 0], [0, 0], [0, 0], [0, 0]]  # x;y
        self.annot_idx = 0

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('PCB Annotation')
        self.setGeometry(100, 100, 1200, 1000)

        self.label_image = MouseLabel(self)
        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setWidget(self.label_image)
        self.scroll.setGeometry(0, 0, 1200, 1000)

        self.connect(self.label_image, QtCore.SIGNAL('clicked(int, int, int)'), self.point_selected)

    def point_selected(self, x, y, button):
        if button != 1 and button != 2:
            return

        if button == 1:
            # new annotation point

            self.annot_coords[self.annot_idx][0] = x
            self.annot_coords[self.annot_idx][1] = y

            self.annot_idx += 1
            if self.annot_idx == 4:
                rect = cv2.minAreaRect(np.array(self.annot_coords))
                self.annot.append(Annot(rect))

                self.annot_idx = 0

        if button == 2:
            # remove existing annotation

            id = 0
            del_id = -1
            for an in self.annot:
                tmp = np.zeros(self.image.shape[:2], dtype=np.uint8)
                bp = cv2.cv.BoxPoints(an.rect)
                bp = np.int0(bp)
                cv2.drawContours(tmp, [bp], 0, (255), -1)

                if tmp[y, x] > 0:
                    del_id = id
                    break

                id += 1

            if del_id >= 0:
                del self.annot[del_id]

        self.redraw()

    def redraw(self):
        vis = self.image.copy()
        id = 1
        for an in self.annot:
            bp = cv2.cv.BoxPoints(an.rect)
            bp = np.int0(bp)
            cv2.drawContours(vis, [bp], 0, (0, 255, 0), 2)

            cv2.putText(vis, '{}'.format(id), (bp[0, 0]+5, bp[0, 1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            id += 1

        for idx in np.arange(self.annot_idx):
            cv2.line(vis, tuple(self.annot_coords[idx]), tuple(self.annot_coords[idx]), (0, 0, 255), 2)

        self.label_image.setPixmap(QtGui.QPixmap.fromImage(mat_to_qimage(vis)))
        self.label_image.setGeometry(QtCore.QRect(0, 0, vis.shape[1], vis.shape[0]))

    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'Save?', 'Save changes to file?', QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            print('Writing annotation data to "{}"'.format(self.annot_path))
            write_annotation_file(self.annot, self.annot_path)

        event.accept()


parser = argparse.ArgumentParser(description='Annotate PCB.')
parser.add_argument('file', type=str, help='Path to image file')
args = parser.parse_args()

if args.file is None:
    parser.print_help()
    sys.exit(1)

if not os.path.isfile(args.file):
    sys.exit('Specified file does not exist.')

if os.path.splitext(args.file)[1] != '.jpg':
    sys.exit('Only .jpg files are supported')

img = cv2.imread(args.file)

annot_name = os.path.join(os.path.dirname(args.file), '{}-annot.txt'.format(os.path.splitext(args.file)[0]))
annot_data = []

if os.path.isfile(annot_name):
    print('Loading existing annotation file')
    annot_data = parse_annotation_file(annot_name)

app = QtGui.QApplication(sys.argv)
win = Window(args.file, img, annot_name, annot_data)
win.redraw()
win.show()

sys.exit(app.exec_())
