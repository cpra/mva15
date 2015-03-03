
'''
Segment a PCB into foreground and background.
Based on OpenCV's grapcut example.
~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

USAGE:
  python2 mask_pcb.py <filename> <scalefactor>
    filename: path to pcb image
    scalefactor: scale factor to apply for faster processing (2 = half size)

Two windows will show up, one for input and one for output.

At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the resulting mask
'''

import os.path
import numpy as np
import cv2
import sys


# configure

BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

# setting up flags
rect = (0, 0, 1, 1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness


def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over

    # draw rectangle

    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle is True:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over is False:
            print("first draw rectangle")
        else:
            drawing = True
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing is True:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)


# print documentation
print(__doc__)

# parse args

filename = None
scalefactor = 1

if len(sys.argv) == 3:
    filename = sys.argv[1]
    scalefactor = int(sys.argv[2])
else:
    print("Usage: python2 pcbmask.py <filename> <scalefactor>")
    sys.exit(1)

# load image and setup

img = cv2.imread(filename)
origsize = img.shape[:2]
img = cv2.resize(img, (img.shape[1]/scalefactor, img.shape[0]/scalefactor))

img2 = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)
output = np.zeros(img.shape, np.uint8)
res = np.zeros(img.shape[:2], dtype=np.uint8)

cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.namedWindow('mask')
cv2.setMouseCallback('input', onmouse)
cv2.moveWindow('mask', 1921, 0)

# enter run loop

while(1):
    res = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    res[res > 0] = 255

    [cnt, _] = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(cnt) > 0:
        cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[0]

        res = np.zeros(res.shape, dtype=np.uint8)
        cv2.fillPoly(res, [cnt], 255)

    # draw and wait for input
    cv2.imshow('output', output)
    cv2.imshow('input', img)
    cv2.imshow('mask', res)
    k = 0xFF & cv2.waitKey(1)

    # key bindings
    if k == 27:         # esc to exit
        break
    elif k == ord('0'):  # BG drawing
        print(" mark background regions with left mouse button")
        value = DRAW_BG
    elif k == ord('1'):  # FG drawing
        print(" mark foreground regions with left mouse button")
        value = DRAW_FG
    elif k == ord('2'):  # PR_BG drawing
        value = DRAW_PR_BG
    elif k == ord('3'):  # PR_FG drawing
        value = DRAW_PR_FG
    elif k == ord('s'):  # save image
        res2 = cv2.resize(res, (origsize[1], origsize[0]), interpolation=cv2.INTER_NEAREST)
        resname = os.path.join(os.path.dirname(filename), '{}-mask.png'.format(os.path.splitext(os.path.basename(filename))[0]))
        cv2.imwrite(resname, res2)
        print(" result saved as image")
    elif k == ord('r'):  # reset everything
        print("resetting")
        rect = (0, 0, 1, 1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG
        img = img2.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        output = np.zeros(img.shape, np.uint8)           # output image to be shown
        res = np.zeros(img.shape[:2], dtype=np.uint8)
    elif k == ord('n'):  # segment the image
        if (rect_or_mask == 0):         # grabcut with rect
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1
        elif rect_or_mask == 1:         # grabcut with mask
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    output = cv2.bitwise_and(img2, img2, mask=mask2)

cv2.destroyAllWindows()
